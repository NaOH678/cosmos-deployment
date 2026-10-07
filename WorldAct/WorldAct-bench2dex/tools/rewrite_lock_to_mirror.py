#!/usr/bin/env python3
"""Rewrite artifact URLs in a uv.lock to an internal package mirror.

`uv sync --frozen` downloads the absolute artifact URLs recorded in ``uv.lock``;
`--default-index` / `UV_INDEX_URL` are ignored in that mode. So to install from an
internal mirror while keeping the lock's exact versions, rewrite the URLs in the lock
itself.

Only URLs change. Versions and ``sha256`` hashes are left untouched, so uv still
verifies every downloaded artifact against the committed lock: a bad mapping fails
loudly with a hash mismatch instead of silently installing something else.

Usage::

    python tools/rewrite_lock_to_mirror.py uv.lock --verify-sample 8
    uv sync --all-extras --group=cu130-train --frozen
    git checkout -- uv.lock     # restore the committed lock; it is git-tracked

See docs/setup_offline.md for the full procedure and the mirror's coverage limits.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import random
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

# Public artifact hosts -> the internal repo that carries them. Hosts absent from this
# map (nvidia-cosmos.github.io, github.com) have no internal equivalent and are left
# pointing at the public URL.
DEFAULT_HOST_REPOS = {
    "files.pythonhosted.org": "official-pypi-proxy",
    "pypi.org": "official-pypi-proxy",
    "download.pytorch.org": "pypi-pytorch",
    "download-r2.pytorch.org": "pypi-pytorch",
}

_NAME_RE = re.compile(r'^name = "([^"]+)"', re.M)
_BLOCK_RE = re.compile(r"(?=^\[\[package\]\])", re.M)
# Artifact URLs only: a trailing file extension keeps registry identifiers such as
# `source = { registry = "https://pypi.org/simple" }` out of the rewrite.
_ARTIFACT_RE = re.compile(
    r"(https://(?:files\.pythonhosted\.org|pypi\.org|download\.pytorch\.org|download-r2\.pytorch\.org)"
    r'/[^"\']*?\.(?:whl|tar\.gz|zip|tar\.bz2))(?=")'
)
_REWRITTEN_RE = re.compile(r'url = "([^"]+\.(?:whl|tar\.gz))", hash = "sha256:([a-f0-9]{64})"')


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lock", type=Path, nargs="?", default=Path("uv.lock"), help="lockfile to rewrite in place")
    parser.add_argument(
        "--mirror-base",
        default="http://pkg.pjlab.org.cn/repository",
        help="base URL of the internal Nexus",
    )
    parser.add_argument("--pypi-repo", default="official-pypi-proxy", help="repo on the mirror carrying PyPI")
    parser.add_argument("--pytorch-repo", default="pypi-pytorch", help="repo on the mirror carrying torch wheels")
    parser.add_argument("--jobs", type=int, default=24, help="parallel index fetches")
    parser.add_argument("--dry-run", action="store_true", help="report what would change without writing")
    parser.add_argument(
        "--verify-sample",
        type=int,
        default=0,
        metavar="N",
        help="after rewriting, download N rewritten artifacts and check them against the lock hashes",
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling seed for --verify-sample")
    return parser.parse_args()


def _fetch_index(base: str, repo: str, package: str, timeout: float = 60.0) -> dict[str, str]:
    """Return {filename: url} for one package, as listed by the mirror's /simple/ page."""
    index_url = f"{base}/{repo}/simple/{package}/"
    try:
        with urllib.request.urlopen(index_url, timeout=timeout) as response:
            html = response.read().decode("utf-8", "replace")
    except Exception:
        return {}

    out: dict[str, str] = {}
    for href in re.findall(r'href="([^"]+)"', html):
        href = href.split("#")[0]
        if not href:
            continue
        # Resolve relative to /simple/<package>/, then keep only links that stay on the
        # mirror; absolute external links are dropped.
        url = urllib.parse.urljoin(index_url, href)
        if not url.startswith(base + "/"):
            continue
        out.setdefault(urllib.parse.unquote(url.rsplit("/", 1)[-1]), url)
    return out


def _collect(blocks: list[str], host_repos: dict[str, str]) -> dict[tuple[str, str], set[str]]:
    """(package, repo) -> filenames referenced by that package's URLs."""
    wanted: dict[tuple[str, str], set[str]] = {}
    for block in blocks:
        match = _NAME_RE.search(block)
        if not match:
            continue
        package = match.group(1)
        for url in _ARTIFACT_RE.findall(block):
            repo = host_repos.get(urllib.parse.urlparse(url).netloc)
            if repo:
                wanted.setdefault((package, repo), set()).add(urllib.parse.unquote(url.rsplit("/", 1)[-1]))
    return wanted


def _build_map(wanted: dict[tuple[str, str], set[str]], base: str, jobs: int) -> tuple[dict, list]:
    """Resolve every referenced filename to its mirror URL; return (map, missing)."""
    indexes: dict[tuple[str, str], dict[str, str]] = {}
    with futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        pending = {pool.submit(_fetch_index, base, repo, pkg): (pkg, repo) for pkg, repo in wanted}
        for done, future in enumerate(futures.as_completed(pending), 1):
            indexes[pending[future]] = future.result()
            if done % 100 == 0:
                print(f"  fetched {done}/{len(pending)} package indexes", flush=True)

    rewrites: dict[tuple[str, str], str] = {}
    missing: list[tuple[str, str]] = []
    for key, filenames in wanted.items():
        table = indexes.get(key, {})
        for filename in filenames:
            url = table.get(filename)
            if url:
                rewrites[(key[0], filename)] = url
            else:
                missing.append((key[0], filename))
    return rewrites, missing


def _apply(blocks: list[str], rewrites: dict[tuple[str, str], str]) -> tuple[str, int]:
    """Replace URLs we resolved; return (new text, count of replacements)."""
    count = 0

    def substitute(match: re.Match) -> str:
        nonlocal count
        url = match.group(1)
        filename = urllib.parse.unquote(url.rsplit("/", 1)[-1])
        replacement = rewrites.get((package, filename))
        if replacement:
            count += 1
            return replacement
        return url

    out = []
    for block in blocks:
        match = _NAME_RE.search(block)
        if match:
            package = match.group(1)
            block = _ARTIFACT_RE.sub(substitute, block)
        out.append(block)
    return "".join(out), count


def _verify(text: str, base: str, sample: int, seed: int) -> int:
    """Download `sample` rewritten artifacts and check them against the lock's hashes."""
    pairs = [(u, h) for u, h in _REWRITTEN_RE.findall(text) if u.startswith(base + "/")]
    if not pairs:
        print("nothing rewritten to verify")
        return 0
    random.seed(seed)
    failures = 0
    for url, expected in random.sample(pairs, min(sample, len(pairs))):
        name = url.rsplit("/", 1)[-1]
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                actual = hashlib.sha256(response.read()).hexdigest()
        except Exception as exc:
            print(f"  ERR  {name}: {type(exc).__name__}: {exc}")
            failures += 1
            continue
        if actual == expected:
            print(f"  OK   {name}")
        else:
            print(f"  BAD  {name}\n       expected {expected}\n       actual   {actual}")
            failures += 1
    return failures


def main() -> int:
    args = _parse_args()
    host_repos = {
        host: (args.pytorch_repo if "pytorch" in repo else args.pypi_repo) for host, repo in DEFAULT_HOST_REPOS.items()
    }

    text = args.lock.read_text(encoding="utf-8")
    blocks = _BLOCK_RE.split(text)
    named = [b for b in blocks if _NAME_RE.search(b)]
    print(f"{args.lock}: {len(blocks)} blocks, {len(named)} packages")

    wanted = _collect(blocks, host_repos)
    print(f"looking up {len(wanted)} (package, repo) pairs on {args.mirror_base}")
    rewrites, missing = _build_map(wanted, args.mirror_base, args.jobs)

    print(f"\nrewritable artifacts : {len(rewrites)}")
    print(f"not on the mirror    : {len(missing)}")
    for package, filename in missing[:10]:
        print(f"    {package}: {filename}")
    if len(missing) > 10:
        print(f"    ... and {len(missing) - 10} more")

    if not rewrites:
        print("\nnothing to rewrite; left the lock untouched")
        return 0

    new_text, count = _apply(blocks, rewrites)
    print(f"\nURLs rewritten: {count}")

    if args.dry_run:
        print("--dry-run: not writing")
        return 0

    args.lock.write_text(new_text, encoding="utf-8")
    print(f"wrote {args.lock}")
    print("restore afterwards with: git checkout -- uv.lock")

    if args.verify_sample:
        print(f"\nverifying {args.verify_sample} rewritten artifacts against the lock hashes:")
        failures = _verify(new_text, args.mirror_base, args.verify_sample, args.seed)
        if failures:
            print(f"\n{failures} verification failure(s) -- restore the backup before syncing")
            return 1
        print("all sampled artifacts match the lock")
    return 0


if __name__ == "__main__":
    sys.exit(main())

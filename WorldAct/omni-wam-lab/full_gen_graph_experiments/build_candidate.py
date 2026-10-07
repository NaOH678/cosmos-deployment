"""Build isolated source with full GEN-stack graph dispatch, no production edits."""

import json
import os
from pathlib import Path

root = Path(__file__).resolve().parent
lab = root.parent
source = Path(json.loads((lab / "upstream.json").read_text())["source"])
out = root / "source"
for directory, dirs, files in os.walk(source):
    dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
    dest = out / Path(directory).relative_to(source)
    dest.mkdir(parents=True, exist_ok=True)
    for name in files:
        p = dest / name
        if not p.exists():
            p.symlink_to(Path(directory) / name)
rel = Path("vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py")
s = (source / rel).read_text()
old = "    def _run_gen_stack(self, prep: _GenPrepared) -> torch.Tensor:\n"
new = """    def _run_gen_stack(self, prep: _GenPrepared) -> torch.Tensor:
        import os
        if os.environ.get("WAM_FULL_GEN_GRAPH") == "1" and prep.has_action:
            from full_gen_graph_impl import run_full_gen_graph
            return run_full_gen_graph(self, prep, self._run_gen_stack_original)
        return self._run_gen_stack_original(prep)

    def _run_gen_stack_original(self, prep: _GenPrepared) -> torch.Tensor:
"""
assert s.count(old) == 1
s = s.replace(old, new)
p = out / rel
if p.is_symlink():
    p.unlink()
p.write_text(s)
print(out)

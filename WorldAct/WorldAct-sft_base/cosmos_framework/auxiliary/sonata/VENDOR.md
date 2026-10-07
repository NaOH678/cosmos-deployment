# Vendored Sonata

Source: https://github.com/facebookresearch/sonata

Commit: `18c09ff8d713494f78a8213792262b910977a65d`

Retrieved: 2026-09-08. License: Apache-2.0, retained in LICENSE and source headers.

The upstream `sonata/` Python package is copied unchanged into this directory.
Import it as `cosmos_framework.auxiliary.sonata`; no separate global `sonata`
installation or changes to Python's module search path are required.
UPSTREAM_README.md is the original usage documentation (its examples use the
upstream `sonata` import name).

This is an optional geometry encoder dependency. Import it explicitly when needed;
it is not imported by the existing Cosmos training/inference entry points.
Dependencies were installed and tested in the shared Cosmos environment; see
`docs/sonata_environment_validation.md` for versions and installation commands.

Local validation entry point:

```bash
LD_LIBRARY_PATH='' .venv/bin/python -m cosmos_framework.scripts.validate_sonata \
  --checkpoint ../checkpoints/ptv3/sonata_small.pth --device cpu --seed 0
```

CPU mode checks strict checkpoint loading and preprocessing only. Use
`--device cuda --backward` on a GPU node for forward/backward validation.

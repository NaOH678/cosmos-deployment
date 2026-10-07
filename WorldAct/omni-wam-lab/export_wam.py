"""Export legacy WAM DCP with the native exporter; CPU only, EMA required."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["COSMOS_DEVICE"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["COSMOS_TRAINING"] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-only", action="store_true")
    args = parser.parse_args()
    import yaml
    from cosmos_framework.scripts import export_model as exporter
    from cosmos_framework.inference.common.args import ConfigArgs, CheckpointOverrides

    source = args.config.resolve()
    raw = yaml.safe_load(source.read_text())
    dataset = raw["dataloader_train"]["dataloader"]["datasets"]["singlerighthand"][
        "dataset"
    ]
    if dataset["mode"] != "wam" or dataset["max_action_dim"] != 64:
        raise ValueError("Expected native singlerighthand WAM configuration")
    if not raw["model"]["config"]["ema"]["enabled"]:
        raise ValueError("EMA must be enabled in source configuration")
    # The native exporter expects the newer typed dataloader schema. Adapt only
    # its metadata read for this legacy YAML; model configuration is untouched.
    metadata_dataset = dict(dataset, embodiment_type="singlerighthand")
    metadata = SimpleNamespace(
        dataloader_train=SimpleNamespace(
            dataloaders=SimpleNamespace(
                action_data=SimpleNamespace(
                    dataloader=SimpleNamespace(
                        dataset={"list_of_datasets": [{"dataset": metadata_dataset}]}
                    )
                )
            )
        )
    )
    original = ConfigArgs.load_config

    def load_config(self):
        if Path(self.config_file).resolve() == source:
            return metadata
        return original(self)

    ConfigArgs.load_config = load_config
    try:
        exporter.export_model(
            exporter.Args(
                checkpoint=CheckpointOverrides(
                    checkpoint_path=str(args.checkpoint.resolve()),
                    config_file=str(source),
                    use_ema_weights=True,
                ),
                output_dir=args.output.resolve(),
                config_only=args.config_only,
                vit=False,
                use_torch_compile=False,
                use_cuda_graphs=False,
            )
        )
    finally:
        ConfigArgs.load_config = original
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(source),
        "config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "ema": True,
        "config_only": args.config_only,
        "action_chunk_size": dataset["chunk_length"],
        "fps": dataset["fps"],
        "native_action_dim": 27,
        "domain_name": "singlerighthand",
    }
    (args.output / "wam_export.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

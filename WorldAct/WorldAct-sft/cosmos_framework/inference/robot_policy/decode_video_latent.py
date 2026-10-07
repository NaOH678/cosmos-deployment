"""Decode recorded vision latents offline using the matching frozen-config VAE."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("latent", type=Path)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--vae-path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15)
    args = parser.parse_args()
    import cv2
    import torch
    import yaml
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface
    config = yaml.safe_load(args.model_config.read_text())
    tokenizer = dict(config["model"]["config"]["tokenizer"])
    target = tokenizer.pop("_target_")
    if not target.endswith(".Wan2pt2VAEInterface"):
        raise ValueError(f"Unsupported VAE: {target}")
    if args.vae_path:
        tokenizer.update(vae_path=args.vae_path, bucket_name="")
    tokenizer["keep_decoder_cache"] = False
    latent = torch.load(args.latent, map_location="cpu", weights_only=True)
    if latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError(f"Expected single BCTHW latent, got {latent.shape}")
    if args.output.exists():
        raise FileExistsError(args.output)
    vae = Wan2pt2VAEInterface(**tokenizer)
    with torch.inference_mode():
        video = vae.decode(latent.cuda()).clamp(-1, 1)
        frames = ((video[0].permute(1, 2, 3, 0).float().cpu() + 1) * 127.5).round().to(torch.uint8).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                             (frames.shape[2], frames.shape[1]))
    if not writer.isOpened():
        raise RuntimeError("Video writer could not open output")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    print(json.dumps({"output": str(args.output), "frames": len(frames), "fps": args.fps}))

if __name__ == "__main__":
    main()

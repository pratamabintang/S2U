"""Verify zero-init RGB equivalence for the four-channel SAM2-UNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hiera_path", required=True)
    parser.add_argument("--model_cfg", default="sam2_hiera_l.yaml")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--output_json", default=None)
    return parser


def main(args: argparse.Namespace) -> None:
    if not Path(args.hiera_path).is_file():
        raise FileNotFoundError(args.hiera_path)
    from SAM2UNet import SAM2UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    torch.manual_seed(123)
    model = SAM2UNet(args.hiera_path, model_cfg=args.model_cfg).to(device).eval()
    rgb = torch.randn(1, 3, args.size, args.size, device=device)
    zero_dtm = torch.zeros(1, 1, args.size, args.size, device=device)
    random_dtm = torch.randn(1, 1, args.size, args.size, device=device)

    with torch.inference_mode():
        zero_outputs = model(torch.cat((rgb, zero_dtm), dim=1))
        random_outputs = model(torch.cat((rgb, random_dtm), dim=1))

    differences = [
        float((left - right).abs().max().item())
        for left, right in zip(zero_outputs, random_outputs)
    ]
    dtm_slice_max = float(model.dtm_conv.weight[:, 3:4].abs().max().item())
    passed = max(differences) <= args.atol and dtm_slice_max == 0.0
    result = {
        "device": str(device),
        "model_cfg": args.model_cfg,
        "input_size": args.size,
        "dtm_weight_max_abs": dtm_slice_max,
        "output_max_abs_differences": differences,
        "atol": args.atol,
        "passed": passed,
        "interpretation": (
            "At initialisation, arbitrary DTM values contribute exactly zero; "
            "the four-channel network therefore starts from its RGB-only behaviour."
        ),
    }
    print(json.dumps(result, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not passed:
        raise AssertionError("RGB-equivalence check failed")


if __name__ == "__main__":
    main(build_parser().parse_args())

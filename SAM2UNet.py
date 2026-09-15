"""SAM2-UNet with a controlled fourth input channel (RGB + DTM)."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def expand_conv3_to4(old_conv: nn.Conv2d) -> nn.Conv2d:
    """Create a 4-channel copy of a 3-channel convolution.

    RGB weights and bias are copied exactly.  The DTM slice is zero-initialised,
    so the model is numerically identical to the RGB-only model at step zero.
    """

    if not isinstance(old_conv, nn.Conv2d):
        raise TypeError(f"Expected nn.Conv2d, got {type(old_conv).__name__}")
    if old_conv.in_channels != 3:
        raise ValueError(f"Expected 3 input channels, got {old_conv.in_channels}")
    if old_conv.groups != 1:
        raise ValueError("The SAM2 patch embedding must use groups=1 for RGB+DTM expansion")

    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
        device=old_conv.weight.device,
        dtype=old_conv.weight.dtype,
    )
    with torch.no_grad():
        new_conv.weight[:, :3].copy_(old_conv.weight)
        new_conv.weight[:, 3:4].zero_()
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    return new_conv


def replace_first_conv_3_to_4(encoder: nn.Module) -> nn.Conv2d:
    """Replace the *explicit* SAM2 Hiera patch projection, not an arbitrary conv.

    The previous recursive "first Conv2d with in_channels=3" search was fragile:
    a SAM2 implementation change could silently replace the wrong layer.
    """

    patch_embed = getattr(encoder, "patch_embed", None)
    old_conv = getattr(patch_embed, "proj", None)
    if not isinstance(old_conv, nn.Conv2d):
        raise RuntimeError("Expected SAM2 encoder.patch_embed.proj to be nn.Conv2d")
    new_conv = expand_conv3_to4(old_conv)
    patch_embed.proj = new_conv
    print("[Model] encoder.patch_embed.proj expanded 3 -> 4 channels")
    print("[Model] RGB weights copied; DTM weight slice zero-initialised")
    return new_conv


def compute_hiera_padding(
    projection: nn.Conv2d,
    window_size: int,
    height: int,
    width: int,
) -> Tuple[int, int]:
    """Compute minimum bottom/right padding for a divisible Hiera patch grid."""

    if not isinstance(projection, nn.Conv2d):
        raise TypeError(f"projection must be nn.Conv2d, got {type(projection).__name__}")
    if window_size <= 0 or height <= 0 or width <= 0:
        raise ValueError(
            f"window_size, height and width must be > 0; got "
            f"{window_size}, {height}, {width}"
        )

    def output_length(length: int, axis: int) -> int:
        kernel = projection.kernel_size[axis]
        stride = projection.stride[axis]
        padding = projection.padding[axis]
        dilation = projection.dilation[axis]
        return (
            (length + 2 * padding - dilation * (kernel - 1) - 1) // stride
            + 1
        )

    def required(length: int, axis: int) -> int:
        # At most stride*window-1 pixels are needed before the patch-grid
        # divisibility pattern repeats.
        limit = projection.stride[axis] * window_size
        for amount in range(limit):
            patches = output_length(length + amount, axis)
            if patches > 0 and patches % window_size == 0:
                return amount
        raise RuntimeError(
            f"Could not find Hiera-compatible padding for length={length}, "
            f"axis={axis}, window={window_size}"
        )

    return required(height, 0), required(width, 1)


def freeze_rgb_train_dtm(conv_module: nn.Conv2d):
    """Mask RGB gradients while leaving the DTM slice trainable."""

    conv_module.weight.requires_grad_(True)
    if conv_module.bias is not None:
        conv_module.bias.requires_grad_(False)

    def _zero_rgb_gradient(gradient: torch.Tensor) -> torch.Tensor:
        result = gradient.clone()
        result[:, :3].zero_()
        return result

    return conv_module.weight.register_hook(_zero_rgb_gradient)


@torch.no_grad()
def dtm_weight_stats(conv_module: nn.Conv2d, grad_norm: Optional[float] = None) -> Dict[str, float | None]:
    weight = conv_module.weight[:, 3:4]
    if grad_norm is None and conv_module.weight.grad is not None:
        grad_norm = float(conv_module.weight.grad[:, 3:4].norm().item())
    return {
        "mean": float(weight.mean().item()),
        "std": float(weight.std().item()),
        "l2": float(weight.norm().item()),
        "grad_norm": grad_norm,
    }


def report_dtm_weight_stats(conv_module: nn.Conv2d, grad_norm: Optional[float] = None) -> Dict[str, float | None]:
    stats = dtm_weight_stats(conv_module, grad_norm=grad_norm)
    print(
        "[DTM weight] "
        f"mean={stats['mean']:.8f} std={stats['std']:.8f} "
        f"l2={stats['l2']:.8f} grad_norm={stats['grad_norm']}"
    )
    return stats



def clean_legacy_state_dict(state_dict: Mapping[str, torch.Tensor]) -> OrderedDict:
    """Remove inactive/duplicate keys written by earlier project versions."""
    cleaned = OrderedDict(state_dict)
    for suffix in ("weight", "bias"):
        legacy_key = f"dtm_conv.{suffix}"
        canonical_key = f"encoder.patch_embed.proj.{suffix}"
        if legacy_key in cleaned and canonical_key not in cleaned:
            cleaned[canonical_key] = cleaned[legacy_key]
        cleaned.pop(legacy_key, None)
    for key in [key for key in cleaned if key.startswith("up4.")]:
        cleaned.pop(key)
    return cleaned

def _group_count(channels: int, maximum: int = 32) -> int:
    groups = min(maximum, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(mid_channels), mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, deeper: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Interpolate to the exact skip size.  This also handles odd dimensions;
        # scale_factor+padding could create negative padding for some inputs.
        deeper = F.interpolate(
            deeper, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.conv(torch.cat((skip, deeper), dim=1))


class Adapter(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        dim = block.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x + self.prompt_learn(x))


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size,
        stride: int = 1,
        padding=0,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        # Keep the historical attribute name ``bn`` for checkpoint compatibility;
        # the module itself is GroupNorm, not BatchNorm.
        self.bn = nn.GroupNorm(_group_count(out_planes), out_planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class RFBModified(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.branch0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7),
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            (self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)), dim=1
        )
        return self.relu(self.conv_cat(features) + self.conv_res(x))


# Backward-compatible class name used by old checkpoints/scripts.
RFB_modified = RFBModified


class SAM2UNet(nn.Module):
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_cfg: str = "sam2_hiera_l.yaml",
        decoder_channels: int = 64,
    ) -> None:
        super().__init__()

        # Lazy import lets dataset/loss unit tests run even when hydra-core is
        # not installed.  Full model construction still fails with a clear
        # dependency error if the environment is incomplete.
        try:
            from sam2.build_sam import build_sam2
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Building SAM2UNet requires hydra-core. Install requirements.txt "
                "in the active notebook kernel."
            ) from exc

        # Build on CPU first.  The old call used build_sam2's default
        # device='cuda', which failed on CPU and caused avoidable peak GPU memory
        # before unused SAM2 modules were deleted.
        sam2_model = build_sam2(
            model_cfg,
            checkpoint_path,
            device="cpu",
            apply_postprocessing=False,
        )

        removable = (
            "sam_mask_decoder",
            "sam_prompt_encoder",
            "memory_encoder",
            "memory_attention",
            "mask_downsample",
            "obj_ptr_tpos_proj",
            "obj_ptr_proj",
        )
        for name in removable:
            if hasattr(sam2_model, name):
                delattr(sam2_model, name)
        if hasattr(sam2_model.image_encoder, "neck"):
            del sam2_model.image_encoder.neck

        self.model_cfg = model_cfg
        self.decoder_channels = int(decoder_channels)
        self.encoder = sam2_model.image_encoder.trunk
        dtm_conv = replace_first_conv_3_to_4(self.encoder)

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self._rgb_gradient_hook = freeze_rgb_train_dtm(dtm_conv)
        self.register_buffer(
            "_frozen_rgb_reference",
            dtm_conv.weight[:, :3].detach().clone(),
            persistent=False,
        )

        self.encoder.blocks = nn.Sequential(*(Adapter(block) for block in self.encoder.blocks))

        # Hiera stores channels deepest->shallowest, whereas forward returns
        # shallowest->deepest.  Deriving these values removes the hard-coded
        # 144/288/576/1152 assumption and supports all supplied SAM2 configs.
        channel_list = list(getattr(self.encoder, "channel_list", []))
        feature_channels = list(reversed(channel_list))
        if len(feature_channels) != 4:
            raise RuntimeError(
                f"Expected four Hiera feature stages, got channel_list={channel_list}"
            )
        self.feature_channels = tuple(int(value) for value in feature_channels)

        self.rfb1 = RFBModified(self.feature_channels[0], self.decoder_channels)
        self.rfb2 = RFBModified(self.feature_channels[1], self.decoder_channels)
        self.rfb3 = RFBModified(self.feature_channels[2], self.decoder_channels)
        self.rfb4 = RFBModified(self.feature_channels[3], self.decoder_channels)

        self.up1 = Up(2 * self.decoder_channels, self.decoder_channels)
        self.up2 = Up(2 * self.decoder_channels, self.decoder_channels)
        self.up3 = Up(2 * self.decoder_channels, self.decoder_channels)

        self.side1 = nn.Conv2d(self.decoder_channels, 1, kernel_size=1)
        self.side2 = nn.Conv2d(self.decoder_channels, 1, kernel_size=1)
        self.head = nn.Conv2d(self.decoder_channels, 1, kernel_size=1)

    def _padding_for_hiera(self, height: int, width: int) -> Tuple[int, int]:
        """Return bottom/right padding required by Hiera positional windows.

        The bundled Hiera implementation tiles ``pos_embed_window`` with integer
        division, so the patch grid must be exactly divisible by the first
        window size.  A non-square input such as 64x80 otherwise fails before
        the first block even though the decoder can handle arbitrary shapes.
        """

        projection = self.dtm_conv
        window_spec = getattr(self.encoder, "window_spec", None)
        if not window_spec:
            return 0, 0
        return compute_hiera_padding(
            projection=projection,
            window_size=int(window_spec[0]),
            height=height,
            width=width,
        )

    @property
    def dtm_conv(self) -> nn.Conv2d:
        # Property avoids registering the same module twice.  The old
        # `self.dtm_conv = dtm_conv` duplicated its keys in every state_dict.
        return self.encoder.patch_embed.proj

    @torch.no_grad()
    def refresh_frozen_rgb_reference(self) -> None:
        self._frozen_rgb_reference.copy_(self.dtm_conv.weight[:, :3])

    @torch.no_grad()
    def restore_frozen_rgb_weights(self) -> None:
        """Project RGB weights back after each optimiser step.

        This is a second line of defence against AdamW decay or stale optimiser
        momentum when resuming an old checkpoint.  The gradient hook alone does
        not prevent those two update paths.
        """

        self.dtm_conv.weight[:, :3].copy_(self._frozen_rgb_reference)

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True, assign: bool = False):
        # Legacy checkpoints stored duplicate aliases because the same conv was
        # registered as encoder.patch_embed.proj and dtm_conv.  They also
        # contain the unused up4 block.
        cleaned = clean_legacy_state_dict(state_dict)

        try:
            result = super().load_state_dict(cleaned, strict=strict, assign=assign)
        except TypeError:  # compatibility with older PyTorch without `assign`
            result = super().load_state_dict(cleaned, strict=strict)
        self.refresh_frozen_rgb_reference()
        return result

    def trainable_parameter_report(self) -> Dict[str, int | float]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_fraction": trainable / max(total, 1),
        }

    def forward(self, x: torch.Tensor):
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(f"Expected input [B,4,H,W], got {tuple(x.shape)}")
        output_size = x.shape[-2:]

        pad_h, pad_w = self._padding_for_hiera(*output_size)
        if pad_h or pad_w:
            # Zero is ImageNet-mean RGB and the neutral/missing value for the
            # default z-scored DTM, so it is preferable to an arbitrary edge
            # value. Padding is only on bottom/right; coordinates remain fixed.
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        padded_size = x.shape[-2:]

        x1, x2, x3, x4 = self.encoder(x)
        x1 = self.rfb1(x1)
        x2 = self.rfb2(x2)
        x3 = self.rfb3(x3)
        x4 = self.rfb4(x4)

        decoded3 = self.up1(x4, x3)
        out1 = F.interpolate(
            self.side1(decoded3), size=padded_size, mode="bilinear", align_corners=False
        )
        decoded2 = self.up2(decoded3, x2)
        out2 = F.interpolate(
            self.side2(decoded2), size=padded_size, mode="bilinear", align_corners=False
        )
        decoded1 = self.up3(decoded2, x1)
        out = F.interpolate(
            self.head(decoded1), size=padded_size, mode="bilinear", align_corners=False
        )
        height, width = output_size
        return (
            out[..., :height, :width],
            out1[..., :height, :width],
            out2[..., :height, :width],
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAM2UNet().to(device).eval()
    with torch.inference_mode():
        sample = torch.randn(1, 4, 352, 352, device=device)
        outputs = model(sample)
    print([tuple(output.shape) for output in outputs])

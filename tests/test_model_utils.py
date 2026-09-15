import torch
import torch.nn as nn

from SAM2UNet import SAM2UNet, compute_hiera_padding, expand_conv3_to4, freeze_rgb_train_dtm


def test_zero_init_preserves_rgb_output_for_any_dtm():
    torch.manual_seed(0)
    old = nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=True)
    new = expand_conv3_to4(old)
    rgb = torch.randn(2, 3, 8, 8)
    dtm_a = torch.zeros(2, 1, 8, 8)
    dtm_b = torch.randn(2, 1, 8, 8)
    with torch.no_grad():
        expected = old(rgb)
        output_a = new(torch.cat((rgb, dtm_a), dim=1))
        output_b = new(torch.cat((rgb, dtm_b), dim=1))
    assert torch.equal(new.weight[:, :3], old.weight)
    assert torch.count_nonzero(new.weight[:, 3:4]) == 0
    assert torch.allclose(output_a, expected, atol=0, rtol=0)
    assert torch.allclose(output_b, expected, atol=0, rtol=0)

    patch_projection = nn.Conv2d(4, 96, kernel_size=7, stride=4, padding=3)
    assert compute_hiera_padding(patch_projection, 8, 64, 80) == (0, 13)
    assert compute_hiera_padding(patch_projection, 8, 512, 512) == (0, 0)


def test_gradient_hook_updates_only_dtm_slice():
    torch.manual_seed(0)
    conv = expand_conv3_to4(nn.Conv2d(3, 2, kernel_size=1, bias=False))
    freeze_rgb_train_dtm(conv)
    rgb_before = conv.weight[:, :3].detach().clone()
    dtm_before = conv.weight[:, 3:4].detach().clone()
    optimizer = torch.optim.SGD([conv.weight], lr=0.1)
    x = torch.randn(2, 4, 4, 4)
    loss = conv(x).square().mean()
    loss.backward()
    assert torch.count_nonzero(conv.weight.grad[:, :3]) == 0
    assert torch.count_nonzero(conv.weight.grad[:, 3:4]) > 0
    optimizer.step()
    assert torch.equal(conv.weight[:, :3], rgb_before)
    assert not torch.equal(conv.weight[:, 3:4], dtm_before)



def test_rgb_reference_restoration_defeats_adamw_weight_decay():
    conv = expand_conv3_to4(nn.Conv2d(3, 2, kernel_size=1, bias=False))
    freeze_rgb_train_dtm(conv)

    model = SAM2UNet.__new__(SAM2UNet)
    nn.Module.__init__(model)
    model.encoder = nn.Module()
    model.encoder.patch_embed = nn.Module()
    model.encoder.patch_embed.proj = conv
    model.register_buffer(
        "_frozen_rgb_reference", conv.weight[:, :3].detach().clone(), persistent=False
    )

    rgb_before = model._frozen_rgb_reference.clone()
    optimizer = torch.optim.AdamW([conv.weight], lr=0.1, weight_decay=0.2)
    loss = conv(torch.randn(2, 4, 4, 4)).square().mean()
    loss.backward()
    optimizer.step()
    assert not torch.equal(conv.weight[:, :3], rgb_before)

    model.restore_frozen_rgb_weights()
    assert torch.equal(conv.weight[:, :3], rgb_before)

def test_legacy_state_dict_cleanup_removes_duplicate_and_dead_keys():
    from SAM2UNet import clean_legacy_state_dict

    state = {
        "encoder.patch_embed.proj.weight": torch.ones(1),
        "dtm_conv.weight": torch.zeros(1),
        "dtm_conv.bias": torch.zeros(1),
        "up4.conv.double_conv.0.weight": torch.ones(1),
        "head.weight": torch.ones(1),
    }
    cleaned = clean_legacy_state_dict(state)
    assert "encoder.patch_embed.proj.weight" in cleaned
    assert "dtm_conv.weight" not in cleaned
    assert "dtm_conv.bias" not in cleaned
    assert not any(key.startswith("up4.") for key in cleaned)
    assert "head.weight" in cleaned

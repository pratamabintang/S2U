"""
Cek apakah model benar-benar collapse (output sigmoid ~1 di hampir semua
piksel, termasuk tile background), atau cuma soal threshold yang kurang pas.

CARA PAKAI (Colab / Jupyter): edit CONFIG, lalu jalankan cell ini langsung.
"""

import torch
from torch.utils.data import DataLoader

from dataset import FullDataset
from SAM2UNet import SAM2UNet


# =========================================================
# CONFIG - edit sesuai path kamu
# =========================================================
HIERA_PATH = "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/2.MODIFICATION_SAM2-UNet/SAM2-UNet/sam2_hiera_large.pt"
CHECKPOINT_PATH = "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/6.TRAINING RESULT/2.MODIFY_CODE/IMAGE_DTM/0%_NEW/best_by_val_iou.pth"

VAL_IMAGE_PATH = "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/4.VALIDATION/IMAGE"
VAL_MASK_PATH = "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/4.VALIDATION/LABEL"
VAL_DTM_PATH = "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/4.VALIDATION/DTM_NORM"

TRAINSIZE = 512
DTM_SCALE = 1.0
DTM_AVAILABILITY = 1.0
N_TILES_TO_CHECK = 10   # cukup sample beberapa tile, tidak perlu semua 184
# =========================================================


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SAM2UNet(HIERA_PATH).to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
model.load_state_dict(state_dict, strict=True)
model.eval()

val_dataset = FullDataset(
    image_root=VAL_IMAGE_PATH,
    mask_root=VAL_MASK_PATH,
    dtm_root=VAL_DTM_PATH,
    trainsize=TRAINSIZE,
    mode="test",
    dtm_scale=DTM_SCALE,
    dtm_availability=DTM_AVAILABILITY
)

loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

print("=" * 80)
print(f"{'idx':>4} | {'is_background_tile':>18} | {'prob_min':>9} | "
      f"{'prob_max':>9} | {'prob_mean':>10} | {'%pixel>0.5':>10}")
print("-" * 80)

with torch.no_grad():
    for i, batch in enumerate(loader):
        if i >= N_TILES_TO_CHECK:
            break

        x = batch["image"].to(device)
        target = batch["label"].to(device)

        pred0, _, _ = model(x)
        prob = torch.sigmoid(pred0)

        is_background = (target.sum().item() == 0)
        pct_above_half = (prob >= 0.5).float().mean().item() * 100

        print(f"{i:>4} | {str(is_background):>18} | "
              f"{prob.min().item():>9.4f} | {prob.max().item():>9.4f} | "
              f"{prob.mean().item():>10.4f} | {pct_above_half:>9.2f}%")

print("=" * 80)
print("INTERPRETASI:")
print("- Kalau prob_mean mendekati 1.0 bahkan di tile is_background=True,")
print("  ini KONFIRMASI model collapse total (selalu prediksi positif),")
print("  bukan sekadar soal threshold kurang pas.")
print("- Kalau prob_mean masih rendah (~0.1-0.4) tapi %pixel>0.5 tinggi,")
print("  berarti banyak piksel borderline di sekitar 0.5 - beda masalah")
print("  (soal kalibrasi/threshold, bukan collapse total).")
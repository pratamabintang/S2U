"""
Diagnostik: apakah val_iou rendah disebabkan oleh false positive
yang tersebar di banyak tile background (bukan model gagal total),
atau karena tile crack hilang dari validation set akibat matching
DTM yang tidak lengkap?

CARA PAKAI (Colab / Jupyter):
Edit variabel CONFIG di bawah, lalu jalankan cell ini langsung.
"""

import os
import torch
from torch.utils.data import DataLoader

from dataset import FullDataset
from SAM2UNet import SAM2UNet


# =========================================================
# CONFIG - edit bagian ini sesuai path kamu
# =========================================================
HIERA_PATH = "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/2.MODIFICATION_SAM2-UNet/SAM2-UNet/sam2_hiera_large.pt"
CHECKPOINT_PATH = "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/6.TRAINING RESULT/2.MODIFY_CODE/IMAGE_DTM/0%_NEW/best_by_val_iou.pth"

VAL_IMAGE_PATH = "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/4.VALIDATION/IMAGE"
VAL_MASK_PATH = "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/4.VALIDATION/LABEL"
VAL_DTM_PATH = "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/4.VALIDATION/DTM_NORM"

TRAINSIZE = 512
THRESHOLD = 0.5
DTM_SCALE = 1.0
DTM_AVAILABILITY = 1.0   # samakan dengan availability run yang dites
BATCH_SIZE = 4
NUM_WORKERS = 2
# =========================================================


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")

model = SAM2UNet(HIERA_PATH).to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
model.load_state_dict(state_dict, strict=True)
model.eval()

print(f"[Checkpoint] Loaded: {CHECKPOINT_PATH}")

val_dataset = FullDataset(
    image_root=VAL_IMAGE_PATH,
    mask_root=VAL_MASK_PATH,
    dtm_root=VAL_DTM_PATH,
    trainsize=TRAINSIZE,
    mode="test",
    dtm_scale=DTM_SCALE,
    dtm_availability=DTM_AVAILABILITY
)

# -----------------------------------------------------
# CEK PALING PENTING: bandingkan jumlah tile yang ke-load
# vs jumlah file RGB asli di folder. Kalau beda jauh, berarti
# banyak tile ke-drop karena DTM tidak match namanya.
# -----------------------------------------------------
raw_image_count = len([
    f for f in os.listdir(VAL_IMAGE_PATH)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])
loaded_count = len(val_dataset)

print()
print(f"[CEK MATCHING] Jumlah file RGB asli di folder : {raw_image_count}")
print(f"[CEK MATCHING] Jumlah tile ter-load ke dataset : {loaded_count}")

if loaded_count < raw_image_count:
    dropped = raw_image_count - loaded_count
    print(f"[PERINGATAN] {dropped} tile ({dropped/raw_image_count:.1%}) HILANG dari "
          f"validation set - kemungkinan karena file DTM tidak ketemu/tidak match nama.")
    print("Ini bisa jadi PENYEBAB UTAMA val_iou rendah, kalau tile yang hilang")
    print("justru banyak yang mengandung crack. Cek nama file DTM_NORM kamu.")
else:
    print("[OK] Semua tile RGB berhasil match dengan DTM, tidak ada yang hilang.")

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

total_tp = 0.0
total_fp = 0.0
total_fn = 0.0

n_background_tiles = 0
n_background_tiles_with_fp = 0

per_tile_ious = []

with torch.no_grad():
    for batch in val_loader:
        x = batch["image"].to(device)
        target = batch["label"].to(device)

        pred0, _, _ = model(x)
        prob = torch.sigmoid(pred0)
        pred_binary = (prob >= THRESHOLD)
        target_binary = (target > 0.5)

        for i in range(x.shape[0]):
            p = pred_binary[i, 0]
            t = target_binary[i, 0]

            tp = torch.logical_and(p, t).sum().item()
            fp = torch.logical_and(p, ~t).sum().item()
            fn = torch.logical_and(~p, t).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn

            is_background_tile = (t.sum().item() == 0)

            if is_background_tile:
                n_background_tiles += 1
                if fp > 0:
                    n_background_tiles_with_fp += 1
                per_tile_iou = 1.0 if fp == 0 else 0.0
            else:
                union = tp + fp + fn
                per_tile_iou = tp / union if union > 0 else 1.0

            per_tile_ious.append(per_tile_iou)

pooled_iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 1.0
mean_per_tile_iou = sum(per_tile_ious) / len(per_tile_ious) if per_tile_ious else 0.0

n_crack_tiles = loaded_count - n_background_tiles

print()
print("=" * 70)
print("DIAGNOSTIK VAL IoU")
print("=" * 70)
print(f"Total tile validasi (ter-load) : {loaded_count}")
print(f"Tile dengan crack (ground truth): {n_crack_tiles} "
      f"({n_crack_tiles / max(loaded_count,1):.1%})")
print(f"Tile background (no crack)     : {n_background_tiles} "
      f"({n_background_tiles / max(loaded_count,1):.1%})")
print(f"Tile background yang kena FP   : {n_background_tiles_with_fp} "
      f"({n_background_tiles_with_fp / max(n_background_tiles,1):.1%} dari tile background)")
print()
print(f"Total TP pixels: {total_tp:.0f}")
print(f"Total FP pixels: {total_fp:.0f}")
print(f"Total FN pixels: {total_fn:.0f}")
print()
print(f"Pooled IoU (global, cara train.py)     : {pooled_iou:.6f}")
print(f"Mean per-tile IoU (rata-rata per tile) : {mean_per_tile_iou:.6f}")
print("=" * 70)

if n_crack_tiles == 0:
    print("[TEMUAN KRITIS] TIDAK ADA satupun tile validasi yang punya crack "
          "di ground truth! val_iou akan SELALU nol/nyaris-nol, bukan karena "
          "model gagal, tapi karena tidak ada foreground sama sekali untuk diukur.")
    print("Cek folder mask validasi dan proses matching file DTM.")
elif n_crack_tiles < 5:
    print(f"[CATATAN] Cuma ada {n_crack_tiles} tile bercrack di validation - "
          "sample sangat kecil, metrik akan sangat high-variance/noisy per epoch.")
elif total_fp > total_tp * 5:
    print("[INDIKASI KUAT] False positive jauh melebihi true positive.")
    print("Union didominasi FP tersebar di tile background, bukan model gagal")
    print("total di tile yang benar-benar ada crack.")
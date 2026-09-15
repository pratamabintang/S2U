"""
Test definitif: apakah collapse disebabkan oleh BatchNorm running
statistics yang tidak terkalibrasi?

Cara kerja: jalankan tile YANG SAMA dua kali -
  (a) model.eval()  -> BatchNorm pakai running_mean/running_var
  (b) model.train() -> BatchNorm pakai statistik batch saat itu (tapi
                        TANPA update gradient - cuma untuk observasi)

Kalau (a) collapse tapi (b) jauh lebih masuk akal (prob rendah di
background), itu KONFIRMASI penyebabnya BatchNorm running stats yang
buruk, bukan bobot network yang salah secara fundamental.

CARA PAKAI: edit CONFIG, jalankan sebagai cell Python langsung.
"""

import torch
from torch.utils.data import DataLoader

from dataset import FullDataset
from SAM2UNet import SAM2UNet


# =========================================================
# CONFIG - edit sesuai path kamu
# =========================================================
HIERA_PATH = "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/2.MODIFICATION_SAM2-UNet/SAM2-UNet/sam2_hiera_large.pt"
CHECKPOINT_PATH = "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/6.TRAINING RESULT/2.MODIFY_CODE/IMAGE_DTM/0%_TEST/best_by_val_iou.pth"

VAL_IMAGE_PATH = "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/4.VALIDATION/IMAGE"
VAL_MASK_PATH = "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/4.VALIDATION/LABEL"
VAL_DTM_PATH = "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/4.VALIDATION/DTM_NORM"

TRAINSIZE = 512
DTM_SCALE = 1.0
DTM_AVAILABILITY = 1.0
BATCH_SIZE_FOR_TRAIN_MODE_TEST = 4   # samakan dengan batch_size training asli
N_BATCHES_TO_CHECK = 5
# =========================================================


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SAM2UNet(HIERA_PATH).to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
model.load_state_dict(state_dict, strict=True)

val_dataset = FullDataset(
    image_root=VAL_IMAGE_PATH,
    mask_root=VAL_MASK_PATH,
    dtm_root=VAL_DTM_PATH,
    trainsize=TRAINSIZE,
    mode="test",
    dtm_scale=DTM_SCALE,
    dtm_availability=DTM_AVAILABILITY
)

loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_FOR_TRAIN_MODE_TEST, shuffle=False)

# -----------------------------------------------------
# Cek dulu: apakah ada BatchNorm di decoder, dan seperti
# apa running_mean/running_var-nya sekarang
# -----------------------------------------------------
print("=" * 80)
print("BATCHNORM RUNNING STATS DI DECODER")
print("=" * 80)
bn_found = False
for name, module in model.named_modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        bn_found = True
        rm = module.running_mean
        rv = module.running_var
        print(f"{name:40s} | running_mean: min={rm.min().item():.4f} "
              f"max={rm.max().item():.4f} mean={rm.mean().item():.4f} | "
              f"running_var: min={rv.min().item():.4f} max={rv.max().item():.4f}")

if not bn_found:
    print("Tidak ada nn.BatchNorm2d ditemukan di model (tidak sesuai dugaan).")

print()
print("=" * 80)
print("BANDINGAN: model.eval() (running stats) vs model.train() (batch stats)")
print("=" * 80)
print(f"{'batch':>5} | {'mode':>8} | {'prob_min':>9} | {'prob_max':>9} | "
      f"{'prob_mean':>10} | {'target_has_crack':>17}")
print("-" * 80)

with torch.no_grad():
    for i, batch in enumerate(loader):
        if i >= N_BATCHES_TO_CHECK:
            break

        x = batch["image"].to(device)
        target = batch["label"].to(device)
        has_crack = (target.sum().item() > 0)

        # (a) eval mode - running stats (yang dipakai saat testing biasa)
        model.eval()
        pred_eval, _, _ = model(x)
        prob_eval = torch.sigmoid(pred_eval)

        # (b) train mode - batch stats (TANPA backward/update apapun)
        model.train()
        pred_train, _, _ = model(x)
        prob_train = torch.sigmoid(pred_train)
        model.eval()  # kembalikan ke eval supaya tidak mengganggu loop berikutnya

        print(f"{i:>5} | {'eval':>8} | {prob_eval.min().item():>9.4f} | "
              f"{prob_eval.max().item():>9.4f} | {prob_eval.mean().item():>10.4f} | "
              f"{str(has_crack):>17}")
        print(f"{i:>5} | {'train':>8} | {prob_train.min().item():>9.4f} | "
              f"{prob_train.max().item():>9.4f} | {prob_train.mean().item():>10.4f} | "
              f"{str(has_crack):>17}")
        print("-" * 80)

print()
print("INTERPRETASI:")
print("- Kalau 'eval' selalu prob_mean~1.0 TAPI 'train' (batch stats) jauh")
print("  lebih masuk akal (prob rendah di tile background) -> TERKONFIRMASI:")
print("  BatchNorm running statistics adalah biang keladinya.")
print("- Kalau keduanya sama-sama collapse -> bukan soal BN, kemungkinan")
print("  besar network memang belajar bobot yang salah/collapse beneran")
print("  (perlu training ulang dengan LR lebih kecil / cek loss function).")
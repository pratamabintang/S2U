"""
Kumpulkan hasil training dari beberapa folder eksperimen dtm_availability
(0/25/50/75/100%), lalu bikin:
  1. Tabel ringkasan (best val_iou, epoch terbaik, val_loss terkait)
  2. Plot val_iou vs epoch, semua level dalam satu grafik
  3. Plot ringkasan: best val_iou vs dtm_availability (kurva utama riset kamu)
  4. Plot dtm_weight_mean vs epoch (sanity check - buktikan weight belajar)

Cara pakai:
    python summarize_ablation.py \
        --runs 0=".../IMAGE_DTM/0%" 25=".../IMAGE_DTM/25%" \
               50=".../IMAGE_DTM/50%" 75=".../IMAGE_DTM/75%" \
               100=".../IMAGE_DTM/100%" \
        --output_dir ".../6.TRAINING RESULT/summary"

Format --runs: "LABEL=PATH", LABEL berupa angka persen (0, 25, 50, 75, 100).
Bisa juga cuma sebagian run kalau belum semua selesai training, script
akan skip run yang log.csv-nya belum ada / kosong dan kasih peringatan.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_runs(run_args):
    """
    run_args: list string "LABEL=PATH"
    Return: dict {label(int): path(str)}, terurut naik berdasarkan label.
    """
    runs = {}
    for item in run_args:
        if "=" not in item:
            raise ValueError(
                f"Format --runs salah: '{item}'. Harus 'LABEL=PATH', "
                f"contoh: 25=\"/path/ke/folder25persen\""
            )
        label_str, path = item.split("=", 1)
        label = int(label_str.strip())
        runs[label] = path.strip()

    return dict(sorted(runs.items()))


def read_log_csv(csv_path):
    """
    Baca log.csv hasil train.py. Return list of dict per baris,
    dengan field numerik sudah di-cast (val_iou/val_loss bisa None
    kalau kosong / tidak ada validation).
    """
    rows = []

    if not os.path.exists(csv_path):
        return rows

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            def to_float_or_none(value):
                if value is None or value == "":
                    return None
                try:
                    return float(value)
                except ValueError:
                    return None

            rows.append({
                "epoch": int(row["epoch"]),
                "train_loss": to_float_or_none(row.get("train_loss")),
                "train_iou": to_float_or_none(row.get("train_iou")),
                "val_loss": to_float_or_none(row.get("val_loss")),
                "val_iou": to_float_or_none(row.get("val_iou")),
                "dtm_weight_mean": to_float_or_none(row.get("dtm_weight_mean")),
                "dtm_weight_std": to_float_or_none(row.get("dtm_weight_std")),
            })

    return rows


def summarize(runs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    all_run_rows = {}   # label -> list of row dict
    summary_records = []  # list of dict for the final table

    for label, path in runs.items():
        csv_path = os.path.join(path, "log.csv")
        rows = read_log_csv(csv_path)

        if not rows:
            print(f"[SKIP] dtm_availability={label}%: log.csv tidak ditemukan/kosong di {csv_path}")
            continue

        all_run_rows[label] = rows

        # Cari baris dengan val_iou terbaik (tertinggi)
        rows_with_val = [r for r in rows if r["val_iou"] is not None]

        if rows_with_val:
            best_row = max(rows_with_val, key=lambda r: r["val_iou"])
            best_val_iou = best_row["val_iou"]
            best_val_loss = best_row["val_loss"]
            best_epoch = best_row["epoch"]
        else:
            best_val_iou = None
            best_val_loss = None
            best_epoch = None
            print(f"[WARNING] dtm_availability={label}%: tidak ada val_iou "
                  f"tercatat (mungkin tidak ada validation set / masih training).")

        last_row = rows[-1]

        summary_records.append({
            "dtm_availability_pct": label,
            "epochs_completed": last_row["epoch"],
            "best_val_iou": best_val_iou,
            "best_val_iou_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_train_iou": last_row["train_iou"],
            "final_dtm_weight_mean": last_row["dtm_weight_mean"],
            "final_dtm_weight_std": last_row["dtm_weight_std"],
        })

        print(f"[OK] dtm_availability={label}%: "
              f"best_val_iou={best_val_iou} @ epoch {best_epoch}, "
              f"epochs_completed={last_row['epoch']}")

    if not summary_records:
        raise RuntimeError("Tidak ada run yang berhasil dibaca. Cek path --runs.")

    # -----------------------------------------------------
    # 1. Tabel ringkasan -> CSV
    # -----------------------------------------------------
    summary_csv_path = os.path.join(output_dir, "ablation_summary.csv")

    with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(summary_records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in summary_records:
            writer.writerow(record)

    print(f"\n[Saved] Tabel ringkasan: {summary_csv_path}")

    # Print tabel juga ke console biar langsung kelihatan
    print("\n" + "=" * 90)
    print(f"{'DTM avail %':>12} | {'epochs':>7} | {'best val_iou':>13} | "
          f"{'@epoch':>7} | {'best val_loss':>14} | {'final train_iou':>16}")
    print("-" * 90)
    for record in summary_records:
        print(
            f"{record['dtm_availability_pct']:>12} | "
            f"{record['epochs_completed']:>7} | "
            f"{('%.6f' % record['best_val_iou']) if record['best_val_iou'] is not None else 'N/A':>13} | "
            f"{record['best_val_iou_epoch'] if record['best_val_iou_epoch'] is not None else 'N/A':>7} | "
            f"{('%.6f' % record['best_val_loss']) if record['best_val_loss'] is not None else 'N/A':>14} | "
            f"{('%.6f' % record['final_train_iou']) if record['final_train_iou'] is not None else 'N/A':>16}"
        )
    print("=" * 90)

    # -----------------------------------------------------
    # 2. Plot val_iou vs epoch, semua level DTM availability
    # -----------------------------------------------------
    plt.figure(figsize=(8, 6))

    for label in sorted(all_run_rows.keys()):
        rows = all_run_rows[label]
        epochs = [r["epoch"] for r in rows if r["val_iou"] is not None]
        val_ious = [r["val_iou"] for r in rows if r["val_iou"] is not None]

        if epochs:
            plt.plot(epochs, val_ious, marker="o", markersize=3, label=f"DTM {label}%")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Foreground IoU")
    plt.title("Validation IoU vs Epoch, per DTM Availability Level")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    val_iou_plot_path = os.path.join(output_dir, "val_iou_vs_epoch_all_levels.png")
    plt.savefig(val_iou_plot_path, dpi=200)
    plt.close()

    print(f"[Saved] Plot val_iou vs epoch: {val_iou_plot_path}")

    # -----------------------------------------------------
    # 3. Plot ringkasan utama: best val_iou vs dtm_availability
    #    (INI kurva utama untuk menjawab pertanyaan riset kamu)
    # -----------------------------------------------------
    records_with_iou = [r for r in summary_records if r["best_val_iou"] is not None]
    records_with_iou.sort(key=lambda r: r["dtm_availability_pct"])

    if records_with_iou:
        x = [r["dtm_availability_pct"] for r in records_with_iou]
        y = [r["best_val_iou"] for r in records_with_iou]

        plt.figure(figsize=(7, 5))
        plt.plot(x, y, marker="o", linewidth=2, color="tab:blue")

        for xi, yi in zip(x, y):
            plt.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9)

        plt.xlabel("DTM Availability (%)")
        plt.ylabel("Best Validation Foreground IoU")
        plt.title("Kontribusi DTM terhadap Akurasi Segmentasi Crack\n"
                   "(Best Val IoU vs Persentase Ketersediaan DTM)")
        plt.xticks(x)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        main_plot_path = os.path.join(output_dir, "MAIN_best_val_iou_vs_dtm_availability.png")
        plt.savefig(main_plot_path, dpi=200)
        plt.close()

        print(f"[Saved] Plot utama (best_val_iou vs dtm_availability): {main_plot_path}")
    else:
        print("[WARNING] Tidak ada data val_iou yang cukup untuk plot utama.")

    # -----------------------------------------------------
    # 4. Plot dtm_weight_mean vs epoch (sanity check)
    #    Menunjukkan weight DTM memang bergerak/belajar, kecuali
    #    di run 0% (yang wajar diam karena gradien selalu nol).
    # -----------------------------------------------------
    plt.figure(figsize=(8, 6))

    for label in sorted(all_run_rows.keys()):
        rows = all_run_rows[label]
        epochs = [r["epoch"] for r in rows if r["dtm_weight_mean"] is not None]
        weight_means = [r["dtm_weight_mean"] for r in rows if r["dtm_weight_mean"] is not None]

        if epochs:
            plt.plot(epochs, weight_means, marker="o", markersize=3, label=f"DTM {label}%")

    plt.xlabel("Epoch")
    plt.ylabel("DTM conv weight - mean")
    plt.title("Sanity Check: DTM Weight Mean vs Epoch\n"
              "(harus bergerak untuk run >0%; wajar diam untuk run 0%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    weight_plot_path = os.path.join(output_dir, "dtm_weight_mean_vs_epoch_sanity_check.png")
    plt.savefig(weight_plot_path, dpi=200)
    plt.close()

    print(f"[Saved] Plot sanity check DTM weight: {weight_plot_path}")

    print(f"\n[Done] Semua hasil disimpan di: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kumpulkan & bandingkan hasil training ablation dtm_availability"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help='Daftar run, format "LABEL=PATH", contoh: '
             '0="/path/0%%" 25="/path/25%%" 50="/path/50%%" 75="/path/75%%" 100="/path/100%%"'
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Folder untuk menyimpan tabel & plot ringkasan"
    )

    args = parser.parse_args()

    runs = parse_runs(args.runs)
    summarize(runs, args.output_dir)
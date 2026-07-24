"""
YOLO11m training script - CSD.5 dataset
Classes: bulldozer, concrete mixer, dump truck, excavator, worker (5 classes)
200 epochs, batch=4 (safe default for RTX 5070 Ti 16GB, based on prior OOM history)

Includes:
- Hardcoded batch size (avoids Ultralytics auto-batch doubling during validation,
  which caused GPU OOM crashes in prior runs)
- Memory-clearing callback at validation start
- Standard train -> auto-evaluate on test split at the end
"""

import os
import torch
import gc
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================
DATA_YAML = r"D:\model train0.2\CSD.5 DS\data.yaml"
MODEL_BASE = "yolo11m.pt"          # pretrained checkpoint to fine-tune from
PROJECT_DIR = r"D:\model train0.2\runs\detect"
RUN_NAME = "yolov11m_csd5_v1"

EPOCHS = 200
BATCH_SIZE = 4          # safe default; try 6 only if you confirm no OOM (see note below)
IMG_SIZE = 960           # matches your prior successful training resolution
DEVICE = 0               # GPU 0 (RTX 5070 Ti)

# ============================================================
# Memory-clearing callback - prevents GPU OOM during validation
# (Ultralytics doubles batch size internally during val unless controlled)
# ============================================================
def on_val_start(validator):
    torch.cuda.empty_cache()
    gc.collect()


def main():
    os.makedirs(PROJECT_DIR, exist_ok=True)

    print(f"Loading base model: {MODEL_BASE}")
    model = YOLO(MODEL_BASE)

    # register memory-clearing callback before validation each epoch
    model.add_callback("on_val_start", on_val_start)

    print(f"Starting training for {EPOCHS} epochs, batch={BATCH_SIZE}, imgsz={IMG_SIZE}")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,

        # --- stability settings based on your prior OOM history ---
        workers=4,               # dataloader workers; lower if RAM is tight
        cache=False,             # avoid caching entire dataset in RAM (linked to earlier RAM issue)
        amp=True,                # mixed precision - reduces VRAM usage significantly
        val=True,                # validate every epoch

        # --- training hyperparameters (reasonable defaults for construction detection) ---
        patience=50,             # early stopping patience (won't trigger before real plateau)
        optimizer="auto",
        lr0=0.01,
        cos_lr=True,             # cosine LR schedule - smoother convergence over 200 epochs
        save=True,
        save_period=10,          # checkpoint every 10 epochs (safety net against crashes)
        plots=True,
        verbose=True,
        seed=42,
    )

    print("\n[INFO] Training complete.")
    best_weights = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
    print(f"[INFO] Best checkpoint: {best_weights}")

    # ============================================================
    # Final evaluation on test split
    # ============================================================
    print("\n[INFO] Running final evaluation on test split...")
    torch.cuda.empty_cache()
    gc.collect()

    eval_model = YOLO(best_weights)
    metrics = eval_model.val(
        data=DATA_YAML,
        split="test",
        batch=BATCH_SIZE,        # must be a positive integer, NOT -1 (fixes prior crash)
        imgsz=IMG_SIZE,
        project=PROJECT_DIR,
        name=f"{RUN_NAME}_test_eval",
        exist_ok=True,
        plots=True,
    )

    print("\n[INFO] Final test metrics:")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
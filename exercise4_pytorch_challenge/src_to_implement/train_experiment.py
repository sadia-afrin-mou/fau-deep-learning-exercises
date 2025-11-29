import os
import json
import csv
import argparse
from datetime import datetime
import shutil

import torch as t
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from data import ChallengeDataset
from trainer import Trainer
import model as model_baseline


def parse_args():
    p = argparse.ArgumentParser(description="Configurable training runner")
    # run/meta
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    # data
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    # model
    p.add_argument("--model-variant", type=str, default="baseline",
                   choices=["baseline", "pretrained18", "pretrained34", "pretrained50"])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--freeze-epochs", type=int, default=3, help="Freeze backbone for first N epochs (pretrained only)")
    # optim
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--patience", type=int, default=15)
    # scheduler
    p.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine", "plateau"])
    p.add_argument("--min-lr", type=float, default=1e-6)
    return p.parse_args()


# Unified CSV header
EXP_CSV_FIELDS = [
    "run_name","timestamp","model","optimizer","lr","batch_size","weight_decay",
    "epochs_requested","epochs_trained","patience","val_size","num_workers","seed",
    "best_epoch","best_val_loss","final_train_loss","final_val_loss","final_eval_loss",
    "f1_crack","f1_inactive","f1_mean","dropout","scheduler","freeze_epochs","loss_plot"
]
def set_seed(seed: int):
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)


def build_model(args):
    if args.model_variant == "baseline":
        m = model_baseline.ResNet()
        return m
    else:
        import model_pretrained as model_pt
        backbone = {
            "pretrained18": "resnet18",
            "pretrained34": "resnet34",
            "pretrained50": "resnet50"
        }[args.model_variant]
        m = model_pt.ResNet(backbone=backbone, pretrained=True, dropout=args.dropout)
        return m


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []
    with t.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += float(loss.item())
            n_batches += 1
            all_preds.append(out.cpu())
            all_labels.append(y.cpu())
    avg_loss = total_loss / max(1, n_batches)
    all_preds = t.cat(all_preds, dim=0)
    all_labels = t.cat(all_labels, dim=0)
    bin_preds = (all_preds > 0.5).float()

    def f1_score_np(y_true, y_pred):
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tp, fp, fn = int(tp), int(fp), int(fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    y0 = all_labels[:, 0].numpy().astype(np.int32)
    p0 = bin_preds[:, 0].numpy().astype(np.int32)
    y1 = all_labels[:, 1].numpy().astype(np.int32)
    p1 = bin_preds[:, 1].numpy().astype(np.int32)

    f1_crack = f1_score_np(y0, p0)
    f1_inactive = f1_score_np(y1, p1)
    f1_mean = (f1_crack + f1_inactive) / 2

    return avg_loss, f1_crack, f1_inactive, f1_mean


def freeze_backbone_for_epochs(model, epoch, freeze_epochs):
    if freeze_epochs <= 0:
        return
    if hasattr(model, "backbone"):
        requires_grad = epoch <= freeze_epochs
        for name, p in model.backbone.named_parameters():
            if "fc" in name:
                p.requires_grad = True
            else:
                p.requires_grad = not (not requires_grad)


def main():
    args = parse_args()
    set_seed(args.seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_csv = os.path.join(base_dir, "data.csv")

    # Run name and folders
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"run_{timestamp}_{args.model_variant}"
    runs_root = os.path.join(base_dir, "runs")
    run_dir = os.path.join(runs_root, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(runs_root, exist_ok=True)

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    df = pd.read_csv(data_csv, sep=";")

    # Check class distribution
    print("Overall class distribution:")
    class_map = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    for (c, i), v in class_map.items():
        count = sum((df['crack'] == c) & (df['inactive'] == i))
        print(f"  Crack={c}, Inactive={i}: {count} ({count/len(df)*100:.1f}%)")

    stratify_labels = [class_map[(x, y)] for x, y in df[['crack', 'inactive']].to_numpy()]
    train_df, val_df = train_test_split(df, test_size=0.1, shuffle=True, random_state=2, stratify=stratify_labels)

    train_ds = ChallengeDataset(train_df, "train")
    val_ds = ChallengeDataset(val_df, "val")

    num_workers = args.num_workers if t.cuda.is_available() else 0
    train_loader = t.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = t.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # Model, loss, optimizer
    model = build_model(args)
    criterion = t.nn.BCELoss()
    if args.optimizer == "adam":
        optimizer = t.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = t.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Scheduler
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr)
    elif args.scheduler == "plateau":
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=args.min_lr)

    # Trainer
    trainer = Trainer(
        model=model,
        crit=criterion,
        optim=optimizer,
        train_dl=train_loader,
        val_test_dl=val_loader,
        cuda=t.cuda.is_available(),
        early_stopping_patience=args.patience,
        scheduler=scheduler,
        freeze_epochs=args.freeze_epochs
    )

    # Print hyperparameters
    print("Starting training with hyperparameters:")
    print(f"Run Name: {run_name}")
    print(f"Model Variant: {args.model_variant}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Validation Split: {args.val_size}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Early Stopping Patience: {args.patience} epochs")
    print(f"Optimizer: {args.optimizer.upper()}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Dropout: {args.dropout}")
    print(f"Freeze Epochs: {args.freeze_epochs}")
    print(f"Using device: {'CUDA' if t.cuda.is_available() else 'CPU'}")

    # Use trainer's fit loop (no manual training loop)
    train_losses, val_losses = trainer.fit(epochs=args.epochs)

    # Plot and save losses
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(train_losses)), train_losses, label="train loss", color="blue")
    plt.plot(np.arange(len(val_losses)), val_losses, label="val loss", color="red")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_plot_path = os.path.join(run_dir, f"losses_{run_name}.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Best epoch and checkpoint handling
    best_val_loss = float(min(val_losses)) if len(val_losses) > 0 else float("inf")
    best_epoch = int(np.argmin(val_losses) + 1) if len(val_losses) > 0 else -1
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

    if best_epoch > 0:
        src_ckpt = os.path.join(base_dir, "checkpoints", f"checkpoint_{best_epoch:03d}.ckp")
        if os.path.exists(src_ckpt):
            # Restore best checkpoint and export ONNX
            trainer.restore_checkpoint(best_epoch)
            # Use competition format: checkpoint_NNN.onnx
            onnx_path = os.path.join(run_dir, f"checkpoint_{best_epoch:03d}.onnx")
            trainer.save_onnx(onnx_path)
            if t.cuda.is_available():
                trainer._model = trainer._model.cuda()
            print(f"Using checkpoint: {src_ckpt}")
            print(f"Exported ONNX to: {onnx_path}")
        else:
            print(f"Warning: expected best checkpoint not found: {src_ckpt}")

    # Evaluate (no threshold tuning)
    eval_loss, f1_crack, f1_inactive, f1_mean = evaluate(trainer._model, val_loader, criterion, device)

    # Save hyperparams for this run
    hparams = {
        "run_name": run_name,
        "timestamp": timestamp,
        "model": args.model_variant,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "epochs_requested": args.epochs,
        "epochs_trained": len(train_losses),
        "patience": args.patience,
        "val_size": args.val_size,
        "num_workers": num_workers,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_train_loss": float(train_losses[-1]) if len(train_losses) else None,
        "final_val_loss": float(val_losses[-1]) if len(val_losses) else None,
        "final_eval_loss": eval_loss,
        "f1_crack": f1_crack,
        "f1_inactive": f1_inactive,
        "f1_mean": f1_mean,
        "dropout": args.dropout,
        "scheduler": args.scheduler,
        "freeze_epochs": args.freeze_epochs,
        "loss_plot": os.path.relpath(loss_plot_path, base_dir)
    }
    with open(os.path.join(run_dir, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=2)

    # Append results row to a CSV with unified header, rewrite if header mismatches
    results_csv = os.path.join(runs_root, "experiments.csv")
    file_exists = os.path.exists(results_csv)
    desired_header = ",".join(EXP_CSV_FIELDS)
    mode = 'a'
    if file_exists:
        with open(results_csv, 'r') as rf:
            first_line = rf.readline().strip()
            if first_line != desired_header:
                mode = 'w'
    else:
        mode = 'w'
    with open(results_csv, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXP_CSV_FIELDS)
        w.writeheader()
        w.writerow(hparams)

    print("")
    print("Training completed.")
    print(f"Final training loss: {train_losses[-1]:.6f}" if len(train_losses) else "Final training loss: N/A")
    print(f"Final validation loss: {val_losses[-1]:.6f}" if len(val_losses) else "Final validation loss: N/A")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Total epochs: {len(train_losses)}")
    print(f"Scores - F1 Crack: {f1_crack:.6f}, F1 Inactive: {f1_inactive:.6f}, F1 Mean: {f1_mean:.6f}")
    print(f"Loss plot: {loss_plot_path}")
    print(f"Run folder: {run_dir}")
    print(f"Results CSV: {results_csv}")


if __name__ == "__main__":
    main()

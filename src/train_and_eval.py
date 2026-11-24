import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from models import CLASSES, NEUDataset, build_model, get_transforms, set_seed

SEEDS = [42, 123, 999]
MODELS = ["scratch", "resnet_frozen", "resnet_full"]

def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(y.numpy())
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro")
    return acc, f1, np.array(targets), np.array(preds)

def train_one_run(model_name, seed, df, device, epochs=20, batch_size=32):
    set_seed(seed)
    rgb = model_name != "scratch"
    lr = 1e-3 if model_name in ("scratch", "resnet_frozen") else 3e-4

    train_ds = NEUDataset(df[df["split"] == "train"], get_transforms(rgb=rgb, augment=True), rgb=rgb)
    val_ds = NEUDataset(df[df["split"] == "val"], get_transforms(rgb=rgb, augment=False), rgb=rgb)
    test_ds = NEUDataset(df[df["split"] == "test"], get_transforms(rgb=rgb, augment=False), rgb=rgb)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    best_val_f1 = 0.0
    ckpt_path = f"outputs/models/{model_name}_seed{seed}.pt"
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * x.size(0)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                v_loss += criterion(model(x), y).item() * x.size(0)

        _, val_f1, _, _ = evaluate(model, val_loader, device)
        history["train_loss"].append(t_loss / len(train_ds))
        history["val_loss"].append(v_loss / len(val_ds))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)

    # Touch the test set ONCE using the best validation checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"[{model_name} | seed {seed}] Best Val F1: {best_val_f1:.4f} | Test Acc: {test_acc:.4f} | Test Macro-F1: {test_f1:.4f}")

    # Save curves & confusion matrix for seed 42
    if seed == 42:
        plt.figure(figsize=(6, 4))
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.title(f"{model_name} Loss Curve (seed 42)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"outputs/figures/loss_{model_name}.png", dpi=150)
        plt.close()

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
        fig, ax = plt.subplots(figsize=(7, 6))
        disp.plot(ax=ax, xticks_rotation=45, cmap="Blues")
        plt.title(f"{model_name} Test Confusion Matrix (seed 42)")
        plt.tight_layout()
        plt.savefig(f"outputs/figures/cm_{model_name}.png", dpi=150)
        plt.close()

        print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

    return test_acc, test_f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    df = pd.read_csv("data/splits.csv")

    summary = []
    for model_name in MODELS:
        accs, f1s = [], []
        for seed in SEEDS:
            acc, f1 = train_one_run(model_name, seed, df, device)
            accs.append(acc)
            f1s.append(f1)
        summary.append({
            "model": model_name,
            "acc_mean": np.mean(accs), "acc_std": np.std(accs),
            "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
        })

    res_df = pd.DataFrame(summary)
    print("\n=== 3-Seed Test Summary ===")
    print(res_df.to_string(index=False))
    res_df.to_csv("outputs/test_summary.csv", index=False)

if __name__ == "__main__":
    main()
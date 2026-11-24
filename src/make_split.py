from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw/NEU-DET")
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

def main():
    records = []
    for split_dir in ["train", "validation"]:
        img_root = RAW_DIR / split_dir / "images"
        for cls in CLASSES:
            for img_path in sorted((img_root / cls).glob("*.jpg")):
                records.append({"filepath": str(img_path), "label": cls})

    df = pd.DataFrame(records)
    assert len(df) == 1800, f"Expected 1800 images, found {len(df)}"

    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df.to_csv("data/splits.csv", index=False)

    print("Class balance check (should be 210 train / 45 val / 45 test):")
    print(pd.crosstab(full_df["label"], full_df["split"]))

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for ax, cls in zip(axes.flat, CLASSES):
        sample_path = full_df[(full_df["label"] == cls) & (full_df["split"] == "train")].iloc[0]["filepath"]
        ax.imshow(Image.open(sample_path), cmap="gray")
        ax.set_title(cls)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/figures/samples.png", dpi=150)
    print("Saved data/splits.csv and outputs/figures/samples.png")

if __name__ == "__main__":
    main()
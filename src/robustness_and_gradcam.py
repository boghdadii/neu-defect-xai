import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from models import CLASSES, CLASS_TO_IDX, NEUDataset, build_model, get_transforms

def add_noise_and_brightness(tensor, noise_std=0.15, brightness_delta=0.25):
    noisy = tensor + torch.randn_like(tensor) * noise_std + brightness_delta
    return noisy

def eval_robustness(model, loader, device, noise_std=0.0, brightness_delta=0.0):
    model.eval()
    preds, targets = [], []
    torch.manual_seed(42)
    with torch.no_grad():
        for x, y in loader:
            if noise_std > 0 or brightness_delta != 0:
                x = add_noise_and_brightness(x, noise_std, brightness_delta)
            out = model(x.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(y.numpy())
    return accuracy_score(targets, preds), np.array(targets), np.array(preds)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv("data/splits.csv")
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # Load seed-42 models
    scratch = build_model("scratch").to(device)
    scratch.load_state_dict(torch.load("outputs/models/scratch_seed42.pt", map_location=device, weights_only=True))
    scratch.eval()

    resnet = build_model("resnet_full").to(device)
    resnet.load_state_dict(torch.load("outputs/models/resnet_full_seed42.pt", map_location=device, weights_only=True))
    resnet.eval()

    # 1. Robustness Test
    scratch_loader = DataLoader(NEUDataset(test_df, get_transforms(rgb=False), rgb=False), batch_size=32, num_workers=0)
    resnet_loader = DataLoader(NEUDataset(test_df, get_transforms(rgb=True), rgb=True), batch_size=32, num_workers=0)

    conditions = [
        ("Clean", 0.0, 0.0),
        ("Gaussian Noise (0.2)", 0.2, 0.0),
        ("Brightness (+0.4)", 0.0, 0.4),
        ("Noise + Brightness", 0.2, 0.4),
    ]

    print("=== Robustness Test (Accuracy) ===")
    for name, n_std, b_delta in conditions:
        s_acc, _, _ = eval_robustness(scratch, scratch_loader, device, n_std, b_delta)
        r_acc, _, _ = eval_robustness(resnet, resnet_loader, device, n_std, b_delta)
        print(f"{name:22s} | Scratch CNN: {s_acc:.4f} | ResNet18: {r_acc:.4f}")

    # 2. Side-by-Side Grad-CAM (pick 3 correct + up to 3 ResNet/Scratch mistakes)
    _, y_true, s_preds = eval_robustness(scratch, scratch_loader, device)
    _, _, r_preds = eval_robustness(resnet, resnet_loader, device)

    correct_idx = np.where((s_preds == y_true) & (r_preds == y_true))[0][:3]
    mistake_idx = np.where((s_preds != y_true) | (r_preds != y_true))[0][:3]
    chosen = list(correct_idx) + list(mistake_idx)

    cam_scratch = GradCAM(model=scratch, target_layers=[scratch.features[-1]])
    cam_resnet = GradCAM(model=resnet, target_layers=[resnet.layer4[-1]])

    tf_s = get_transforms(rgb=False)
    tf_r = get_transforms(rgb=True)

    fig, axes = plt.subplots(len(chosen), 3, figsize=(10, 3.2 * len(chosen)))
    if len(chosen) == 1:
        axes = [axes]

    for row, idx in enumerate(chosen):
        path = test_df.loc[idx, "filepath"]
        true_cls = CLASSES[y_true[idx]]

        img_gray = Image.open(path).convert("L")
        img_rgb = Image.open(path).convert("RGB").resize((224, 224))
        rgb_float = np.array(img_rgb, dtype=np.float32) / 255.0

        s_in = tf_s(img_gray).unsqueeze(0).to(device)
        r_in = tf_r(img_rgb).unsqueeze(0).to(device)

        s_cam = cam_scratch(input_tensor=s_in, targets=[ClassifierOutputTarget(int(s_preds[idx]))])[0]
        r_cam = cam_resnet(input_tensor=r_in, targets=[ClassifierOutputTarget(int(r_preds[idx]))])[0]

        s_vis = show_cam_on_image(np.array(img_gray.convert("RGB"), dtype=np.float32) / 255.0, s_cam, use_rgb=True)
        r_vis = show_cam_on_image(rgb_float, r_cam, use_rgb=True)

        axes[row][0].imshow(img_gray, cmap="gray")
        axes[row][0].set_title(f"True: {true_cls}")
        axes[row][1].imshow(s_vis)
        axes[row][1].set_title(f"Scratch pred: {CLASSES[s_preds[idx]]}")
        axes[row][2].imshow(r_vis)
        axes[row][2].set_title(f"ResNet18 pred: {CLASSES[r_preds[idx]]}")
        for col in range(3):
            axes[row][col].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/figures/gradcam_comparison.png", dpi=150)
    print("Saved outputs/figures/gradcam_comparison.png")

if __name__ == "__main__":
    main()
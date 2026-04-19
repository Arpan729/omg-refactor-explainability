import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')        # Prevent Tkinter crashes
import matplotlib.pyplot as plt
import argparse
import cv2

from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

from common import (
    RawFace3DCNNModel,
    load_config,
    denorm_target,
    SampleIndex,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Facial Feature Analysis with Captum")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--story", type=int, default=2)
    parser.add_argument("--start_frame", type=int, default=None, 
                       help="Starting frame index (None = use last 10 frames)")
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--config", type=str, default="Captum_config.yaml")
    return parser.parse_args()


def load_faces_from_subject_img(sample: SampleIndex):
    base_dir = Path(r"D:\Final_Project\S3 Project\Validation\Faces")
    face_dir = base_dir / f"Subject_{sample.subject}_Story_{sample.story}" / "Subject_img"
    
    face_paths = sorted(face_dir.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else 9999)
    print(f"Found {len(face_paths)} face frames")

    imgs = []
    for p in face_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = img.astype(np.float32) / 255.0
        m = float(np.mean(img))
        s = float(np.std(img))
        img = (img - m) / s if s > 1e-8 else (img - m)
        imgs.append(img)

    return np.asarray(imgs, dtype=np.float32)


def main():
    args = parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt_path = Path(cfg["paths"]["checkpoint_dir"]) / "raw_face_3dcnn.pt"
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return

    saved = torch.load(ckpt_path, map_location=device)
    target_min = float(saved.get("target_min", -1.0))
    target_max = float(saved.get("target_max", 1.0))

    model = RawFace3DCNNModel(cfg).to(device)
    model.load_state_dict(saved["model_state"])
    model.eval()

    sample = SampleIndex(subject=args.subject, story=args.story, split="val")

    x = load_faces_from_subject_img(sample)
    seq_len = int(cfg["model"]["seq_len"])

    # Select frames
    if args.start_frame is not None:
        start = min(args.start_frame, len(x) - seq_len)
        x_seq = x[start : start + seq_len]
        frame_range = f"frames {start} to {start + seq_len - 1}"
        print(f"Using custom window: {frame_range}")
    else:
        start = max(0, len(x) - seq_len)
        x_seq = x[start:]
        frame_range = f"last {seq_len} frames ({start} to {len(x)-1})"
        print(f"Using default: {frame_range}")

    x_tensor = torch.from_numpy(x_seq).unsqueeze(0).unsqueeze(1).to(device)
    sid_tensor = torch.tensor([sample.subject - 1], dtype=torch.long).to(device)

    def forward_func(x_input):
        bs = x_input.shape[0]
        return model(x_input, sid_tensor.repeat(bs)).squeeze(-1)

    # Prediction
    with torch.no_grad():
        pred_norm = model(x_tensor, sid_tensor).item()
        pred_val = denorm_target(np.array([pred_norm]), target_min, target_max)[0]

    print(f"Predicted Valence: {pred_val:.4f}")

    # Captum
    print("Computing Integrated Gradients...")
    ig = IntegratedGradients(forward_func)
    attributions = ig.attribute(x_tensor, target=None, n_steps=args.n_steps, internal_batch_size=1)

    # Visualization
    attr_np = attributions[0].cpu().detach().numpy()
    input_np = x_tensor[0].cpu().detach().numpy()

    attr_2d = attr_np.mean(axis=(0, 1)).squeeze()
    input_2d = input_np.mean(axis=(0, 1)).squeeze()

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(input_2d, cmap='gray')
    axs[0].set_title("Input Face (averaged)")
    axs[0].axis('off')

    viz.visualize_image_attr(
        np.expand_dims(attr_2d, 2),
        np.expand_dims(input_2d, 2),
        method='heat_map',
        sign='all',
        show_colorbar=True,
        title="Integrated Gradients",
        plt_fig_axis=(fig, axs[1])
    )

    viz.visualize_image_attr(
        np.expand_dims(attr_2d, 2),
        np.expand_dims(input_2d, 2),
        method='blended_heat_map',
        sign='positive',
        show_colorbar=True,
        title="Positive Contribution",
        plt_fig_axis=(fig, axs[2])
    )

    plt.tight_layout()

    save_dir = Path("artifacts/explainability")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"explain_S{args.subject}_T{args.story}_start{start if args.start_frame else 'last'}.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nImage saved to:\n   {save_path}")


if __name__ == "__main__":
    main()
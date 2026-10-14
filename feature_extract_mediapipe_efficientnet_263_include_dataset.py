import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T


DATASET_DIR = Path(r"C:\Users\sudee\Desktop\Include_263")
OUT_DIR = Path(r"C:\Users\sudee\Desktop\include263_mediapipe_efficientnet_features")

N_FRAMES = 30
IMG_SIZE = 224
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def build_mediapipe():
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def build_efficientnet(device):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    feature_dim = model.classifier[1].in_features
    model.classifier = torch.nn.Identity()

    model.eval()
    model.to(device)

    return model, feature_dim


def get_transform():
    return T.Compose([
        T.ToPILImage(),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def collect_samples(dataset_dir):
    samples = []

    for label_dir in sorted(dataset_dir.iterdir()):
        if not label_dir.is_dir() or label_dir.name.startswith("."):
            continue

        label = label_dir.name

        for video_path in sorted(label_dir.iterdir()):
            if not video_path.is_file():
                continue

            if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            samples.append({
                "video_path": video_path,
                "label": label,
            })

    return samples


def sample_frames(video_path, n_frames):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok:
            frame = frames[-1] if frames else np.zeros(
                (IMG_SIZE, IMG_SIZE, 3),
                dtype=np.uint8,
            )

        frames.append(frame)

    cap.release()
    return frames


def mediapipe_fused_frame(frame_bgr, holistic):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    result = holistic.process(rgb)

    skeleton = np.zeros_like(frame_bgr)

    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles
    holistic_module = mp.solutions.holistic

    if result.pose_landmarks:
        drawing.draw_landmarks(
            skeleton,
            result.pose_landmarks,
            holistic_module.POSE_CONNECTIONS,
            landmark_drawing_spec=styles.get_default_pose_landmarks_style(),
        )

    if result.left_hand_landmarks:
        drawing.draw_landmarks(
            skeleton,
            result.left_hand_landmarks,
            holistic_module.HAND_CONNECTIONS,
            landmark_drawing_spec=styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=styles.get_default_hand_connections_style(),
        )

    if result.right_hand_landmarks:
        drawing.draw_landmarks(
            skeleton,
            result.right_hand_landmarks,
            holistic_module.HAND_CONNECTIONS,
            landmark_drawing_spec=styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=styles.get_default_hand_connections_style(),
        )

    fused = cv2.addWeighted(frame_bgr, 0.70, skeleton, 0.90, 0.0)
    return fused


def extract_video_features(
    video_path,
    holistic,
    efficientnet,
    transform,
    device,
    n_frames,
):
    frames = sample_frames(video_path, n_frames)
    tensors = []

    for frame in frames:
        fused_bgr = mediapipe_fused_frame(frame, holistic)
        fused_rgb = cv2.cvtColor(fused_bgr, cv2.COLOR_BGR2RGB)
        tensors.append(transform(fused_rgb))

    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        features = efficientnet(batch)

    return features.cpu().numpy().astype(np.float32)


def make_feature_filename(video_path):
    return f"{video_path.stem}_{video_path.suffix[1:].lower()}.npy"


def save_metadata(samples, label_to_id, out_dir):
    metadata_path = out_dir / "metadata.csv"

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "feature_path",
                "video_path",
                "label",
                "label_id",
            ],
        )

        writer.writeheader()

        for sample in samples:
            writer.writerow({
                "feature_path": sample["feature_path"],
                "video_path": sample["video_path"],
                "label": sample["label"],
                "label_id": label_to_id[sample["label"]],
            })


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = collect_samples(DATASET_DIR)
    labels = sorted({sample["label"] for sample in samples})
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    print(f"Dataset directory : {DATASET_DIR}")
    print(f"Output directory  : {OUT_DIR}")
    print(f"Videos found      : {len(samples)}")
    print(f"Classes used      : {len(labels)}")
    print(f"Device            : {device}")

    holistic = build_mediapipe()
    efficientnet, feature_dim = build_efficientnet(device)
    transform = get_transform()

    print(f"Feature shape per video: ({N_FRAMES}, {feature_dim})")

    processed = 0
    skipped = 0
    failed = 0
    processed_samples = []

    try:
        for idx, sample in enumerate(samples, start=1):
            video_path = sample["video_path"]
            label = sample["label"]

            output_path = OUT_DIR / label / make_feature_filename(video_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.exists():
                skipped += 1
                sample["feature_path"] = str(output_path)
                sample["video_path"] = str(video_path)
                processed_samples.append(sample)
                continue

            try:
                features = extract_video_features(
                    video_path=video_path,
                    holistic=holistic,
                    efficientnet=efficientnet,
                    transform=transform,
                    device=device,
                    n_frames=N_FRAMES,
                )

                if features.shape != (N_FRAMES, feature_dim):
                    raise RuntimeError(f"Bad feature shape: {features.shape}")

                np.save(output_path, features)

                sample["feature_path"] = str(output_path)
                sample["video_path"] = str(video_path)
                processed_samples.append(sample)

                processed += 1

            except Exception as e:
                failed += 1
                print(f"FAILED: {video_path}")
                print(f"Reason: {e}")

            if idx % 25 == 0 or idx == len(samples):
                print(
                    f"{idx}/{len(samples)} "
                    f"processed={processed} "
                    f"skipped={skipped} "
                    f"failed={failed}"
                )

    finally:
        holistic.close()

    save_metadata(processed_samples, label_to_id, OUT_DIR)

    print("Done")
    print(f"Processed: {processed}")
    print(f"Skipped  : {skipped}")
    print(f"Failed   : {failed}")
    print(f"Classes  : {len(labels)}")
    print(f"Metadata : {OUT_DIR / 'metadata.csv'}")


if __name__ == "__main__":
    main()
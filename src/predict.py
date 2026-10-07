import cv2
import torch
import numpy as np
import os

from src.model import CNNLSTM


# -----------------------------
# 1. Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------
# 2. Class names
# -----------------------------

class_names = [
    "Clear",
    "Drive",
    "Drop_Shot",
    "Net_Shot",
    "Smash"
]


# -----------------------------
# 3. Load model
# -----------------------------

model = CNNLSTM(num_classes=5)

model_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "model.pth"
)

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model loaded successfully!")


# -----------------------------
# 4. Prediction function
# -----------------------------

def predict_video(video_path):

    # -----------------------------
    # Open video
    # -----------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video")

    # -----------------------------
    # Get total frames
    # -----------------------------

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print("Total video frames:", total_frames)

    # -----------------------------
    # Select 16 frames
    # -----------------------------

    sequence_length = 16

    if total_frames < sequence_length:

        cap.release()

        raise ValueError(
            "Video has fewer than 16 frames"
        )

    frame_numbers = np.linspace(
        0,
        total_frames - 1,
        sequence_length
    ).astype(int)

    frames = []

    # -----------------------------
    # Read selected frames
    # -----------------------------

    for frame_number in frame_numbers:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        ret, frame = cap.read()

        if not ret:

            cap.release()

            raise ValueError(
                f"Error reading frame {frame_number}"
            )

        # BGR → RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Resize to 224 × 224
        frame = cv2.resize(
            frame,
            (224, 224)
        )

        # Same preprocessing used during training
        frame = frame / 255.0

        frames.append(frame)

    cap.release()

    # -----------------------------
    # Convert frames to NumPy
    # -----------------------------

    frames = np.array(frames)

    # -----------------------------
    # NumPy → PyTorch tensor
    # -----------------------------

    frames = torch.tensor(
        frames,
        dtype=torch.float32
    )

    # Current:
    # [16, 224, 224, 3]

    # Change:
    # [16, 224, 224, 3]
    # →
    # [16, 3, 224, 224]

    frames = frames.permute(
        0,
        3,
        1,
        2
    )

    # -----------------------------
    # Add batch dimension
    # -----------------------------

    frames = frames.unsqueeze(0)

    # Final:
    # [1, 16, 3, 224, 224]

    frames = frames.to(device)

    print("Input shape:", frames.shape)

    # -----------------------------
    # Model prediction
    # -----------------------------

    with torch.no_grad():

        outputs = model(frames)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

    # -----------------------------
    # Confidence
    # -----------------------------

    confidence = (
        probabilities[
            0,
            predicted_class
        ].item() * 100
    )

    # -----------------------------
    # All probabilities
    # -----------------------------

    probability_dict = {
        class_names[i]: round(
            probabilities[
                0,
                i
            ].item() * 100,
            2
        )
        for i in range(len(class_names))
    }

    # -----------------------------
    # Return result
    # -----------------------------

    return {
        "prediction": class_names[predicted_class],
        "confidence": round(
            confidence,
            2
        ),
        "probabilities": probability_dict
    }
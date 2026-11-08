import base64
from io import BytesIO

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

import socketio
import eventlet
import eventlet.wsgi
from flask import Flask


# ============================================================
# MODEL
# ============================================================

class DrivingCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(10560, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        x = torch.flatten(x, 1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        return self.fc4(x)


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = DrivingCNN().to(device)

model.load_state_dict(
    torch.load(
        "best_driving_model.pth",
        map_location=device
    )
)

model.eval()

print("Model loaded on:", device)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess(image):

    # Your training images were 320 x 160.
    # Remove top 55 pixels exactly like during training.
    image = image.crop((0, 65, image.width, image.height))

    image = np.array(image).astype(np.float32)

    # Normalize
    image = image / 255.0

    # HWC -> CHW
    image = np.transpose(image, (2, 0, 1))

    # CHW -> NCHW
    image = np.expand_dims(image, axis=0)

    image = torch.tensor(
        image,
        dtype=torch.float32
    ).to(device)

    return image


# ============================================================
# SIMULATOR SERVER
# ============================================================

sio = socketio.Server(
    logger=True,
    engineio_logger=True
)

app = Flask(__name__)


def send_control(steering, throttle):

    sio.emit(
        "steer",
        data={
            "steering_angle": str(steering),
            "throttle": str(throttle)
        }
    )


@sio.on("connect")
def connect(sid, environ):

    print("Simulator connected!")

    send_control(0, 0)


@sio.on("telemetry")
def telemetry(sid, data):

    if not data:
        return

    # --------------------------------------------------------
    # Get simulator camera image
    # --------------------------------------------------------

    image_data = data["image"]

    image = Image.open(
        BytesIO(
            base64.b64decode(image_data)
        )
    ).convert("RGB")

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    image_tensor = preprocess(image)

    # --------------------------------------------------------
    # Predict steering
    # --------------------------------------------------------

    with torch.no_grad():
        prediction = model(image_tensor)[0]

    steering = prediction[0].item()
    throttle = prediction[1].item()

    # Keep predictions within simulator control limits
    steering = max(-1.0, min(1.0, steering))
    throttle = max(0.0, min(1.0, throttle))

    print(f"Steering: {steering:.3f}, Throttle: {throttle:.3f}")
    # --------------------------------------------------------
    # Throttle
    # --------------------------------------------------------
    # --------------------------------------------------------
    # Send controls to simulator
    # --------------------------------------------------------

    send_control(
        steering,
        throttle
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    app = socketio.Middleware(
        sio,
        app
    )

    print("Waiting for simulator...")

    eventlet.wsgi.server(
        eventlet.listen(("", 4567)),
        app
    )
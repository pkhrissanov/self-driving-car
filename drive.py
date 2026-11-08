
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
# CONFIGURATION
# ============================================================

SPEED_SCALE = 30.0  # Must match the value used during training
CROP_TOP = 65        # Must match the crop used during training


# ============================================================
# MODEL
# ============================================================

class DrivingCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 24, 5, 2)
        self.conv2 = nn.Conv2d(24, 36, 5, 2)
        self.conv3 = nn.Conv2d(36, 48, 5, 2)
        self.conv4 = nn.Conv2d(48, 64, 3, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1)

        # Visual features (10560) + speed (1)
        self.fc1 = nn.Linear(10561, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

        # Steering + throttle
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x, speed):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        x = torch.flatten(x, start_dim=1)

        # Combine image features with speed
        x = torch.cat((x, speed), dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = DrivingCNN().to(device)

model.load_state_dict(
    torch.load(
        "best_driving_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Model loaded on:", device)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess(image):

    image = image.crop(
        (0, CROP_TOP, image.width, image.height)
    )

    image = np.array(image).astype(np.float32)

    image = image / 255.0

    # HWC -> CHW
    image = np.transpose(image, (2, 0, 1))

    # CHW -> NCHW
    image = np.expand_dims(image, axis=0)

    return torch.tensor(
        image,
        dtype=torch.float32,
        device=device
    )


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
        print("No telemetry - resetting controls")
        send_control(0.0, 0.0)
        return

    # --------------------------------------------------------
    # Get simulator image
    # --------------------------------------------------------

    image_data = data["image"]

    image = Image.open(
        BytesIO(base64.b64decode(image_data))
    ).convert("RGB")

    image_tensor = preprocess(image)

    # --------------------------------------------------------
    # Get and normalize current speed
    # --------------------------------------------------------

    current_speed = float(data["speed"])

    normalized_speed = current_speed / SPEED_SCALE

    speed_tensor = torch.tensor(
        [[normalized_speed]],
        dtype=torch.float32,
        device=device
    )

    # --------------------------------------------------------
    # Predict steering and throttle
    # --------------------------------------------------------

    with torch.no_grad():
        prediction = model(
            image_tensor,
            speed_tensor
        )[0]

    steering = prediction[0].item()
    throttle = prediction[1].item()

    # Keep predictions within simulator control limits
    steering = max(-1.0, min(1.0, steering))
    throttle = max(0.0, min(1.0, throttle))

    print(
        f"Speed: {current_speed:.2f}, "
        f"Steering: {steering:.3f}, "
        f"Throttle: {throttle:.3f}"
    )

    # --------------------------------------------------------
    # Send controls
    # --------------------------------------------------------

    send_control(steering, throttle)


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

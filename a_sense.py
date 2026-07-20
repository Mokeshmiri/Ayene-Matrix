#!/usr/bin/env python3
"""
a_sense.py  -  Entity A: face sensing

I read the webcam with MediaPipe Face Landmarker and turn my face into 8
numbers (expression scores). I send those numbers to Wekinator over OSC on
port 6448. Wekinator then classifies them into 4 moods:
Neutral, Happy, Surprised, Focused.

I use MediaPipe "blendshapes" here. Blendshapes are expression scores that the
model already gives me (like mouthSmile, jawOpen, browDown...), each from 0 to
1. I tried building my own features from raw landmark distances first, and that
was more creative but it was sensitive to head movement. The blendshapes are
much more reliable for a live demo, so this final version uses them.

On macOS only one app can use the built-in camera at a time. Processing showed
a black feed when both programs tried to open the webcam. I fixed that here:
only this script opens the camera. I also send a 96x72 brightness grid to
B_Ayene over OSC (/ayene/grid on port 12000) so Processing can draw the mirror
without opening the webcam.

Run:    python3 a_sense.py     (q = quit)
Install: pip install opencv-python mediapipe numpy python-osc
Needs:   face_landmarker.task  (in the same folder)
"""

import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pythonosc import udp_client

# the 8 numbers I send to Wekinator, in this order
LABELS = ["Smile", "JawOpen", "BrowUp", "BrowDown", "Squint", "EyeWide", "Cheek", "Neutral"]
MOODS  = ["Neutral", "Happy", "Surprised", "Focused"]

# brightness grid for B_Ayene (must match B_Ayene.pde)
# finer grid = face reads clearer in the dot-matrix mirror
GRID_W, GRID_H = 96, 72


# MediaPipe face outline — only pixels INSIDE this shape go to Processing
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

BLACK_GRID = np.zeros((GRID_H, GRID_W), dtype=np.uint8).tobytes()


def oval_points(landmarks, w, h):
    """Turn the face-outline landmarks into pixel coordinates."""
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)]
                    for i in FACE_OVAL], dtype=np.int32)
    return pts


def portrait_grid_bytes(frame, landmarks):
    """
    Only my head shape — crop tight around the face outline, mask everything
    outside the real head contour to black. No shoulders, no room, no oval frame.
    """
    if landmarks is None:
        return BLACK_GRID

    fh, fw = frame.shape[:2]
    oval = oval_points(landmarks, fw, fh)

    # tight square crop around the head (not 16:9 portrait)
    x0, y0 = oval.min(axis=0)
    x1, y1 = oval.max(axis=0)
    head = max(x1 - x0, y1 - y0)
    pad = int(head * 0.05)
    size = head + pad * 2
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0 = max(0, cx - size // 2)
    y0 = max(0, cy - size // 2)
    x1 = min(fw, x0 + size)
    y1 = min(fh, y0 + size)
    x0, y0 = max(0, x1 - size), max(0, y1 - size)

    crop = frame[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]

    # hard head mask from the actual face outline
    mask = np.zeros((ch, cw), dtype=np.uint8)
    local = oval.copy()
    local[:, 0] -= x0
    local[:, 1] -= y0
    cv2.fillConvexPoly(mask, local, 255)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray *= mask.astype(np.float32) / 255.0

    inside = gray[mask > 0]
    if inside.size > 0:
        lo, hi = np.percentile(inside, (5, 95))
        span = max(1.0, float(hi - lo))
        gray = np.clip((gray - lo) * 255.0 / span, 0, 255)

    small = cv2.resize(gray.astype(np.uint8), (GRID_W, GRID_H),
                       interpolation=cv2.INTER_AREA)

    # keep the head shape sharp at grid resolution
    mask_s = cv2.resize(mask, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
    _, mask_s = cv2.threshold(mask_s, 80, 255, cv2.THRESH_BINARY)
    small = (small.astype(np.float32) * (mask_s.astype(np.float32) / 255.0))
    return np.clip(small, 0, 255).astype(np.uint8).tobytes()


def draw_head_outline(frame, landmarks):
    """Draw the real head outline on the Python preview."""
    h, w = frame.shape[:2]
    pts = oval_points(landmarks, w, h)
    cv2.polylines(frame, [pts], True, (0, 255, 200), 2)


def clamp(x):
    return max(0.0, min(1.0, x))


def read_blendshapes(result):
    """Put MediaPipe's blendshape scores into a simple name -> value dict."""
    shapes = {}
    if result.face_blendshapes:
        for cat in result.face_blendshapes[0]:
            shapes[cat.category_name] = cat.score
    return shapes


def features(s):
    """
    Pick 8 expression scores and group the left/right ones together.
    Each value is already between 0 and 1, so I do not need any calibration.

      Smile     -> Happy
      JawOpen   -> Surprised
      BrowUp    -> Surprised
      BrowDown  -> Focused
      Squint    -> Focused
      EyeWide   -> Surprised
      Cheek     -> Happy (a real smile also pushes the cheeks up)
      Neutral   -> high when nothing else is active
    """
    smile     = (s.get("mouthSmileLeft", 0) + s.get("mouthSmileRight", 0)) / 2
    jaw_open  =  s.get("jawOpen", 0)
    brow_up   =  s.get("browInnerUp", 0)
    brow_down = (s.get("browDownLeft", 0) + s.get("browDownRight", 0)) / 2
    squint    = (s.get("eyeSquintLeft", 0) + s.get("eyeSquintRight", 0)) / 2
    eye_wide  = (s.get("eyeWideLeft", 0) + s.get("eyeWideRight", 0)) / 2
    cheek     = (s.get("cheekSquintLeft", 0) + s.get("cheekSquintRight", 0)) / 2

    expressive = smile + jaw_open + brow_up + brow_down + squint + eye_wide
    neutral = clamp(1.0 - expressive / 1.5)

    return [smile, jaw_open, brow_up, brow_down, squint, eye_wide, cheek, neutral]


def guess_mood(f):
    """Pick the most likely mood. This is only for the label on the window."""
    scores = [
        f[7],                  # Neutral
        max(f[0], f[6]),       # Happy     (smile / cheeks)
        max(f[1], f[2], f[5]), # Surprised (jaw open / brows up / eyes wide)
        max(f[3], f[4]),       # Focused   (brows down / squint)
    ]
    return int(np.argmax(scores))


def draw_face(frame, landmarks):
    """Draw the tracked face points so I can see the tracking is working."""
    h, w = frame.shape[:2]
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 230, 0), -1)


def draw_bars(frame, f, mood):
    """Show the current mood on top and the 8 feature bars at the bottom."""
    h, w = frame.shape[:2]
    cv2.putText(frame, "MOOD: " + MOODS[mood], (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    base_y = h - 22
    for i, val in enumerate(f):
        x = 15 + i * ((w - 30) // 8)
        cv2.rectangle(frame, (x, base_y - 70), (x + 26, base_y), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, base_y - int(val * 70)), (x + 26, base_y),
                      (0, 200, 255), -1)
        cv2.putText(frame, LABELS[i], (x - 6, base_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1)


def main():
    # 1) OSC out: Wekinator (features) + Processing (brightness grid for the dots)
    wek_client = udp_client.SimpleUDPClient("127.0.0.1", 6448)
    vis_client = udp_client.SimpleUDPClient("127.0.0.1", 12000)

    # 2) load the face model and turn ON blendshapes (the expression scores)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        output_face_blendshapes=True,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # 3) open the camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[a_sense] Running. Look at the camera. Press q to quit.")
    print("[a_sense] OSC -> Wekinator /wek/inputs :6448")
    print("[a_sense] OSC -> B_Ayene     /ayene/grid :12000  (%dx%d)" % (GRID_W, GRID_H))

    f = [0, 0, 0, 0, 0, 0, 0, 1]
    mood = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)               # mirror, like a real mirror

        # --- detect face and build the 8 Wekinator features ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        landmarks = result.face_landmarks[0] if result.face_landmarks else None

        if landmarks:
            draw_face(frame, landmarks)
            draw_head_outline(frame, landmarks)
            f = features(read_blendshapes(result))
            mood = guess_mood(f)
        else:
            f = [0, 0, 0, 0, 0, 0, 0, 1]         # no face -> neutral
            mood = 0

        # --- send OSC every frame: features to Wekinator, portrait to Processing ---
        wek_client.send_message("/wek/inputs", [float(v) for v in f])
        vis_client.send_message("/ayene/grid", portrait_grid_bytes(frame, landmarks))

        draw_bars(frame, f, mood)
        cv2.imshow("ayene-matrix  a_sense", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()

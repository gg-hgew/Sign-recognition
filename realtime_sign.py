import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers

# =======================
# 1. Actions (classes)
# =======================
actions = np.array(['cat', 'food', 'help'])  # must match training

# =======================
# 2. Legacy LSTM to fix old model config (time_major issue)
# =======================
class LegacyLSTM(layers.LSTM):
    def __init__(self, *args, **kwargs):
        # Drop any 'time_major' argument from old Keras configs
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)

# =======================
# 3. MediaPipe setup
# =======================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model_mp):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model_mp.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    # Face
    mp_drawing.draw_landmarks(
        image, results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION
    )
    # Pose
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS
    )
    # Left hand
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )
    # Right hand
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )

def extract_keypoints(results):
    pose = np.array([[r.x, r.y, r.z, r.visibility]
                     for r in getattr(results, "pose_landmarks", []).landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[r.x, r.y, r.z]
                     for r in getattr(results, "face_landmarks", []).landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[r.x, r.y, r.z]
                   for r in getattr(results, "left_hand_landmarks", []).landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[r.x, r.y, r.z]
                   for r in getattr(results, "right_hand_landmarks", []).landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# =======================
# 4. Load the trained model
# =======================
print("🔁 Loading model...")
model = tf.keras.models.load_model(
    "model.h5",
    custom_objects={"LSTM": LegacyLSTM}
)
print("✅ Model loaded!")

# =======================
# 5. Realtime sign detection
# =======================
sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam. Check your camera or permissions.")
    exit()

with mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    print("🎥 Realtime LSTM Sign Detection running. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from webcam.")
            break

        # Resize for consistency (optional)
        frame = cv2.resize(frame, (640, 480))

        # 1. Mediapipe detection
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        # 2. Build sequence of last 30 frames
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        # 3. Predict when we have 30 frames
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            pred_class = np.argmax(res)
            predictions.append(pred_class)

            # Simple smoothing over last 10 predictions
            if len(predictions) >= 10 and np.unique(predictions[-10:])[0] == pred_class:
                if res[pred_class] > threshold:
                    if len(sentence) == 0 or actions[pred_class] != sentence[-1]:
                        sentence.append(actions[pred_class])

            sentence = sentence[-5:]

        # 4. Draw sentence bar
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(
            image,
            " ".join(sentence),
            (3, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # 5. Show window
        cv2.imshow("Realtime LSTM Sign Detection", image)

        # Exit on 'q' key
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print("👋 Closed.")

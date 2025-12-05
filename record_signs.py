import os
import cv2
import numpy as np
import mediapipe as mp

# =========================
# CONFIG
# =========================
# These are the 3 signs you'll record
actions = np.array(['cat', 'food', 'help'])

# How many sequences (short videos) per sign
no_sequences = 30  # you can change to 20 if 30 feels too much

# Frames per sequence
sequence_length = 30

# Where to save keypoints
DATA_PATH = os.path.join('MP_Data_Custom')

# =========================
# Mediapipe setup
# =========================
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
    mp_drawing.draw_landmarks(
        image, results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION
    )
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )
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

# =========================
# Create folders
# =========================
for action in actions:
    for sequence in range(no_sequences):
        dirpath = os.path.join(DATA_PATH, action, str(sequence))
        os.makedirs(dirpath, exist_ok=True)

print("📂 Folder structure created in:", DATA_PATH)

# =========================
# Capture data
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

with mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    print("🎥 Starting data collection.")
    print("Instructions:")
    print("- For each sign, you'll record", no_sequences, "short sequences.")
    print("- Follow the text on the top-left of the video window.")
    print("- Press 'q' any time to stop early.")

    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame from webcam.")
                    break

                frame = cv2.resize(frame, (640, 480))

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                # Display status text
                if frame_num == 0:
                    # Show "STARTING COLLECTION" for 1 second at start of each sequence
                    cv2.putText(
                        image, 'STARTING COLLECTION',
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA
                    )
                    cv2.putText(
                        image,
                        f'Action: {action} | Sequence: {sequence + 1}/{no_sequences}',
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA
                    )
                    cv2.imshow('Data Collection', image)
                    cv2.waitKey(1000)  # 1 second pause

                else:
                    cv2.putText(
                        image,
                        f'Action: {action} | Sequence: {sequence + 1}/{no_sequences}',
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA
                    )
                    cv2.putText(
                        image,
                        f'Frame: {frame_num + 1}/{sequence_length}',
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA
                    )
                    cv2.imshow('Data Collection', image)

                # Save keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(
                    DATA_PATH, action, str(sequence), f"{frame_num}.npy"
                )
                np.save(npy_path, keypoints)

                # Allow early quit
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("🛑 Early stop by user.")
                    exit()

    cap.release()
    cv2.destroyAllWindows()

print("✅ Data collection complete!")

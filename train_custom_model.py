import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

# =========================
# Config (must match recording)
# =========================
DATA_PATH = os.path.join('MP_Data_Custom')
actions = np.array(['cat', 'food', 'help'])
sequence_length = 30

# =========================
# Load data
# =========================
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        raise RuntimeError(f"No data found for action '{action}' in {action_path}")

    for sequence in os.listdir(action_path):
        seq_dir = os.path.join(action_path, sequence)
        if not os.path.isdir(seq_dir):
            continue

        window = []
        for frame_num in range(sequence_length):
            npy_path = os.path.join(seq_dir, f"{frame_num}.npy")
            if not os.path.exists(npy_path):
                raise RuntimeError(f"Missing file: {npy_path}")
            res = np.load(npy_path)
            window.append(res)

        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print("✅ Data loaded!")
print("X shape:", X.shape)  # (num_samples, 30, 1662)
print("y shape:", y.shape)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=42
)

# =========================
# Build model
# =========================
num_features = X.shape[2]
num_classes = actions.shape[0]

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, num_features)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

model.summary()

# =========================
# Train
# =========================
EPOCHS = 200  # start with 200; you can increase later if needed

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    validation_data=(X_test, y_test)
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test accuracy: {acc:.3f}")

# =========================
# Save model
# =========================
model.save('my_model.h5')
print("💾 Saved trained model to my_model.h5")

# ASL Fingerspelling Model Demonstration
# Created by Reyansh Gupta, 2026
# This program is available at https://github.com/gupta-reyansh/asl-fingerspelling

# Import all necessary libraries
import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from autocorrect import Speller

begin=0
end=None
spell=Speller(lang='en')
interpreter = tf.lite.Interpreter('Model.tflite',experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
interpreter.allocate_tensors()

REQUIRED_SIGNATURE = "serving_default"
REQUIRED_OUTPUT = "outputs"

with open ("Data/character_to_prediction_index.json", "r") as f:
    character_map = json.load(f)
rev_character_map = {j:i for i,j in character_map.items()}
found_signatures = list(interpreter.get_signature_list().keys())
if REQUIRED_SIGNATURE not in found_signatures:
    raise NameError('Required input signature not found.')

prediction_fn = interpreter.get_signature_runner("serving_default")

#Initialize Imported Classes
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
custom_dots = mp_drawing.DrawingSpec(color=(188, 112, 255), thickness=2, circle_radius=2)

# Initialize mediapipe holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# For FPS calculation
pTime = 0

# Storage for all frames
data_2d = []

# Capture state
capturing = False
frames_captured = 0
FRAMES_TO_CAPTURE = 768

def extract_landmarks(results):
    frame_row = []
    
    # Extract all X coordinates
    # Face (468 landmarks)
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            frame_row.append(lm.x)
    else:
        frame_row.extend([0.0] * 468)
    
    # Left hand (21 landmarks)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            frame_row.append(lm.x)
    else:
        frame_row.extend([0.0] * 21)
    
    # Pose (33 landmarks)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            frame_row.append(lm.x)
    else:
        frame_row.extend([0.0] * 33)
    
    # Right hand (21 landmarks)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            frame_row.append(lm.x)
    else:
        frame_row.extend([0.0] * 21)
    
    # Extract all Y coordinates
    # Face (468 landmarks)
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            frame_row.append(lm.y)
    else:
        frame_row.extend([0.0] * 468)
    
    # Left hand (21 landmarks)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            frame_row.append(lm.y)
    else:
        frame_row.extend([0.0] * 21)
    
    # Pose (33 landmarks)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            frame_row.append(lm.y)
    else:
        frame_row.extend([0.0] * 33)
    
    # Right hand (21 landmarks)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            frame_row.append(lm.y)
    else:
        frame_row.extend([0.0] * 21)
    
    # Extract all Z coordinates
    # Face (468 landmarks)
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            frame_row.append(lm.z)
    else:
        frame_row.extend([0.0] * 468)
    
    # Left hand (21 landmarks)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            frame_row.append(lm.z)
    else:
        frame_row.extend([0.0] * 21)
    
    # Pose (33 landmarks)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            frame_row.append(lm.z)
    else:
        frame_row.extend([0.0] * 33)
    
    # Right hand (21 landmarks)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            frame_row.append(lm.z)
    else:
        frame_row.extend([0.0] * 21)
    
    return frame_row

def create_column_names():
    columns = []
    # Creates column names for the file
    for i in range(468):
        columns.append(f'x_face_{i}')
    for i in range(21):
        columns.append(f'x_left_hand_{i}')
    for i in range(33):
        columns.append(f'x_pose_{i}')
    for i in range(21):
        columns.append(f'x_right_hand_{i}')
    for i in range(468):
        columns.append(f'y_face_{i}')
    for i in range(21):
        columns.append(f'y_left_hand_{i}')
    for i in range(33):
        columns.append(f'y_pose_{i}')
    for i in range(21):
        columns.append(f'y_right_hand_{i}')
    for i in range(468):
        columns.append(f'z_face_{i}')
    for i in range(21):
        columns.append(f'z_left_hand_{i}')
    for i in range(33):
        columns.append(f'z_pose_{i}')
    for i in range(21):
        columns.append(f'z_right_hand_{i}')
    return columns

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    # Draw landmarks
    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,landmark_drawing_spec=custom_dots)
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,landmark_drawing_spec=custom_dots)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,landmark_drawing_spec=custom_dots)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,landmark_drawing_spec=custom_dots)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == 32:  
        # Press SPACE to start capturing 300 frames
        if not capturing:
            capturing = True
            frames_captured = 0
            data_2d = []
            print("Starting capture of 500 frames...")
    elif key == 27:  
        # Press ESC to exit
        capturing = False
        # Convert to dataframe and save
        if len(data_2d) > 0:
            data_2d = np.array(data_2d)
            columns = create_column_names()
            df = pd.DataFrame(data_2d, columns=columns)
            df.to_parquet("output.parquet", index=False)
        break

    # Capture frames if in capture mode
    if capturing:
        frame_row = extract_landmarks(results)
        data_2d.append(frame_row)
        frames_captured += 1
        
        # Display capture progress
        cv2.putText(frame, f'Capturing: {frames_captured}/{FRAMES_TO_CAPTURE}', (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Check if capture is complete
        if frames_captured >= FRAMES_TO_CAPTURE:
            capturing = False
            # Convert to dataframe and save
            data_2d = np.array(data_2d)
            columns = create_column_names()
            df = pd.DataFrame(data_2d, columns=columns)
            df.to_parquet("output.parquet", index=False)

    # FPS display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Holistic", frame)
cap.release()
cv2.destroyAllWindows()
# ------------------------------ Model Prediction Begins ------------------------------#

data=pd.read_parquet('output.parquet')
data=data.iloc[begin:end]
data=tf.cast(data, tf.float32)
raw_output = prediction_fn(inputs=data)
probs = np.array(raw_output[REQUIRED_OUTPUT], copy=True)
prediction_str = "".join([rev_character_map.get(int(s), "") for s in np.argmax(probs, axis=1)])

print('------------------------------------------------------------------------------------------------')
print('Output without autocorrect:', prediction_str)
print('Output with autocorrect:', spell(prediction_str))
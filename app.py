import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle

# --- Setup MediaPipe ---
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# --- Load Model ---
@st.cache_resource
def load_model():
    with open('body_language.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: 'body_language.pkl' not found. Please ensure your trained model is in the same directory.")
    st.stop()

# --- Streamlit UI ---
st.title("Body Language Detection App")
st.markdown("This app uses MediaPipe Holistic and a custom Scikit-Learn model to detect body language in real-time.")

# Checkbox to start/stop the webcam
run = st.checkbox('Start Webcam')

# Placeholder for the video frame
frame_window = st.image([])

# --- Main Video Loop ---
if run:
    cap = cv2.VideoCapture(0)
    
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video.")
                break
            
            # Recolor Feed for MediaPipe (BGR to RGB)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        
            
            # Make Detections
            results = holistic.process(image)
            
            # Recolor image back to BGR for OpenCV drawing
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 1. Draw face landmarks
            if results.face_landmarks:
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                         mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                         mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                         )
            
            # 2. Right hand
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                         mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                         )

            # 3. Left Hand
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                         mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                         )

            # 4. Pose Detections
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                         mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                         )
            
            # --- Feature Extraction and Prediction ---
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                
                # Concatenate rows
                row = pose_row + face_row
                
                # Make Prediction
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                
                # --- Overlay Prediction on Video ---
                # Get status box
                cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                
                # Display Class
                cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(image, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                max_prob = str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                cv2.putText(image, max_prob, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
            except Exception as e:
                # Passes if no landmarks are detected in the frame
                pass
            
            # Convert BGR back to RGB for Streamlit rendering
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Update the Streamlit image placeholder
            frame_window.image(image_rgb)
            
    # Release camera when loop ends
    cap.release()
else:
    st.write("Click 'Start Webcam' to begin.")
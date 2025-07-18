import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load your trained LRCN model
model = load_model("path_to_your_model.h5")

# Function to preprocess frames for the LRCN model
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))  # Resize to match model input
    frame = frame / 255.0  # Normalize pixel values
    return frame

# Initialize the webcam (0 is the default camera, change if needed)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera is active. Press 'q' to quit.")

# Buffer to store frames for LRCN input
sequence = []
sequence_length = 20  # Adjust based on your model's input requirements

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the live frame
        cv2.imshow('Live Video Feed', frame)

        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)
        sequence.append(preprocessed_frame)

        # Keep the sequence length fixed
        if len(sequence) > sequence_length:
            sequence.pop(0)

        # Run prediction when the sequence is ready
        if len(sequence) == sequence_length:
            input_sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
            prediction = model.predict(input_sequence)
            label = "Robbery Detected" if np.argmax(prediction) == 1 else "No Robbery"

            # Display prediction on the frame
            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Show the processed frame with predictions
        cv2.imshow('Detection Output', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user. Closing...")

finally:
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

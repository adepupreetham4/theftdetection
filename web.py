import os
import cv2
import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
import smtplib
import logging
from imutils.video import VideoStream
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from moviepy.editor import ColorClip
from email.mime.multipart import MIMEMultipart
from flask import Flask, redirect, url_for
import webbrowser
import requests
import urllib3 
import time
from moviepy.editor import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.basicConfig(level=logging.DEBUG)
# Set random seeds for reproducibility
seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

# Create a Matplotlib figure and specify the size of the figure
plt.figure(figsize=(20, 20))

# Get the names of all classes/categories in UCF50
all_classes_names = os.listdir(r"C:\Users\adepu\OneDrive\Desktop\theft detection\data")

# Generate a list of 13 random values between 0-50 (total number of classes)

all_classes_names = [...]  # Your list of class names
random_range=[]
if len(all_classes_names) < 13:
    pass
else:qq
    sample_size = min(len(all_classes_names), 13)
    random_range = random.sample(range(len(all_classes_names)), sample_size)

# Iterating through all the generated random values
for counter, random_index in enumerate(random_range, 1):
    # Retrieve a Class Name using the Random Index
    selected_class_Name = all_classes_names[random_index]

    # Retrieve the list of all video files present in the randomly selected Class Directory
    video_files_names_list = os.listdir(r"C:\Users\adepu\OneDrive\Desktop\theft detection\data\UCF50\UCF50\\{selected_class_Name}")
    
    # Randomly select a video file from the list
    selected_video_file_name = random.choice(video_files_names_list)

    # Initialize a VideoCapture object to read from the video file
    video_reader = cv2.VideoCapture(r'C:\Users\adepu\OneDrive\Desktop\theft detection\data\UCF50\UCF50\\{selected_class_Name}\\{selected_video_file_name}')
    
    # Read the first frame of the video
    _, bgr_frame = video_reader.read()

    # Release the VideoCapture object
    video_reader.release()

    # Convert the frame from BGR into RGB format
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    # Write the class name on the video frame
    cv2.putText(rgb_frame, selected_class_Name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    plt.subplot(5, 4, counter)
    plt.imshow(rgb_frame)
    plt.axis('off')

# Specify the height and width for video frame resizing
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

def preprocess_frames(frames):
    frames = frames.astype('float32') / 255.0
    return frames

# Specify the number of frames of a video to be fed to the model as one sequence
SEQUENCE_LENGTH = 15

# Directory containing the UCF50 dataset
DATASET_DIR = r"C:\Users\adepu\OneDrive\Desktop\theft detection\data"

# List of class names used for training
CLASSES_LIST = ["Robbery","Fighting", "Running", "Firing"]

def frames_extraction(video_path):
    # List to store video frames
    frames_list = [] 
    
    # Read the Video File
    video_reader = cv2.VideoCapture(video_path) 
    
    # Get total number of frames in the video
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)
    
    # Iterate through video frames
    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        #Resize and normalize frame
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)
    
    # Release VideoCapture object
    video_reader.release()
    
    return frames_list

def create_dataset():
    features = []
    labels = []
    video_files_paths = []
    
    # Iterate through all classes
    for class_index, class_name in enumerate(CLASSES_LIST):
        print(f'Extracting Data of Class: {class_name}')
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))
        
        for file_name in files_list:
            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)
            frames = frames_extraction(video_file_path)
            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)
    
    features = np.asarray(features)
    labels = np.array(labels)
    
    return features, labels, video_files_paths

# Create the dataset
features, labels, video_files_paths = create_dataset()
# One-hot encode the labels
one_hot_encoded_labels = to_categorical(labels)

# Split the data into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size=0.25, shuffle=True, random_state=seed_constant)

# Manually split data
features_train_split = features_train[:int(0.8 * len(features_train))]
labels_train_split = labels_train[:int(0.8 * len(labels_train))]
features_val_split = features_train[int(0.8 * len(features_train)):]
labels_val_split = labels_train[int(0.8 * len(labels_train)):]

def create_LRCN_model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'), input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32))
    model.add(Dense(len(CLASSES_LIST), activation='softmax'))
    
    model.summary()
    return model

# Create LRCN model
LRCN_model = create_LRCN_model()
print("Model Created Successfully!")

# Early stopping callback
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)

# Compile the model
LRCN_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

# Train the model

LRCN_model_training_history = LRCN_model.fit(x=features_train, y=labels_train, epochs=70, batch_size=4, shuffle=True, validation_split=0.2, callbacks=[early_stopping_callback])

# Fit the model
LRCN_model_training_history = LRCN_model.fit(
    x=features_train,
    y=labels_train,
    epochs=70,
    batch_size=4,
    shuffle=True,
    validation_data=(features_val_split, labels_val_split),
    callbacks=[early_stopping_callback]
)

# Evaluate the model
model_evaluation_history = LRCN_model.evaluate(features_test, labels_test)
model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history

# Define date format
date_time_format = '%Y_%m_%d__%H_%M_%S'
current_date_time_dt = dt.datetime.now()
current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)

# Model file name
model_file_name = f'LRCN_model___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.h5'

# Save the model
LRCN_model.save('my_model.keras')

def plot_metric(training_history, metric, val_metric, title):
    plot_metric(LRCN_model_training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')
    plot_metric(LRCN_model_training_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')
def plot_metric(training_history, metric, val_metric, title):
    plt.plot(training_history.history[metric])
    plt.plot(training_history.history[val_metric])
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["Training", "Validation"])
    plt.show()

test_videos_directory = 'test_videos'
os.makedirs(test_videos_directory, exist_ok=True)

# Input video file path
input_video_file_path = r"C:\Users\adepu\OneDrive\Desktop\theft detection\test_videos\Burglary028_x264.mp4"

# SMTP configuration
smtp_ssl_host = 'smtp.gmail.com'  # Example: smtp.mail.yahoo.com
smtp_ssl_port = 465
username = 'rahulnaiks779@gmail.com'
password = 'rahul@naiks_779'
sender = 'rahulnaiks779@gmail.com'
targets = ['adepupreetham7@gmail.com']

# Load Yolo
# Download the weight file (yolov3_training_2000.weights) from this link:
# https://drive.google.com/file/d/10uJEsUpQI3EmD98iwrwzbD4e19Ps-LHZ/view?usp=sharing

net = cv2.dnn.readNet(r"C:\Users\adepu\OneDrive\Desktop\theft detection\yolov3_training_2000.weights",r"C:\Users\adepu\OneDrive\Desktop\theft detection\yolov3.cfg")

# Define the classes
classes = ["Weapon"]


# Get layer names and output layers from the network
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Generate random colors for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Define VideoCamer class
class VideoCamer(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.app = None

    def __del__(self):
        self.video.release()

    # Enter file name for example "ak47.mp4" or press "Enter" to start webcam
    def value(self):
        val = input("Enter file name or press Enter to start webcam: \n")
        if val == "":
            val = 0
        return val

    def get_frame(self):
        # Initialize detection and email alert flags
        flag = 0
        cr = 0

        ret, img = self.video.read()
        if not ret:
            print("Error: Unable to read video frame.")
            return None
        
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indexes) > 0:
            print("Weapon detected in frame")
            flag = 1

        if flag == 1 and cr == 0:
            print("Weapon detected! Sending email...")
            name = r"C:\Users\adepu\Downloads\metadata.jpg"
            cv2.imwrite(name, img)
            time.sleep(10)

            # Open alert URL
            url = "http://127.0.0.1:5100/alert"
            webbrowser.open(url, new=2)

            # Sending alert email
            msg = MIMEMultipart()
            msg['Subject'] = 'Alert: Weapon Detected'
            msg['From'] = sender
            msg['To'] = ', '.join(targets)

            # Add the text content to the email
            txt = MIMEText('There is a robbery in the bank!!')
            msg.attach(txt)

            # Attach the image file
            filepath = r"C:\\Users\\adepu\\Downloads\\metadata.jpg"
            try:
                with open(filepath, 'rb') as f:
                    imgi = MIMEImage(f.read())
                    imgi.add_header('Content-Disposition', 'attachment', filename=os.path.basename(filepath))
                    msg.attach(imgi)
            except Exception as e:
                print(f"Error attaching image: {e}")

            # Send the email
            try:
                server = smtplib.SMTP_SSL(smtp_ssl_host, smtp_ssl_port)
                server.login(username, password)
                server.sendmail(sender, targets, msg.as_string())
                server.quit()
                print("Email sent successfully!")
            except Exception as e:
                print(f"Error sending email: {e}")

            # Mark that an email has been sent
            cr = 1

        # Draw bounding boxes for detected objects
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]

                # Draw the rectangle around the detected object
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                # Add the label to the detected object
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

        # Encode the processed image as JPEG
        ret, jpeg = cv2.imencode('.jpg', img)
        if ret:
            return jpeg.tobytes()
        else:
            print("Error: Unable to encode frame as JPEG.")
            return None

THRESHOLD = 0.7  # Confidence threshold for predictions

def predict_on_camera(SEQUENCE_LENGTH):
    video_reader = cv2.VideoCapture(0)
    if not video_reader.isOpened():
        print("Error: Unable to access the camera.")
        return

    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''
    predicted_labels_probabilities = np.zeros(len(CLASSES_LIST))  # Initialize with zeros

    while True:
        ok, frame = video_reader.read()
        if not ok:
            print("Error: Unable to read from the camera.")
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            
            # Apply threshold filtering
            if np.max(predicted_labels_probabilities) < THRESHOLD:
                predicted_class_name = "No Action"
            else:
                predicted_label = np.argmax(predicted_labels_probabilities)
                predicted_class_name = CLASSES_LIST[predicted_label]

        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Live Video Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_reader.release()
    cv2.destroyAllWindows()

# Call the function
predict_on_camera(SEQUENCE_LENGTH)


# Output video path (make sure to define current_date_time_string)
current_date_time_string = time.strftime("%Y%m%d_%H%M%S")
output_video_file_path = r"C:\\Users\\adepu\\OneDrive\\Desktop\\theft detection\\data\\newoutput.mp4"


output_path ="C:\\Users\\adepu\\OneDrive\\Desktop\\theft detection\\data\\newoutput.mp4"



# Create a 10-second red color video clip with resolution 300x300
video = ColorClip(size=(300, 300), color=(255, 0, 0), duration=10)  # Red color
video = video.set_fps(24)  # Set frame rate

# Specify output path
output_path ="C:\\Users\\adepu\\OneDrive\\Desktop\\theft detection\\data\\newoutput.mp4"


# Attempt to write video
try:
    video.write_videofile(output_path)
    print("Video saved successfully.")
except Exception as e:
    print(f"Error saving video: {e}")

# Save the video with fps=24
video.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac')


# Run prediction on video
test_video_file_path=r"C:\Users\adepu\OneDrive\Desktop\theft detection\test_videos\1108331275-preview.mp4"
  # Update with actual test video path
SEQUENCE_LENGTH = 30  # Adjust as needed
predict_on_camera(SEQUENCE_LENGTH)
# Display the output video
cap = cv2.VideoCapture(output_video_file_path)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('Output', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
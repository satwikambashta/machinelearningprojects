import os
import cv2
import face_recognition
import IPython.display as display
from PIL import Image
from io import BytesIO
import time

def load_known_faces(folder_path):
    known_face_encodings = []
    known_face_labels = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            sample_image = face_recognition.load_image_file(file_path)
            sample_encoding = face_recognition.face_encodings(sample_image)[0]
            known_face_encodings.append(sample_encoding)
            known_face_labels.append(file_name.split(".")[0])  # Extract label from filename

    return known_face_encodings, known_face_labels

def auto_adjust_color(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    adjusted_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return adjusted_image


known_faces_folder = "path_to_known_faces_folder" #change this

known_face_encodings, known_face_labels = load_known_faces(known_faces_folder)

video_capture = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = video_capture.read()

        frame = auto_adjust_color(frame)

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_labels[first_match_index]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the live video feed
        image = Image.fromarray(rgb_frame)
        img_buffer = BytesIO()
        image.save(img_buffer, format="png")
        display.display(display.Image(data=img_buffer.getvalue()))
        time.sleep(0.1)

except KeyboardInterrupt:
    video_capture.release()
    print("Video capture released.")

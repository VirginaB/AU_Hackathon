import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Constants
MIN_ATTENDANCE_DURATION = 10  # 10 seconds
SCHEDULED_CLASS_TIME = "13:00:00"  # Replace with your scheduled class time
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.5
FONT_COLOR = (255, 0, 0)
THICKNESS = 3
LINE_TYPE = 2
MOBILE_CLASS_NAMES = ["cell phone", "mobile phone"]
MOBILE_CSV_FILE = "mobile_usage.csv"
absent_students = []
scheduled_time = datetime.strptime(SCHEDULED_CLASS_TIME, "%H:%M:%S")

# List of student names and their corresponding image filenames
students = {
    "tesla": "tesla.jpg",
    "jobs": "jobs.jpg",
    "mona": "mona.jpg",
    "Benadi":"Benadi.jpg",
    "dhyanesh":"dhyanesh.jpg"
}

# Initialize video capture
video_capture = cv2.VideoCapture(0)

current_date = datetime.now().strftime("%Y-%m-%d")

# Create a single CSV file for the day
csv_file_name = f"{current_date}_attendance.csv"

# Create a set to keep track of students already marked as present
present_students = set()

# Create a CSV file to track mobile device usage
with open(MOBILE_CSV_FILE, 'w', newline='') as mobile_csv_file:
    mobile_csv_writer = csv.writer(mobile_csv_file)
    mobile_csv_writer.writerow(["Student Name", "Mobile Device Usage"])

# Create a dictionary to store the time when each student's face was first detected
student_first_detection_time = {}

# Load face encodings and names for known faces
known_face_encodings = []
known_face_names = []

for name, image_filename in students.items():
    image_path = image_filename
    if os.path.exists(image_path):
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append(name)

# Load Mobile Detection Model
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)



while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error capturing video.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

        if name in students and name not in present_students:
            current_time = datetime.now().strftime("%H:%M:%S")  # Get the current time for this student

            # Check if the student attended at least MIN_ATTENDANCE_DURATION seconds
            if name not in student_first_detection_time:
                student_first_detection_time[name] = datetime.now()
            else:
                elapsed_time = (datetime.now() - student_first_detection_time[name]).total_seconds()
                if elapsed_time >= MIN_ATTENDANCE_DURATION:
                    cv2.putText(frame, f"{name} Present", (10, 100), FONT, FONT_SCALE, FONT_COLOR, THICKNESS, LINE_TYPE)
                    print(f"{name} Present")
                    with open(csv_file_name, 'a+', newline='') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        if datetime.now().time() > scheduled_time.time():
                            late_flag = "Late"
                        else:
                            late_flag = ""
                        csv_writer.writerow([name, current_time, f"Present ({late_flag})"])
                    # Add the student to the set of present students
                    present_students.add(name)
                else:
                    cv2.putText(frame, f"{name} Detected", (10, 100), FONT, FONT_SCALE, FONT_COLOR, THICKNESS, LINE_TYPE)

    classIds, confs, bbox = net.detect(frame, confThreshold=0.65)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]

            # Check if the detected object is a mobile device
            if className.lower() in MOBILE_CLASS_NAMES:
                x, y, w, h = box
                center_x = x + w // 2
                center_y = y + h // 2

                # Find the closest student to the mobile device
                closest_student = None
                closest_distance = float('inf')

                for student_name, student_face_location in zip(face_names, face_locations):
                    student_face_x, student_face_y, _, _ = student_face_location
                    distance = np.sqrt((center_x - student_face_x) ** 2 + (center_y - student_face_y) ** 2)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_student = student_name

                if closest_student:
                    # Record the mobile device usage
                    current_time = datetime.now().strftime("%H:%M:%S")  # Get the current time
                    with open(MOBILE_CSV_FILE, 'a+', newline='') as mobile_csv_file:
                        mobile_csv_writer = csv.writer(mobile_csv_file)
                        mobile_csv_writer.writerow([closest_student, current_time])

                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=3)
                cv2.putText(frame, className, (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

    # Create a separate CSV file for absentees
    absentees_csv_file_name = f"{current_date}_absentees.csv"

    try:
        # Check for absentees and write them to the absentees CSV file
        with open(absentees_csv_file_name, 'w', newline='') as absentees_csv_file:
            absentees_csv_writer = csv.writer(absentees_csv_file)
            absentees_csv_writer.writerow(["Absent Students"])

            for student_name in students.keys():
                if student_name not in present_students:
                    absentees_csv_writer.writerow([student_name])

        print("Absentees have been recorded in:", absentees_csv_file_name)
    except Exception as e:
        print("An error occurred while creating the absentees file:", str(e))

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()

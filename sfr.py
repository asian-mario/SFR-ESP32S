import cv2
import os
import numpy as np
import face_recognition
import glob
import time
from flask import Flask, Response, render_template_string

app = Flask(__name__)

ESP32_STREAM_URL = "http://192.168.0.74:81/stream"

# GLOBAL CACHES (to reduce unnecessary computations)
known_face_encodings = []
known_face_names = []
last_detected_faces = {}  # Stores last seen faces to prevent reprocessing
FRAME_SKIP = 3  # Skip every 3 frames for speed

# --- Load Faces ---
enrollment_files = glob.glob("./data/*.jpg")

if not enrollment_files:
    print("❌ No enrollment images found.")
else:
    for file in enrollment_files:
        try:
            name = os.path.basename(file).rsplit(".", 1)[0]  # Remove file extension
            name = name.replace("_", " ")  # Convert underscores to spaces
            image = face_recognition.load_image_file(file)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                if name not in last_detected_faces:
                    last_detected_faces[name] = []

                last_detected_faces[name].extend(encodings)
                print(f"✅ Enrolled {len(encodings)} faces for {name} from {file}")

        except Exception as e:
            print(f"❌ Error loading {file}: {e}")

# Store final encodings
for name, encodings in last_detected_faces.items():
    known_face_encodings.extend(encodings)
    known_face_names.extend([name] * len(encodings))

print(f"🚀 Enrolled {len(known_face_encodings)} faces for {len(set(known_face_names))} people.")

# FPS Tracker
frame_count = 0
last_time = time.time()


def gen_frames():
    """ Stream processing for ESP32-CAM """
    global frame_count, last_time

    cap = cv2.VideoCapture(ESP32_STREAM_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Error: Cannot open ESP32-CAM stream")
        return

    while True:
        frame_count += 1
        success, frame = cap.read()
        if not success:
            continue


        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        # face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")  # Use if you have GPU

        face_encodings = []

        for loc in face_locations:
            try:
                if not isinstance(loc, tuple) or len(loc) != 4 or not all(isinstance(x, int) for x in loc):
                    continue

                frame_uint8 = np.array(rgb_small_frame, dtype=np.uint8)
                encodings = face_recognition.face_encodings(frame_uint8, [loc], num_jitters=1)  # Lower num_jitters

                if encodings:
                    face_encodings.append(encodings[0])

            except Exception as e:
                print(f"❌ Face encoding error: {e}")

        # Match faces
        tolerance = 0.6
        face_names = []

        for face_encoding in face_encodings:
            if not known_face_encodings:
                face_names.append("Unknown")
                continue

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < tolerance:
                name = known_face_names[best_match_index]
                print("Recognized: " + name)
            else:
                name = "Unknown"

            face_names.append(name)

        # Draw on frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Track FPS
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - last_time)
            last_time = time.time()
            print(f"🎯 FPS: {fps:.2f}")

        # Encode and send frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template_string('''
        <html>
        <head>
            <title>ESP32-CAM Face Recognition</title>
        </head>
        <body>
            <h1>ESP32-CAM Face Recognition</h1>
            <img src="{{ url_for('video_feed') }}" width="640" height="480">
        </body>
        </html>
    ''')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

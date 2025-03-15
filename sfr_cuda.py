import cv2
import os
import numpy as np
import face_recognition
import glob
import time
import dlib
from flask import Flask, Response, render_template_string, send_file, jsonify

app = Flask(__name__)

ESP32_STREAM_URL = "http://<ESP_IP>:81/stream"

# GLOBAL CACHES
known_face_encodings = []
known_face_names = []
reference_images = {}  # Stores reference images for display
FRAME_SKIP = 5

# ✅ Check if Dlib is using CUDA (ignore PyTorch)
USE_CUDA = dlib.DLIB_USE_CUDA

if USE_CUDA:
    print("🚀 Using CUDA for face detection in Dlib!")
else:
    print("⚠️ CUDA not available, running on CPU!")

# ✅ Use dlib's CUDA-accelerated face detector
face_detector = dlib.get_frontal_face_detector()

# --- Load Faces ---
enrollment_files = glob.glob("./data/*.jpg")

if not enrollment_files:
    print("\u274c No enrollment images found.")
else:
    for file in enrollment_files:
        try:
            raw_name = os.path.basename(file).rsplit(".", 1)[0]  # Remove extension
            name = " ".join(raw_name.split("_")[:-1])  # Extract base name
            
            image = face_recognition.load_image_file(file)
            encodings = face_recognition.face_encodings(image, num_jitters=1, model="cnn" if USE_CUDA else "hog")  # Use CNN if CUDA is available
            
            if encodings:
                known_face_encodings.append(encodings[0])  # Store first encoding
                known_face_names.append(name)
                reference_images[name] = file  # Store reference image path
                print(f"\u2705 Enrolled {name} from {file}")

        except Exception as e:
            print(f"\u274c Error loading {file}: {e}")

print(f"\U0001F680 Enrolled {len(known_face_encodings)} faces for {len(set(known_face_names))} people.")

# FPS Tracker
frame_count = 0
last_time = time.time()
detected_face = {"name": "Unknown", "ref_img": ""}  # Stores last detected face info

def gen_frames():
    global frame_count, last_time, detected_face

    cap = cv2.VideoCapture(ESP32_STREAM_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("\u274c Error: Cannot open ESP32-CAM stream")
        return

    while True:
        frame_count += 1
        success, frame = cap.read()
        if not success:
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = small_frame[:, :, ::-1]

        # ✅ Convert image properly
        dlib_image = dlib.numpy_image(rgb_small_frame)

        # ✅ Use CUDA-accelerated face detection
        detections = face_detector(dlib_image)

        if not detections:
            continue

        # Convert dlib detections to (top, right, bottom, left)
        face_locations = [(d.top(), d.right(), d.bottom(), d.left()) for d in detections]

        # Batch process face encodings
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=1, model="cnn" if USE_CUDA else "hog")

        tolerance = 0.5
        face_names = []

        for face_encoding in face_encodings:
            if not known_face_encodings:
                face_names.append("Unknown")
                continue

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < tolerance:
                name = known_face_names[best_match_index]
                detected_face["name"] = name
                detected_face["ref_img"] = reference_images.get(name, "")
                print("Recognized: " + name)
            else:
                name = "Unknown"

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        if frame_count % 10 == 0:
            fps = 10 / (time.time() - last_time)
            last_time = time.time()
            print(f"\U0001F3AF FPS: {fps:.2f}")

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
            <script>
                function updateFaceData() {
                    fetch("/detected_face").then(response => response.json()).then(data => {
                        document.getElementById("detected_name").innerText = data.name;
                        let img = document.getElementById("detected_image");
                        if (data.ref_img) {
                            img.src = "/reference_image?_=" + new Date().getTime();
                            img.style.display = "block";
                        } else {
                            img.style.display = "none";
                        }
                    });
                }
                setInterval(updateFaceData, 1000);
            </script>
        </head>
        <body>
            <h1>ESP32-CAM Face Recognition</h1>
            <img src="{{ url_for('video_feed') }}" width="640" height="480">
            <h2>Last Detected Face: <span id="detected_name">Unknown</span></h2>
            <h2>Matching Reference: <span id="detected_name"></span></h2>
            <img id="detected_image" src="" width="200" style="display:none;">
        </body>
        </html>
    ''')

@app.route('/detected_face')
def detected_face_data():
    return jsonify(detected_face)

@app.route('/reference_image')
def reference_image():
    if detected_face["ref_img"]:
        return send_file(detected_face["ref_img"], mimetype='image/jpeg')
    return "", 404

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

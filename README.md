# SFR-ESP32S

SFR-ESP32S is an importable python module which allow individuals to run server-side face recognition with ESP-32S chips (with OV260 Cameras) through web-streaming. 

### Get Started

**There will be pre-requisites required to run SFR:**
- [x] Glob
- [x] cv2
- [x] python-dlib
- [x] NumPy
- [x] face_recognition
- [x] flask

Download the repository and place the `sfr.py` file in the folder where you will be using the module and import it using `import sfr as sfr` or simply `import sfr`. If you would like to use SFR directly, simply use the file as everything is already set up.

Access `sfr.py` and edit and ensure you have the following:
- Ensure that the `ESP32_STREAM_URL` follows the following format: `http://<ESP_IP>:81/stream`
- Ensure that you are running at maximum, CIF quality or QVGA if you are having processing issues
- ESP-32S2 Chips may run at a higher quality with more stable processing and frames.
- Ensure you have a folder named `./data` with the reference files labelled as following: NAME_{n}.jpg
- SFR supports names with spaces by applying this format: FIRSTNAME_LASTNAME_{n}.jpg
- SFR supports multiple persons detection per frame, both with known and unknown face encodings
- Group encodings can also be added as a reference frame may have one or more persons within them, although they will only recognize when these individuals are flagged in a frame together. 

### How to use SFR?
Once all of the steps above have been followed and reference images have been formatted properly in `./data` you may encounter some issues.

- `No face data in reference {REFERENCE_FILE}` : This issue may be due to bad angles, lighting or a lack of face data
- `Failed encoding of image` : This either indicates your streamed frame is corrupted or the server has failed to capture streamed images
- `TIMEOUT ERROR` : This indicates the server cannnot access the stream, please ensure that you are accessing the server URL in incognito as browsers have an issue functioning with SFR due to cached images as these frames are streamed in FFMPEG.

If you do not encounter any of these issues proceed with the following.

1. Run `sfr.py`, ensure you have Flask installed
2. In the terminal it will provide a URL to access the stream
3. Open the URL in an incognito tab and put your face in frame

If you would like SFR to be more or less tolerant, adjust the `tolerance` value in `sfr.py`, with higher values being more lenient and lower values being stricter.






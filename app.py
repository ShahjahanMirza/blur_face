from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

blur_enabled = False

def process_img(img, face_detection, blur_enabled):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None and blur_enabled:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # Ensure bounding box stays within image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            w = min(W - x1, w)
            h = min(H - y1, h)

            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))

    return img

def generate_frames():
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    cap = cv2.VideoCapture(0)

    global blur_enabled  # Declare blur_enabled as global

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if blur_enabled:  # Apply blur only when blur is enabled
            frame = process_img(frame, face_detection, blur_enabled)

        ret, jpeg = cv2.imencode('.jpg', frame)

        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/toggle_blur')
def toggle_blur():
    global blur_enabled  # Explicitly declare blur_enabled as global
    blur_enabled = not blur_enabled
    return 'Blur toggled'

@app.route('/video_feed.jpg')
def video_feed_jpg():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

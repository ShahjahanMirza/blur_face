import cv2
import mediapipe as mp
print('libraries installed')

def blur():
    print('blur working')
    mp_face_detection = mp.solutions.face_detection
    cap = cv2.VideoCapture(0)  # Move this line outside the 'with' statement
    with mp_face_detection.FaceDetection(min_detection_confidence=0.3) as face_detection:
        ret, frame = cap.read()
        while ret:
            frame = process_img(frame, face_detection)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = cap.read()

    cap.release()

def process_img(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
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

if __name__ == '__main__':
    blur()

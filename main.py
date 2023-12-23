import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from blur import blur, process_img as blur_process_img
import mediapipe as mp

class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Blur App")

        self.label = ttk.Label(self.root)
        self.label.pack(pady=10)

        self.blur_button = ttk.Button(self.root, text="Blur", command=self.toggle_blur)
        self.blur_button.pack()

        self.cap = cv2.VideoCapture(0)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.3)

        self.blur_enabled = False

        self.update()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            frame = self.process_img(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(img)
            self.label.img = img
            self.label.config(image=img)
            self.root.after(10, self.update)

    def toggle_blur(self):
        self.blur_enabled = not self.blur_enabled

    def process_img(self, img):
        if self.blur_enabled:
            return blur_process_img(img, self.face_detection)
        else:
            return img

if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)
    root.mainloop()

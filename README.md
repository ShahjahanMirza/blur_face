# blur_face


https://github.com/ShahjahanMirza/blur_face/assets/103453568/6683ffc1-667d-4ba7-9193-83be14c76118


**Face Blur Webcam App**
This is a simple webcam application that can blur faces in real-time using OpenCV and mediapipe.

<h3>Overview</h3>
The main.py file creates a Tkinter GUI with a webcam feed and a "Blur" button. When blur is enabled, it calls the blur_process_img function from blur.py to detect faces using mediapipe and blur them before displaying the webcam frame.

The blur.py module contains the image processing logic:

* blur() - Applies a Gaussian blur to a specified region in the image
* process_img() - Detects faces using a mediapipe FaceDetection model, calls blur() to blur each face, and returns the processed image<br>
<h3>Requirements</h3>
* OpenCV
* Mediapipe
* Tkinter
* PIL
<h3>Usage</h3>
Run main.py to start the application. The webcam feed will be displayed. Click the "Blur" button to toggle blurring of detected faces on/off.

<h3>Customization</h3>
The min_detection_confidence parameter when creating the FaceDetection model can be adjusted to modify the face detection sensitivity.

The blur strength in blur() can be controlled by changing the kernel size.

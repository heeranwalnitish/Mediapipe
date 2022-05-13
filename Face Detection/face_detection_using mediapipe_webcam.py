import cv2
from matplotlib import image
import mediapipe as mp
from sklearn import model_selection


mp_face_detection = mp.solutions.face_detection
mp_drawings = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence =0.5) as face_detection :
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
        image  = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve perfomance, optionally mark the mage as a writable to pass by reference.
        image.flags.writeable = False
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                mp_drawings.draw_detection(image, detection)

            cv2.imshow('MediaPipe Face Detection', image) 
            if cv2.waitKey(5) & 0xFF == 27:
                break

cap.release()
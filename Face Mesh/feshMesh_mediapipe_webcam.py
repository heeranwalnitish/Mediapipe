import cv2
from cv2 import circle 
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness = 1, circle_radius =1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5,max_num_faces = 2, min_tracking_confidence = 0.5) as face_mesh :
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame. ")
            # If  loading a video, use 'break' instead of 'convert' the BGR image to RGB.
            continue
            
        # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_RGB2BGR)
        # To improve performance, optionally mark the image as not writable to pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)

        # Draw the face mesh annotations on image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image = image,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = drawing_spec,
                    connection_drawing_spec = drawing_spec)

        cv2.imshow("MediaPipe FaceMesh", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
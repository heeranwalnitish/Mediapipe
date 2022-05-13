import cv2
from cv2 import circle
from matplotlib.pyplot import connect
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# For Static image
IMAGE_FILES = [r"D:\Important\backup\Documents\id\Nitish\photo.jpg"]
drawing_spec = mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1)
with mp_face_mesh.FaceMesh(
    static_image_mode = True,
    max_num_faces = 1,
    min_detection_confidence = 0.5 ) as face_mesh :

    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
    # Convert the BGR image to RGB before processing. 
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


    # Print and draw face_mesh landmarks on the image.
        if not results.multi_face_landmarks:
            continue
    annoted_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
        print('face_landmarks:', face_landmarks)
        mp_drawing.draw_landmarks( image = annoted_image, landmark_list = face_landmarks, connections = mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec = drawing_spec, connection_drawing_spec = drawing_spec)
        cv2.imwrite('../annot_image'+ str(idx) + '.png', annoted_image)





import cv2
from matplotlib.pyplot import annotate
import mediapipe as mp

mp_drawings = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# For static image
IMAGE_FILES = [r"C:\Users\Admin\Computer vision\msdhoni-bye-27-1488175390.jpg"]
with mp_pose.Pose(static_image_mode =True, model_complexity = 2, min_detection_confidence = 0.5) as pose :
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


        if not results.pose_landmarks:
            continue
        print(
            f"Nose Coordinates: ("
            f"{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width},"
            f"{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})"
        )
        # Draw pose Landmarks on the image.
        annotated_imiage = image.copy() 
        mp_drawings.draw_landmarks(annotated_imiage, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imwrite("../annnoted_image" + str(idx) + '.png', annotated_imiage)

        #Plot Pose world landmarks.
        mp_drawings.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)



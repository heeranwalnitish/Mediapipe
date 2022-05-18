import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = [r"C:\Users\Admin\Computer vision\msdhoni-bye-27-1488175390.jpg"]

with mp_hands.Hands(static_image_mode = True,
                max_num_hands = 2,
                min_detection_confidence = 0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
        #Read an image, flip it around y-axis for correct handedness output
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hnad landmark on the image.
        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmark in results.multi_hand_landmarks:
            print("hand_landmark:", hand_landmark)
            print(
                f'Index finger tip coordinates:(',
                f'{hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width},'
                f'{hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmark, mp_hands.HAND_CONNECTIONS)

        cv2.imwrite("../aanote_image" + str(idx) + '.png', cv2.flip(annotated_image, 1))

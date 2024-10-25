import cv2
import mediapipe as mp
import random
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def count_fingers(hand_landmarks):
    fingers = []
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for tip_id in [8, 12, 16, 20]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def draw_falling_texts(image, falling_texts):
    for text in falling_texts:
        text['y'] += text['speed']
        cv2.putText(image, 'who there ?', (text['x'], text['y']), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

def add_falling_texts(falling_texts, width):
    if len(falling_texts) < 10:
        for _ in range(5):
            falling_texts.append({'x': random.randint(0, width), 'y': 0, 'speed': random.randint(2, 5)})

def draw_heart(image, center_x, center_y):

    points = np.array([
        [center_x, center_y + 20],
        [center_x - 40, center_y - 20],
        [center_x - 20, center_y - 60],
        [center_x, center_y - 40],
        [center_x + 20, center_y - 60],
        [center_x + 40, center_y - 20]
    ], np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.fillPoly(image, [points], (0, 0, 255))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('Hand Detection', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Hand Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

falling_texts = []
heart_visible = False

with mp_hands.Hands(max_num_hands=8, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        four_fingers_detected = False
        five_fingers_detected = False

        if results.multi_hand_landmarks:
            text_y = 30
            for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = count_fingers(hand_landmarks)
                num_fingers = fingers.count(1)
                handedness = hand_info.classification[0].label
                cv2.putText(image, f'{handedness}: {num_fingers} fingers', (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                text_y += 40

                if num_fingers == 4:
                    four_fingers_detected = True
                elif num_fingers == 5:
                    five_fingers_detected = True

        if four_fingers_detected and not five_fingers_detected:
            add_falling_texts(falling_texts, image.shape[1])
            cv2.putText(image, 'Imam here', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 5, cv2.LINE_AA)
            heart_visible = True
        elif five_fingers_detected:
            heart_visible = False

        draw_falling_texts(image, falling_texts)

        if heart_visible:
            draw_heart(image, image.shape[1] // 2, image.shape[0] // 2)

        cv2.imshow('Hand Detection', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()












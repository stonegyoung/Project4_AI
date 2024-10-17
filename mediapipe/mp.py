import cv2
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
import numpy as np

hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)  # 웹캠 입력

width = 1280
height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

while cap.isOpened():
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    if not ret:
        break
    
    # BGR 이미지를 RGB로 변환 후 손 추적
    result = hands.process( cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    
    img = np.full((height,width,3), fill_value=255, dtype=np.uint8)
    
    
    if result.multi_hand_landmarks:
        for hand_landmarks, hand_lr in zip(result.multi_hand_landmarks, result.multi_handedness) :
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            point_9 = hand_landmarks.landmark[9]
            point_x, point_y = int(hand_landmarks.landmark[9].x * width), (hand_landmarks.landmark[9].y * height)
            
            # 양쪽 손 다 인식
            if hand_lr.classification[0].label == 'Left':
                print({"lr": 'Left', "x": point_x, "y": point_y})
            else:
                print({"lr": 'Right', "x": point_x, "y": point_y})
            
        # print({"lr": hand_lr.classification[0].label, "x": point_x, "y": point_y}) # 먼저 나온 손만 인식
        
    cv2.imshow('', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

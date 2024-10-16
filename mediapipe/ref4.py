import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
import numpy as np

def check():
    w = np.random.randint(100, 1000, size=10)
    h = np.random.randint(100, 600, size = 10)
    return  set(map(tuple, np.column_stack((w, h))))
redlist = check()
bluelist = check()

toolstate = (0,0,255)
# print(whlist)

hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)  # 웹캠 입력

width = 1280
height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
redcnt, bluecnt = 0, 0
save_idx = 0

while cap.isOpened():
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    if not ret:
        break
    
    # BGR 이미지를 RGB로 변환 후 손 추적
    result = hands.process( cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    
    img = np.full((height,width,3), fill_value=255, dtype=np.uint8)
    
    
    for w,h in redlist:
        cv2.circle(img, (w,h), 30, (0,0,255), -1)
    for w,h in bluelist:
        cv2.circle(img, (w,h), 30, (255,0,0), -1)
    
    if result.multi_hand_landmarks:
        for hand_landmarks, hand_lr in zip(result.multi_hand_landmarks, result.multi_handedness) :
            mp_drawing.draw_landmarks(
                img, hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(toolstate, -1, 7)
            )
            point_9 = hand_landmarks.landmark[9]
            point_x, point_y = int(hand_landmarks.landmark[9].x * width), (hand_landmarks.landmark[9].y * height)
            
            if toolstate == (0,0,255):
                for xy in redlist:
                    if (xy[0]-50 <= point_x <= xy[0]+50) and (xy[1]-50 <= point_y <= xy[1]+50):
                        redlist.remove((xy[0], xy[1]))
                        redcnt += 1
                        break
            elif toolstate == (255,0,0):
                for xy in bluelist:
                    if (xy[0]-50 <= point_x <= xy[0]+50) and (xy[1]-50 <= point_y <= xy[1]+50):
                        bluelist.remove((xy[0], xy[1]))
                        bluecnt += 1
                        break
            
    cv2.putText(img, str(redcnt), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 5)  
    cv2.putText(img, str(bluecnt), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 5)  
    cv2.putText(img, 'red' if toolstate == (0,0,255) else 'blue', (1000, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 5)   
    cv2.imshow('', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    if cv2.waitKey(1) == ord('1'):
        toolstate = (0,0,255)
    elif cv2.waitKey(1) == ord('2'):
        toolstate = (255,0,0)

cap.release()
cv2.destroyAllWindows()

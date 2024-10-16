import cv2
import numpy as np
import mediapipe as mp

# MediaPipe 손 추적 모듈 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 캔버스 크기 지정
canvas_height = 480
canvas_width = 640

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 프레임 크기 맞추기
    frame = cv2.resize(frame, (canvas_width, canvas_height))
    
    # BGR을 RGB로 변환 (MediaPipe는 RGB 입력을 받음)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe로 손 추적 실행
    result = hands.process(rgb_frame)

    # 흰색 캔버스 생성
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

    # 손이 감지되면 랜드마크 그리기
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 손 랜드마크를 흰색 캔버스에 그리기
            mp_drawing.draw_landmarks(
                canvas, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
    
    # 웹캠과 흰색 캔버스 위에 그려진 MediaPipe 결과를 함께 보여주기
    combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # 결과 프레임 출력
    cv2.imshow('MediaPipe Hands on White Canvas', combined_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
hands.close()
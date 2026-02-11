import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Camera
cap = cv2.VideoCapture(0)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Screen size
screen_w, screen_h = pyautogui.size()

# Cursor tuning (IMPORTANT)
prev_x, prev_y = 0, 0
smoothening = 4          # ↓ lower = faster response
speed_multiplier = 1.5   # ↑ higher = more sensitivity

# Click control
click_delay = 0.5
last_click_time = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            landmarks = hand.landmark

            # Finger tips
            ix, iy = int(landmarks[8].x * w), int(landmarks[8].y * h)   # Index
            mx, my = int(landmarks[12].x * w), int(landmarks[12].y * h) # Middle
            tx, ty = int(landmarks[4].x * w), int(landmarks[4].y * h)   # Thumb

            # Visual markers
            cv2.circle(frame, (ix, iy), 10, (0, 255, 255), -1)
            cv2.circle(frame, (mx, my), 10, (255, 0, 255), -1)
            cv2.circle(frame, (tx, ty), 10, (0, 255, 0), -1)

            # Cursor position with increased sensitivity
            screen_x = (screen_w * ix / w) * speed_multiplier
            screen_y = (screen_h * iy / h) * speed_multiplier

            # Boundary control
            screen_x = max(0, min(screen_w, screen_x))
            screen_y = max(0, min(screen_h, screen_y))

            # Smooth movement
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Distance calculations
            dist_thumb_index = math.hypot(ix - tx, iy - ty)
            dist_thumb_middle = math.hypot(mx - tx, my - ty)

            current_time = time.time()

            # LEFT CLICK
            if dist_thumb_index < 30 and current_time - last_click_time > click_delay:
                pyautogui.click(button='left')
                last_click_time = current_time
                cv2.putText(frame, "LEFT CLICK", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # RIGHT CLICK
            elif dist_thumb_middle < 30 and current_time - last_click_time > click_delay:
                pyautogui.click(button='right')
                last_click_time = current_time
                cv2.putText(frame, "RIGHT CLICK", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("AI Virtual Mouse - High Sensitivity", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

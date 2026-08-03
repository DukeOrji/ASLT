
import mediapipe as mp
import cv2
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def get_landmarks(frame):
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    wrist = hand.landmark[0]
    index = hand.landmark[5]

    #increases rate of detection by nullyfying failure of far frames
    scale = math.sqrt(
        (index.x - wrist.x)**2 +
        (index.y - wrist.y)**2 +
        (index.z - wrist.z)**2 
    )
    scale = max(scale, 1e-6)

    #draw visual landmarks
    mp_draw.draw_landmarks(
        frame,
        hand,
        mp_hands.HAND_CONNECTIONS
    )

    landmarks = []

    for lm in hand.landmark:

        #centered on the wrist (normalized)
        landmarks.extend([
            (lm.x - wrist.x) / scale,
            (lm.y - wrist.y) / scale,
            (lm.z - wrist.z) / scale
        ])

    

    return landmarks
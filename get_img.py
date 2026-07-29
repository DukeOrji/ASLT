import cv2
import numpy
from mpipeline import get_landmarks



cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    success, frame = cap.read()

    if not success:
        break

    landmarks = get_landmarks(frame)


    cv2.imshow("FrameCap", frame)
    frame = cap.resize(frame, (224, 224))
    

    if cv2.waitkey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
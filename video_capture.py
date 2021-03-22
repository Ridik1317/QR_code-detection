import cv2
from    qr_detection import qr_detect

# Enable camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 420)

while True:
    success, img = cap.read()
    img_detect = qr_detect(img)

    cv2.imshow('face_detect', img_detect)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow('face_detect')

import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

i = 1
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    if c == 32:
        cv2.imwrite('frame' + str(i) + '.jpg', frame)
        i += 1
    elif c == 27:
        break

cap.release()
cv2.destroyAllWindows()

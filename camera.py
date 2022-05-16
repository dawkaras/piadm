import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

ret, frame = cap.read()
frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
cv2.imwrite("first_frame.jpg", frame)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    if c == 27:
        with open("first_frame.jpg", "rb") as image:
            f = image.read()
            b = bytearray(f)
            print(b[0])
        cv2.imwrite("last_frame.jpg", frame)
        break

cap.release()
cv2.destroyAllWindows()

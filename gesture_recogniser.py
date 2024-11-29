import cv2 as cv

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("We completely support your privacy requirements, but need to access your webcam for this app to work.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Exiting stream...")
        break
    cv.imshow('Webcam', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

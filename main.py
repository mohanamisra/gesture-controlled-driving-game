import cv2 as cv
from real_time_model_testing import predict_action

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("We completely support your privacy requirements, but need to access your webcam for this app to work.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    if not ret:
        print("Exiting stream...")
        break

    action = predict_action(frame)

    if action:
        print(f"Predicted action: {action}")

    # Display the frame (optional)
    cv.imshow("Webcam", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

import cv2

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the default webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set the width of the video frame
cap.set(4, 480)  # Set the height of the video frame

while True:
    # Capture a single frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break  # Exit if the frame is not captured

    # Convert the captured frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the classifier
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # Display the result in a window
    cv2.imshow('Face Detection', frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

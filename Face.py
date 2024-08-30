import cv2

# Load the face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier(r"D:\Face_Detection\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"D:\Face_Detection\haarcascade_eye.xml")

# Start video capture from the webcam
video_capture = cv2.VideoCapture(0)

# Create a named window and set its size to 400x400
cv2.namedWindow('Face and Eye Detection', cv2.WINDOW_NORMAL)

while True:
    # Capture a frame from the video
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 6)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

        # Region of interest for the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around each detected eye
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)

    # Display the frame with the detected faces and eyes
    cv2.imshow('Face and Eye Detection', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()

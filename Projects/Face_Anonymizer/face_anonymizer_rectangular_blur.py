import cv2
import mediapipe as mp

# Creating an object or instance for MediaPipe face detection class
mp_face_detection = mp.solutions.face_detection

# Creating an object of webcam accessing
source = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    # Loop runs until the 'q' key is pressed
    while cv2.waitKey(1) != ord('q'):
        has_frame, frame = source.read()
        if not has_frame:
            break

        ih, iw, _ = frame.shape

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converting the frame from BGR to RGB
        output = face_detection.process(image_rgb)  # Processing the Image and Identifying the co-ordinates or landmarks of the face

        if output.detections:
            for detection in output.detections:
                bounding_box = detection.location_data.relative_bounding_box
                xmin, ymin, w, h = bounding_box.xmin, bounding_box.ymin, bounding_box.width, bounding_box.height
                x = int(xmin * iw)
                y = int(ymin * ih)
                w = int(w * iw)
                h = int(h * ih)

                # Blurring the face
                frame[y : y+h ,x : x+w] = cv2.blur(frame[y : y+h, x : x+w], (20,20))

                cv2.imshow('Face Anonymizer',frame)

source.release()
cv2.destroyAllWindows()

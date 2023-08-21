import cv2
import mediapipe as mp
import numpy as np

# Create a FaceMesh instance adn Video Capture instance
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

source = cv2.VideoCapture(0)

# Loop until the 'q' key is pressed
while cv2.waitKey(1) != ord('q'):
    has_frame, frame = source.read()
    if not has_frame:
        break

    # Convert BGR frame to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    yi, xi, _ = image_rgb.shape

    image_copy = image_rgb.copy()

    # Process the frame with FaceMesh
    outputs = face_mesh.process(image_rgb)
    final_img = image_rgb

    landmarks_list = list()
    if outputs.multi_face_landmarks:
        for landmarks in outputs.multi_face_landmarks:
            for landmark in landmarks.landmark:
                x1, y1 = landmark.x, landmark.y
                x = int(x1*xi)
                y = int(y1*yi)
                landmarks_list.append([x, y])   # Store face landmarks
                # cv2.circle(image_rgb, (x,y), 3, ((0,0,255))) #It is used to mark the regions on the face

        # Extracting the outer region of the points without considering the input regions
        face_region_points = cv2.convexHull(np.array(landmarks_list),returnPoints=True)

        # You can use this to plot the polygon lines on the image, it represents the outer region
        # cv2.polylines(image_bgr,[face_region_points], isClosed=True, color=(0,0,255), thickness=3, lineType=cv2.LINE_AA)

        # Create a mask of the face region
        mask = np.zeros((yi,xi), np.uint8)
        cv2.fillConvexPoly(mask, face_region_points,(255,255,255))

        # Blur the image copy and Extract the foreground of the face
        image_copy = cv2.blur(image_copy, (30,30))
        mask_foreground = cv2.bitwise_and(image_copy, image_copy, mask=mask)

        # Extract background
        cv2.fillConvexPoly(image_rgb, face_region_points,0)

        # Combine foreground and background
        final_img = cv2.add(image_rgb, mask_foreground)

    # Display the result
    cv2.imshow('Face Anonymizer', cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))

# Release video source and close windows
source.release()
cv2.destroyAllWindows()


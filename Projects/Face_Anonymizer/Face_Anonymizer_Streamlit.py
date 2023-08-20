import cv2
from streamlit_webrtc import webrtc_streamer
import av
import streamlit as st
import mediapipe as mp

html_code = '''
<style>
.heading_one{
text-align : center;
}
</style>
'''

st.markdown(html_code, unsafe_allow_html=True)
st.markdown("<h1 class='heading_one'> Face Anonymizer</h1>", unsafe_allow_html=True)

blur_value  = st.slider("Blur Value", 1, 100, 10)

def video_callback(frame):
    img = frame.to_ndarray(format='bgr24')
    mp_face_detection = mp.solutions.face_detection

    ih, iw, _ = img.shape

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:


            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converting the frame from BGR to RGB
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
                    img[y: y + h, x: x + w] = cv2.blur(img[y: y + h, x: x + w], (blur_value, blur_value))
                    return av.VideoFrame.from_ndarray(img, format='bgr24')

webrtc_streamer(key='example2', video_frame_callback=video_callback, media_stream_constraints={'video':True, 'audio':False})

from hsv_color_limits import get_color_limits
import cv2
from PIL import Image

black = [255,0,0]
# lower_limit = (110,50,50)      # Lower limit for HSV (Hue, Saturation, Value)
# upper_limit = (130,255,255) # Upper limit for HSV (Hue, Saturation, Value)

source = cv2.VideoCapture(0)

while cv2.waitKey(1)!=ord('q'):
    has_frame, frame = source.read()
    if not has_frame:
        break
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_limit, upper_limit = get_color_limits(black)
    mask = cv2.inRange(hsv_image, lower_limit, upper_limit)

    mask = Image.fromarray(mask)
    bounding_box = mask.getbbox()
    if bounding_box is not None:
        x1, y1, x2, y2 = bounding_box
        cv2.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,0), thickness=2)

    cv2.imshow('Color Object Detector', frame)

source.release()
cv2.destroyAllWindows()
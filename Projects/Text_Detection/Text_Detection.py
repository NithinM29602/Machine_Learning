import easyocr
import cv2
import matplotlib.pyplot as plt

image_path = 'Images/work_in_progress.jpg'
image = cv2.imread(image_path,1)

# Kannada Language is Selected, we can select other languages also.
# https://www.jaided.ai/easyocr/
# For more information go to the site mentioned above, also you can look for other Language code which can be used int this model.
reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext(image)

threshold = 0.4
for result in results:
    bbox, text, score = result
    # Threshold is set to reduce the noise.
    if score > threshold:
        (x1,y1), (x2,y2) = bbox[0], bbox[2] # Extracting the top-left and bottom-right corner co-ordinates
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(image,text,(x1,y1), cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255), 5)


plt.imshow(image[:,:,::-1])
plt.show()
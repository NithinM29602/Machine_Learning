import torch
import cv2
from torchvision import models
from yolo_v1_model import Yolov1
from nms import nms

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Yolov1(split_size=7, num_bboxes=2, num_classes=20)
weights_path = 'path_to_pretrained_weights.pth'
model.load_state_dict(torch.load(weights_path))
model.to(device)
model.eval()

image_path = 'indian_street_image.jpg'
img = cv2.imread(image_path)

img = cv2.resize(img, (448,448))

# Preprocess the image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = torch.from_numpy(img/255.0).permute(2,0,1).float().unsqueeze(0)

with torch.no_grad():
    img = img.to(device)
    output = model(img)

    threshold = 0.5
    iou_threshold = 0.8

    final_bboxes = nms([output[..., 21:25], output[..., 26:30]],
                       threshold, iou_threshold, box_format='midpoint')

    print(final_bboxes)



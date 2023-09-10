import torch
import torchvision
import numpy
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from PIL import Image

from yolo_v1_model import Yolov1
from dataset import VOCDataset
from yolo_loss import YoloLoss
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)

class Compose(object):
  def __init__(self, transforms):
    self.transform = transforms

  def __call__(self, img, bboxes):
    for t in transform:
      img, bboxes = t(img), bboxes


transform = Compose([
    torchvision.transforms.Resize((448,448)),
    torchvision.transforms.ToTensor()
    ])

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

learning_rate = 1e-2
weight_decay = 1e-3
num_epoch = 100
num_workers = 2
batch_size = 16
img_dir = ''
label_dir = ''

train_data = VOCDataset(
    '/content/sample_data/data/20examples.csv',
    transform = transform,
    img_dir = img_dir,
    label_dir = label_dir
)

test_data = VOCDataset(
    '/content/sample_data/test.csv',
    transform = transform,
    img_dir = img_dir,
    label_dir = label_dir
)

train_loader = torch.utils.data.DataLoader(
    dataset = train_data,
    batch_size=batch_size,
    num_workers = num_workers,
    shuffle = True
    )

test_loader = torch.utils.data.DataLoader(
    dataset = train_data,
    batch_size=batch_size,
    num_workers = num_workers,
    shuffle = True
    )

model = Yolov1(split_size=7, num_bboxes=2, num_classes=20)
model.to(device)
loss_func = YoloLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)

for epoch in range(num_epoch):
  pred_boxes, target_boxes = get_bboxes(
        train_loader, model, iou_threshold=0.5, threshold=0.4
        )

  mean_avg_prec = mean_average_precision(
      pred_boxes, target_boxes, iou_threshold=0.5, threshold=0.4
  )

  for i, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)
    predicted = model(images)
    loss = loss_func(predicted, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if (i+1) % 1000 == 0:
      print(f"[{epoch}, {i+1}], loss : {loss}")

correct = toatl = 0
with torch.no_grad():
  for data in test_loader:
    images, labels = data
    images = images.to(device)
    labels = labels.to(device)
    output = model(images)
  
model.eval()

pred_ls = []
target_ls = []

with torch.no_grad():
  for data in test_loader:
    images, labels = data
    images = images.to(device)
    labels = labels.to(device)
    output = model(images)

    pred_bboxes, target_bboxes = get_bboxes(
        test_loader, model, iou_threshold=0.5, threshold=0.4
        )
    
    pred_ls.append(pred_bboxes)
    target_ls.append(target_bboxes)

  mean_avr_prec_test = mean_average_precision(
      pred_ls, target_ls, iou_threshold=0.5, threshold=0.4
  )

  print(F"Mean Average Precision (mAP) : {mean_avr_prec_test}")

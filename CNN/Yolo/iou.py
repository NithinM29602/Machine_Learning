import torch

def intersection_over_union(bbox_preds, bbox_labels, box_format='corners'):

    if box_format == 'midpoint':
        bbox1_x1 = bbox_preds[...,0:1] - bbox_preds[2:3]/2
        bbox1_y1 = bbox_preds[...,1:2] - bbox_preds[3:4]/2
        bbox1_x2 = bbox_preds[..., 0:1] + bbox_preds[2:3] / 2
        bbox1_y2 = bbox_preds[..., 1:2] + bbox_preds[3:4] / 2
        bbox2_x1 = bbox_labels[..., 0:1] - bbox_labels[2:3] / 2
        bbox2_y1 = bbox_labels[..., 1:2] - bbox_labels[3:4] / 2
        bbox2_x2 = bbox_labels[..., 0:1] + bbox_labels[2:3] / 2
        bbox2_y2 = bbox_labels[..., 1:2] + bbox_labels[3:4] / 2

    if box_format == 'corners':
        bbox1_x1 = bbox_preds[..., 0:1]
        bbox1_y1 = bbox_preds[..., 1:2]
        bbox1_x2 = bbox_preds[..., 2:3]
        bbox1_y2 = bbox_preds[..., 3:4]

        bbox2_x1 = bbox_labels[..., 0:1]
        bbox2_y1 = bbox_labels[..., 1:2]
        bbox2_x2 = bbox_labels[..., 2:3]
        bbox2_y2 = bbox_labels[..., 3:4]

    x1 = torch.max(bbox1_x1, bbox2_x1)
    y1 = torch.max(bbox1_y1, bbox2_y1)
    x2 = torch.min(bbox1_x2, bbox2_x2)
    y2 = torch.min(bbox1_y2, bbox2_y2)

    intersection_value = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    bbox1_area = abs((bbox1_x2 - bbox1_x1) * (bbox1_y2 - bbox1_y1))
    bbox2_area = abs((bbox2_x2 - bbox2_x1) * (bbox2_y2 - bbox2_y1))

    iou = intersection_value / (bbox1_area + bbox2_area - intersection_value)

    return iou
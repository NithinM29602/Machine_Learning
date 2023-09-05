import torch
from collections import Counter
from iou import intersection_over_union

def mean_average_precision(pred_bboxes, truth_bboxes, iou_threshold = 0.5, box_format='corners',num_classes = 20):

    # [[train_idx, class_idx, prob_score, x1, y1, x2, y2]]
    detections = []
    ground_truth_bboxes = []
    average_precisions = []

    for c in range(num_classes):
        for detection in pred_bboxes:
            if detection[1] == c:
                detections.append(detection)

        for truth_box in truth_bboxes:
            if truth_box[1] == c:
                ground_truth_bboxes.append(truth_box)

        #Counter to keep track of it
        amount_truth = Counter(g[0] for g in ground_truth_bboxes)
        for key, value in amount_truth.items():
            amount_truth[key] = torch.zeros(value)

        detections.sort(key=lambda x: x[2], reverse=True)

        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_ground_truth = len(ground_truth_bboxes)

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                box for box in ground_truth_bboxes if box[0]  == detection[0]
            ]

            ground_truth_idx = 0
            best_iou = 0

            for idx, gtx in enumerate(ground_truth_img):
                iou_value = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gtx[3:]),
                    box_format=box_format
                )

                if iou_value > best_iou:
                    best_iou = iou_value
                    ground_truth_idx = idx

            if best_iou > iou_threshold:
                if amount_truth[detection[0]][ground_truth_idx] == 0:
                    TP[detection_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        # [1,0,1,1] --> [1,1,2,3] start to particular value
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / total_ground_truth
        precision = TP_cumsum / (TP_cumsum + FP_cumsum)

        recalls = torch.cat((torch.tensor([0]), recalls))
        precision = torch.cat((torch.tensor([1]), precision))

        average_precisions.append(torch.trapz(precision, recalls))

        return sum(average_precisions) / len(average_precisions)



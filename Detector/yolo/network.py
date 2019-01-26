import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.network import MobileNet
from utils.helper import bbox_iou, yolo_correct_boxes, non_maximum_suppression


def YOLODetector(feature_maps, anchors, n_classes, input_shape, compute_loss=False):
    """
    Convert YOLOv3 layer feature maps to bounding box parameters.

    Reference: (1) https://github.com/qqwweee/keras-yolo3/blob/master/yolo3/model.py
               (2) https://github.com/jiasenlu/YOLOv3.pytorch/blob/master/misc/yolo.py

    Parameters
    ----------
    feature_maps: Feature maps learned by the YOLOv3 layer, shape = [1, 3*(5+C), 13, 13]
    anchors: Numpy array of shape = (3, 2). 3 anchors for each scale, and an anchor
        specifies its [width, height]. There are total 9 anchors, 3 for each scale.
    n_classes: int, number of classes
    input_shape: Pytorch tensor, that specifies (height, width). NOTE: height and width 
        are multiples of 32
    compute_loss: bool, if True then return outputs to calculate loss, else return
        predictions

    Return
    ------
    If compute loss is true then:
        grid (cell offsets), size: [1, 13, 13, 1, 2], where [..., 2:] is x,y center of cells
        feature_maps: Feature maps (raw predictions) learned by the YOLOv3 layer, size: [1, 13, 13, 3, 5+C]
        box_xy: Center (x, y) of bounding box, size: [1, 13, 13, 3, 2]
        box_wh: width, height of bounding box, size: [1, 13, 13, 3, 2]
    else:
        box_xy: Center (x, y) of bounding box, size: [1, 13, 13, 3, 2]
        box_wh: width, height of bounding box, size: [1, 13, 13, 3, 2]
        box_confidence: Confidence score, size: [1, 13, 13, 3, 1]
        box_class_probs: Class probabilities, size: [1, 13, 13, 3, C]
    """
    # NOTE: Comments are based on feature_maps of size [N, 3*(5+C), 13, 13]
    if not compute_loss:
        feature_maps = feature_maps.cpu()
        input_shape = input_shape.cpu()

    # Number of anchors for each scale. It should be 3 anchors in each scale
    num_anchors = len(anchors)  # 3

    # Convert NumPy array to Torch tensor and reshape to include dimensions for (num_images, height,
    # width, scales, 5+C), size: [3, 2] -> [1, 1, 1, 3, 2]
    anchors_tensor = torch.from_numpy(anchors).view(1, 1, 1, num_anchors, 2).type_as(feature_maps)

    # Compute grid shape
    grid_shape = feature_maps.shape[2:4]  # height x width

    # Create a grid or cell offsets
    grid_y = torch.arange(0, grid_shape[0])  # size: [13]
    grid_x = torch.arange(0, grid_shape[1])  # size: [13]

    grid_y = grid_y.view(-1, 1, 1, 1)  # size: [13] -> [13, 1, 1, 1]
    grid_x = grid_y.view(1, -1, 1, 1)  # size: [13] -> [1, 13, 1, 1]

    grid_y = grid_y.expand(grid_shape[0], grid_shape[0], 1, 1)  # size: [13, 1, 1, 1] -> [13, 13, 1, 1]
    grid_x = grid_x.expand(grid_shape[1], grid_shape[1], 1, 1)  # size: [1, 13, 1, 1] -> [13, 13, 1, 1]

    # Grid (x, y), where (x, y) is center of cell. Check `grid[0:2, ...]` output
    #  (0,0) (1,0) ... (12,0)
    #  (0,1) (1,1) ... ...
    #  ...         ... ...
    #  (0,12) ...  ... (12,12)
    grid = torch.cat([grid_x, grid_y], dim=3)  # size: [13, 13, 1, 2]

    # Insert one dimension for batch size
    grid = grid.unsqueeze(0).type_as(feature_maps)  # size: [13, 13, 1, 2] -> [1, 13, 13, 1, 2]

    # Reshape feature maps size: [1, 3*(5+C), 13, 13] -> [1, 13, 13, 3, 5+C]
    feature_maps = feature_maps.view(-1, num_anchors, 5 + n_classes, grid_shape[0], grid_shape[1])  # size: [1, 3*(5+C), 13, 13] -> [1, 3, 5+C, 13, 13]
    feature_maps = feature_maps.permute(0, 3, 4, 1, 2).contiguous()  # size: # [1, 3, 5+C, 13, 13] -> [1, 13, 13, 3, 5+C]

    # Compute: bx = sigmoid(tx) + cx and by = sigmoid(ty) + cy, output size: [1, 13, 13, 3, 2]
    box_xy = torch.sigmoid(feature_maps[..., :2]) + grid  # feature_maps[...,:2] -> xy

    # Compute: bw = pw * exp(tw) and bh = ph * exp(th), output size: [1, 13, 13, 3, 2]
    box_wh = anchors_tensor * torch.exp(feature_maps[..., 2:4])  # feature_maps[...,2:4] -> wh

    # Adjust predictions to each spatial grid point and anchor size
    # box_xy some values are > 1 so [sigmoid(tx) + cx]/13 and [sigmoid(ty) + cy]/13
    # makes box_xy values to be in range [0, 1]
    box_xy = box_xy / torch.tensor(grid_shape).view(1, 1, 1, 1, 2).type_as(feature_maps)

    # box_wh values needs to be scaled by input_shape
    box_wh = box_wh / input_shape.view(1, 1, 1, 1, 2)

    # Box confidence score, output size: [1, 13, 13, 3, 1]
    box_confidence = torch.sigmoid(feature_maps[..., 4:5]) # feature_maps[..., 4:5] -> confidence scores

    # Box class probabilities, output size: [1, 13, 13, 3, C]
    box_class_probs = torch.sigmoid(feature_maps[..., 5:]) # feature_maps[..., 5:] -> class scores

    if compute_loss:
        return grid, feature_maps, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


class YOLOLoss(nn.Module):
    """
    Reference: (1) https://github.com/qqwweee/keras-yolo3/blob/master/yolo3/model.py
               (2) https://github.com/jiasenlu/YOLOv3.pytorch/blob/master/misc/yolo.py
    """
    def __init__(self, params):
        super(YOLOLoss, self).__init__()
        self.params = params
        self.anchors = np.array(params.anchors)
        self.num_scales = len(self.anchors) // 3
        self.anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.n_classes = len(params.class_names)
        self.ignore_thresh = 0.5

        # Losses: Mean Squared Error and Binary Cross Entropy
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, yolo_outputs, y_true):
        """
        Parameters
        ----------
        yolo_outputs: list of Pytorch Tensors (YOLO network output. Where tensors 
            shapes are [(N, 3 * (5 + C), 13, 13), (N, 3 * (5 + C), 26, 26), 
            (N, 3 * (5 + C), 52, 52)]
        y_true: list of Pytorch Tensors (preprocessed bounding boxes). Where array 
            shapes are [(N, 13, 13, 3, 5 + C), (N, 26, 26, 3, 5 + C)], 
            (N, 52, 52, 3, 5 + C)]

        Returns
        -------

        """
        # Input shape: [416., 416.]
        dim_x = yolo_outputs[0].shape[2] * 32
        dim_y = yolo_outputs[0].shape[3] * 32
        input_shape = torch.Tensor([dim_x, dim_y]).type_as(yolo_outputs[0])

        # Grid shape: [tensor([13., 13.]), tensor([26., 26.]), tensor([52., 52.])]
        grid_shapes = [torch.Tensor([out.shape[2], out.shape[3]]).type_as(yolo_outputs[0]) for out in yolo_outputs]

        # Convert y_true to PyTorch tensor
        y_true = [torch.tensor(yt) for yt in y_true]

        batch_size = yolo_outputs[0].size(0)

        # Initialize different losses
        loss_xy = 0  # Localization loss
        loss_wh = 0  # Localization loss
        loss_conf = 0  # Confidence loss (Confidence measures the objectness of the box)
        loss_clss = 0  # Classification loss

        # Iterating over all the scales
        for s in range(self.num_scales):
            object_mask = y_true[s][..., 4:5]  # cell value is 1 if grid cell an contains object
            true_class_probs = y_true[s][..., 5:]  # C

            # Use YOLO Detector to compute loss
            grid, raw_preds, pred_xy, pred_wh = YOLODetector(yolo_outputs[s],
                                                             self.anchors[self.anchor_mask[s]],
                                                             self.n_classes,
                                                             input_shape,
                                                             compute_loss=True)

            # Concatenate pred_xy and pred_wh
            pred_box = torch.cat([pred_xy, pred_wh], dim=4)  # size: [1, 13, 13, 3, 4]

            # Ground truth xy: Not sure what is happening here...need to look again
            raw_true_xy = y_true[s][..., :2] * grid_shapes[s].view(1, 1, 1, 1,
                                                                   2) - grid  # size: [1, 13, 13, 3, num_boxes]

            # Ground truth wh (might have problems with log(0)=-inf)
            raw_true_wh = torch.log((y_true[s][..., 2:4] / torch.Tensor(self.anchors[self.anchor_mask[s]]).
                                     type_as(pred_box).view(1, 1, 1, self.num_scales, 2)) *
                                    input_shape.view(1, 1, 1, 1, 2))

            # Fill the -inf values with 0
            raw_true_wh.masked_fill_(object_mask.expand_as(raw_true_wh) == 0, 0)

            # Box loss scale: 2 - w * h?, need to check again
            box_loss_scale = 2 - y_true[s][..., 2:3] * y_true[s][..., 3:4]

            # Iterate over each batch and compute IoU
            best_ious = []
            for batch in range(batch_size):
                true_box = y_true[s][batch, ..., 0:4][object_mask[batch, ..., 0] == 1]
                iou = bbox_iou(pred_box[batch], true_box)  # shape: [13, 13, 3, num_boxes]
                best_iou, _ = torch.max(iou, dim=3)  # shape: [13, 13, 3]
                best_ious.append(best_iou)

            # Find best ious
            best_ious = torch.stack(best_ious, dim=0)  # size: [1, 13, 13, 3, num_boxes]
            best_ious = best_ious.unsqueeze(4)  # size: [1, 13, 13, 3, 1]

            # Find ignore mask
            ignore_mask = (best_ious < self.ignore_thresh).float()

            # Compute losses. TODO: Check this again to understand better!
            # True and pred x,y values would be in range [0,1]. Binary Cross-entropy: If the input data are between zeros and ones
            # then BCE is acceptable as the loss function [Ref: https://www.youtube.com/watch?v=xTU79Zs4XKY&feature=youtu.be&t=330]
            # Check discussion here: https://stats.stackexchange.com/questions/223256/tensorflow-cross-entropy-for-regression
            # and here: https://stats.stackexchange.com/questions/245448/loss-function-for-autoencoders/296277#296277
            # Also, BCE is is helpful to avoid exponent overflow.
            xy_loss = torch.sum(
                object_mask * box_loss_scale * self.bce_loss(raw_preds[..., 0:2], raw_true_xy)) / batch_size

            # Pred w,h values can be greater than 1 so using MSE loss
            wh_loss = torch.sum(
                object_mask * box_loss_scale * self.mse_loss(raw_preds[..., 2:4], raw_true_wh)) / batch_size
            # print('wh_loss: ', wh_loss.item())

            # Confidence loss
            conf_loss = torch.sum(object_mask * self.bce_loss(raw_preds[..., 4:5], object_mask) +
                                  (1 - object_mask) * self.bce_loss(raw_preds[..., 4:5],
                                                                    object_mask) * ignore_mask) / batch_size

            # Class loss
            class_loss = torch.sum(object_mask * self.bce_loss(raw_preds[..., 5:], true_class_probs)) / batch_size

            # Update losses
            loss_xy += xy_loss
            loss_wh += wh_loss
            loss_conf += conf_loss
            loss_clss += class_loss
            # print('loss_xy: {}, loss_wh: {}, loss_conf: {}, loss_clss: {}'.format(loss_xy, loss_wh, loss_conf, loss_clss))

        # Total loss
        loss = loss_xy + loss_wh + loss_conf + loss_clss

        return loss.unsqueeze(0), loss_xy.unsqueeze(0), loss_wh.unsqueeze(0), loss_conf.unsqueeze(0), loss_clss.unsqueeze(0)


def yolo_boxes_and_scores(feature_maps, anchors, n_classes, input_shape, image_shape):
    """
    Process output from YOLODetector

    Parameters
    ----------
    feature_maps: Feature maps learned by the YOLOv3 layer, shape = [1, 3*(5+C), 13, 13]
    anchors: Numpy array of shape = (3, 2). 3 anchors for each scale, and an anchor
        specifies its [width, height]. There are total 9 anchors, 3 for each scale.
    n_classes: int, number of classes
    input_shape: Pytorch tensor, that specifies (height, width). NOTE: height and width 
        are multiples of 32
    image_shape: Pytorch tensor?

    Return
    ------
    """
    # Get output from YOLODetector
    box_xy, box_wh, box_confidence, box_class_probs = YOLODetector(feature_maps, anchors, n_classes, input_shape)

    # Correct the bounding boxes, size: [N, 13, 13, 3, 4] where 4 specifies y_min, x_min, y_max, x_max
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)

    # Resize boxes tensor, size: [N, 13, 13, 3, 4] -> [13 * 13 * num_scales, 4]
    boxes = boxes.view([-1, 4])

    # Box scores = Box confidence * Box class probabilities
    box_scores = box_confidence * box_class_probs  # size: [N, 13, 13, 3, 4]
    box_scores = box_scores.view(-1, n_classes)  # size: [13 * 13 * num_scales, n_classes]

    return boxes.view(feature_maps.size(0), -1, 4), box_scores.view(feature_maps.size(0), -1, n_classes)


def yolo_eval(yolo_outputs, anchors, n_classes, image_shape, score_threshold=0.6,
              nms_threshold=0.3, max_per_image=50):
    """
    Evaluate YOLO model on given input and return filtered boxes.

    Reference: (1) https://github.com/qqwweee/keras-yolo3/blob/master/yolo3/model.py
    (2) https://github.com/jiasenlu/YOLOv3.pytorch/blob/master/misc/yolo.py

    Parameters
    ----------
    yolo_outputs:
    anchors: Numpy array, 
    n_classes: int, number of classes
    image_shape: PyTorch tensor,
    score_threshold:
    nms_threshold:
    max_per_image:

    Returns
    -------
    A tuple of tensors: predicted detections, image indices, predicted classes 
    """
    num_scales = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = torch.Tensor([yolo_outputs[0].size(2) * 32,
                                yolo_outputs[0].size(3) * 32]).type_as(yolo_outputs[0])

    # Create lists to store boxes and scores
    boxes = []
    box_scores = []

    # For each scale process output from YOLODetector
    for s in range(num_scales):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[s], anchors[anchor_mask[s]],
                                                    n_classes, input_shape, image_shape)

        boxes.append(_boxes)  # size: [M, yo_h * yo_w * 3, 4]
        box_scores.append(_box_scores)  # size: [M, yo_h * yo_w * 3, C]

    # Concatenate each scales processed boxes and box scores
    boxes = torch.cat(boxes, dim=1)  # size: [M, 10647, 4]
    box_scores = torch.cat(box_scores, dim=1)  # size: [M, 10647, C]

    # Create lists to store processed detection outputs for a batch
    dets = []
    classes = []
    img_indices = []

    # for each image in a batch
    for i in range(boxes.size(0)):

        # Create mask for selecting boxes that have score greater than threshold
        mask = box_scores[i] > score_threshold

        # Create list to store processed detection outputs for an image in batch
        img_dets = []
        img_classes = []
        img_idx = []

        # For each class
        for c in range(n_classes):

            # Filter out boxes and scores for class c that have score greater than threshold
            class_boxes = boxes[i][mask[:, c]]
            if len(class_boxes) == 0:
                continue
            class_box_scores = box_scores[i][:, c][mask[:, c]]

            # Sort class box scores in descending order
            _, idx = torch.sort(class_box_scores, dim=0, descending=True)

            # Combine class boxes and class box scores for NMS
            class_dets = torch.cat((class_boxes, class_box_scores.view(-1, 1)), dim=1)  # [?, 4+1]

            # Order the class detections in descending order of class box scores
            class_dets = class_dets[idx]

            # Supress boxes using NMS
            keep = non_maximum_suppression(class_dets.data.numpy(), thresh=nms_threshold)

            # Convert list to PyTorch tensor
            keep = torch.from_numpy(np.array(keep))

            # Reshape keep and convert it to a long tensor
            keep = keep.view(-1).long()

            # Filter out class detections to keep
            class_dets = class_dets[keep]

            # For each class, image detections, image classes and image index are appended
            img_dets.append(class_dets)
            img_classes.append(torch.ones(class_dets.size(0)) * c)
            img_idx.append(torch.ones(class_dets.size(0)) * i)

        # Limit detections to maximum per image detections over all classes
        if len(img_dets) > 0:
            img_dets = torch.cat(img_dets, dim=0)
            img_classes = torch.cat(img_classes, dim=0)
            img_idx = torch.cat(img_idx, dim=0)

            if max_per_image > 0:
                if img_dets.size(0) > max_per_image:
                    # Sort image detections by score in descending order
                    _, order = torch.sort(img_dets[:, 4], dim=0, descending=True)
                    retain = order[:max_per_image]
                    img_dets = img_dets[retain]
                    img_classes = img_classes[retain]
                    img_idx = img_idx[retain]

            dets.append(img_dets)
            classes.append(img_classes)
            img_indices.append(img_idx)

    if len(dets):
        dets = torch.cat(dets, dim=0)
        classes = torch.cat(classes, dim=0)
        img_indices = torch.cat(img_indices, dim=0)
    else:
        dets = torch.tensor(dets)
        classes = torch.tensor(classes)
        img_indices = torch.tensor(img_indices)

    return dets, img_indices, classes


class ConvBnLeakyReLU(nn.Module):
    """
    [CONV]-[BN]-[LeakyReLU]
    """

    def __init__(self, inCh, outCh, kernel):
        super(ConvBnLeakyReLU, self).__init__()
        self.inCh = inCh  # Number of input channels
        self.outCh = outCh  # Number of output channels
        self.kernel = kernel  # Kernel size
        padding = (self.kernel - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.inCh, self.outCh, kernel, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(outCh),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class YOLOv3Layer(nn.Module):
    """
    YOLOv3 Layer

    Reference: https://github.com/jiasenlu/YOLOv3.pytorch/blob/master/misc/yolo.py
    """

    def __init__(self, params):
        super(YOLOv3Layer, self).__init__()
        self.params = params
        self.base = MobileNet(weight_file=self.params.pretrained_weights,
                              params=self.params)  # MobileNetV2
        self.base_out_channels = self.base.out_channels  # [256, 512, 1280]
        self.n_classes = self.params.n_classes
        self.out_channels = 3 * (5 + self.n_classes)  # 3 x (B + C)
        self.anchors = np.array(params.anchors)
        self.n_layers = len(self.anchors) // 3
        self.loss = YOLOLoss(params)

        # Conv layer block for 13x13 feature maps from base network
        self.conv_block13 = self._make_conv_block(inCh=self.base_out_channels[-1],
                                                  channel_list=[512, 1024],
                                                  outCh=self.out_channels)

        # Conv layer block for 26x26 feature maps from base network
        self.conv26 = ConvBnLeakyReLU(inCh=512, outCh=256, kernel=1)
        self.conv_block26 = self._make_conv_block(inCh=self.base_out_channels[-2] + 256,
                                                  channel_list=[256, 512],
                                                  outCh=self.out_channels)

        # Conv layer block for 52x52 feature maps from base network
        self.conv52 = ConvBnLeakyReLU(inCh=256, outCh=128, kernel=1)
        self.conv_block52 = self._make_conv_block(inCh=self.base_out_channels[-3] + 128,
                                                  channel_list=[128, 256],
                                                  outCh=self.out_channels)

    def _make_conv_block(self, inCh, channel_list, outCh):
        """Outputs from Base is passed through a few ConvBNReLU layers"""
        modList = nn.ModuleList([
            ConvBnLeakyReLU(inCh, channel_list[0], kernel=1),
            ConvBnLeakyReLU(channel_list[0], channel_list[1], kernel=3),
            ConvBnLeakyReLU(channel_list[1], channel_list[0], kernel=1),
            ConvBnLeakyReLU(channel_list[0], channel_list[1], kernel=3),
            ConvBnLeakyReLU(channel_list[1], channel_list[0], kernel=1),
            ConvBnLeakyReLU(channel_list[0], channel_list[1], kernel=3),
        ])
        modList.add_module("ConvOut", nn.Conv2d(channel_list[1], outCh,
                                                kernel_size=1, stride=1,
                                                padding=0, bias=True))

        return modList

    def _route(self, in_feature, conv_block):
        for i, conv_module in enumerate(conv_block):
            in_feature = conv_module(in_feature)
            if i == 4:
                route = in_feature
        return in_feature, route

    def forward(self, img, label13, label26, label52):
        # Output from base network
        x52, x26, x13 = self.base(img)

        # Forward pass
        out13, out13_route = self._route(x13, self.conv_block13)  # size: 13x13

        # YOLO branch 1
        x26_in = self.conv26(out13_route)  # size: 13x13
        x26_in = F.interpolate(x26_in, scale_factor=2, mode='nearest')  # size: 13x13 -> 26x26
        x26_in = torch.cat([x26_in, x26], dim=1)
        out26, out26_route = self._route(x26_in, self.conv_block26)  # size: 26x26

        # YOLO branch 2
        x52_in = self.conv52(out26_route)  # size: 26x26
        x52_in = F.interpolate(x52_in, scale_factor=2, mode='nearest')  # size: 26x26 -> 52x52
        x52_in = torch.cat([x52_in, x52], dim=1)
        out52, out52_route = self._route(x52_in, self.conv_block52)  # size: 52x52

        # Compute loss
        loss = self.loss((out13, out26, out52), (label13, label26, label52))

        return loss

    def detect(self, img, img_shape):
        """
        img: array
        img_shape: array
        """
        with torch.no_grad():
            # Output from base network
            x52, x26, x13 = self.base(img)

            # Forward pass
            out13, out13_route = self._route(x13, self.conv_block13)  # size: 13x13

            # YOLO branch 1
            x26_in = self.conv26(out13_route)  # size: 13x13
            x26_in = F.interpolate(x26_in, scale_factor=2, mode='nearest')  # size: 13x13 -> 26x26
            x26_in = torch.cat([x26_in, x26], dim=1)
            out26, out26_route = self._route(x26_in, self.conv_block26)  # size: 26x26

            # YOLO branch 2
            x52_in = self.conv52(out26_route)  # size: 26x26
            x52_in = F.interpolate(x52_in, scale_factor=2, mode='nearest')  # size: 26x26 -> 52x52
            x52_in = torch.cat([x52_in, x52], dim=1)
            out52, out52_route = self._route(x52_in, self.conv_block52)  # size: 52x52

        # Detect
        dets, img_indices, classes = yolo_eval((out13, out26, out52), self.anchors, self.n_classes, img_shape)

        return dets, img_indices, classes


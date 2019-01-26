import torch
import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def preprocess_true_boxes(true_boxes, input_shape, anchors, n_classes):
    """
    Preprocess true bounding boxes to training input format.

    Reference: https://github.com/qqwweee/keras-yolo3/blob/master/yolo3/model.py

    Parameters
    ----------
    true_boxes: Numpy array of shape = (N, T, 5), where N: Number of images,
        T: Number of maximum objects in an image, and 5 corresponds to absolute
        x_min, y_min, x_max, y_max (values relative to input_shape) and number of
        classes.
    input_shape: list, [height, width] and length = 2. NOTE: height and width are 
        multiples of 32
    anchors: Numpy array of shape = (9, 2), and array is of form [width, height]
    n_classes: int, number of classes

    Return
    ------
    y_true: list of 3 Numpy arrays, [(n, 13, 13, 3, 5 + c), ...]
    """
    # Check: class_id in true_boxes must be less than n_classes
    assert (true_boxes[..., 4] < n_classes).all()

    # Create masks for anchors
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # Number of scales
    num_scales = len(anchors) // 3

    # Convert true_boxes values to float and convert input_shape list to numpy array
    true_boxes = np.array(true_boxes, dtype=np.float32)
    input_shape = np.array(input_shape, dtype=np.int32)

    # Compute the center coordinates of bounding boxes: (x, y) is center of bbox
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2

    # Compute the width and height of bounding boxes: (w, h)
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]  # w = x_max - x_min and ...

    # Normalize box center coordinates and box width and height, values range = [0, 1]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]  # (h, w) -> (w, h)
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]  # (h, w) -> (w, h)

    # Number of images
    N = true_boxes.shape[0]

    # Compute grid shapes: [array([13, 13]), array([26, 26]), array([52, 52])] for 416x416
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[s] for s in range(num_scales)]

    # Create a list of zero initialized arrays to store processed ground truth boxes: shape = (N, 13, 13, 3, 5 + C) for 13x13
    y_true = [np.zeros((N, grid_shapes[s][0], grid_shapes[s][1], len(anchor_mask[s]), 5 + n_classes), dtype=np.float32)
              for s in range(num_scales)]

    # Expand dimensions to apply broadcasting
    anchors = np.expand_dims(anchors, axis=0)  # (9, 2) -> (1, 9, 2)

    # Anchor max and min values. The idea is to make upper-left corner the origin
    anchor_maxes = anchors / 2.0
    anchor_mins = - anchor_maxes

    # Mask used to discard rows with zero width values from unnormalized boxes
    valid_mask = boxes_wh[..., 0] > 0  # w > 0 -> True and w = 0 -> False

    # Loop over all the images, compute IoU between box and anchor. Get best anchors
    # and based on best anchors populate array that was created to store processed
    # ground truth boxes in training format

    for b in range(N):
        # Discard rows with zero width values from unnormalized boxes
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue

        # Expand dimensions to apply broadcasting
        wh = np.expand_dims(wh, -2)

        # Unnormalized boxes max and min values. The idea is to make upper-left corner the origin
        box_maxes = wh / 2.0
        box_mins = - box_maxes

        # Compute IoU between anchors and bounding boxes to find best anchors
        intersect_mins = np.maximum(box_mins, anchor_mins)  # Upper left coordinates
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)  # Lower right coordinates
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0)  # Intersection width and height
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # Intersection area
        box_area = wh[..., 0] * wh[..., 1]  # Bbox area
        anchor_area = anchors[..., 0] * anchors[..., 1]  # Anchor area
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Get best anchor for each true bbox
        best_anchor = np.argmax(iou, axis=-1)

        # Populating array that was created to store processed ground truth boxes in training format
        for idx, anchor_idx in enumerate(best_anchor):
            for s in range(num_scales):  # 3 scales
                # Choose the corresponding mask, i.e. best anchor in [6, 7, 8] or [3, 4, 5] or [0, 1, 2]
                if anchor_idx in anchor_mask[s]:
                    i = np.floor(true_boxes[b, idx, 0] * grid_shapes[s][1]).astype('int32')
                    j = np.floor(true_boxes[b, idx, 1] * grid_shapes[s][0]).astype('int32')
                    k = anchor_mask[s].index(anchor_idx)  # best anchor
                    c = true_boxes[b, idx, 4].astype('int32')  # class_id
                    # Populate y_true list of arrays, where s: scale, b: image index, i -> y, j -> x of grid(y, x)
                    # k: best anchor
                    y_true[s][b, j, i, k, 0:4] = true_boxes[b, idx, 0:4]  # Normalized box value
                    y_true[s][b, j, i, k, 4] = 1  # score = 1
                    y_true[s][b, j, i, k, 5 + c] = 1  # class = 1, and the others = 0 (zero initialized)

    return y_true


def bbox_iou(box1, box2):
    """
    Calculate IoU between 2 bounding boxes.

    NOTE: Docstring and comments are based on 13x13, approach similar for 
    26x26 and 52x52

    Parameters
    ----------
    bbox1: Pytorch Tensor, predicted bounding box of size=[13, 13, 3, 4], 
        where 4 specifies x, y, w, h
    bbox2: Pytorch Tensor, ground truth bounding box of size=[num_boxes, 4], 
        where 4 specifies x, y, w, h

    Return
    ------
    IoU Pytorch tensor of size=[13, 13, 3, 1], where 1 specifies IoU
    """
    # Expand dimensions to apply broadcasting
    box1 = box1.unsqueeze(3)  # size: [13, 13, 3, 4] -> [13, 13, 3, 1, 4]

    # Extract xy and wh and compute mins and maxes
    box1_xy = box1[..., :2]  # size: [13, 13, 3, 1, 1, 2]
    box1_wh = box1[..., 2:4]  # size: [13, 13, 3, 1, 1, 2]

    box1_wh_half = box1_wh / 2.0
    box1_mins = box1_xy - box1_wh_half
    box1_maxes = box1_xy + box1_wh_half

    # If box2 i.e. ground truth box is empty tensor, then IoU is empty tensor
    if box2.shape[0] == 0:
        iou = torch.zeros(box1.shape[0:4]).type_as(box1)
    else:
        # Expand dimensions to apply broadcasting
        box2 = box2.view(1, 1, 1, box2.size(0), box2.size(1))  # size: [1, 1, 1, num_boxes, 4]

        # Extract xy and wh and compute mins and maxes
        box2_xy = box2[..., :2]  # size: [1, 1, 1, num_boxes, 2]
        box2_wh = box2[..., 2:4]  # size: [1, 1, 1, num_boxes, 2]
        box2_wh_half = box2_wh / 2.0
        box2_mins = box2_xy - box2_wh_half
        box2_maxes = box2_xy + box2_wh_half

        # Compute boxes intersection mins, maxes and area
        intersect_mins = torch.max(box1_mins, box2_mins)
        intersect_maxes = torch.min(box1_maxes, box2_maxes)
        intersect_wh = torch.clamp(intersect_maxes - intersect_mins, min=0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # size: [13, 13, 3, num_boxes]

        # Compute box1 and box2 areas
        box1_area = box1_wh[..., 0] * box1_wh[..., 1]  # size: [13, 13, 3, 1]
        box2_area = box2_wh[..., 0] * box2_wh[..., 1]  # size: [1, 1, 1, num_boxes]

        # Compute IoU
        iou = intersect_area / (box1_area + box2_area - intersect_area)  # size: [13, 13, 3, num_boxes]

    return iou


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """
    Convert YOLO bounding box predictions to bounding box coordinates (x_min,
    y_min, x_max, y_max)

    Parameters
    ----------
    box_xy: PyTorch tensor, box_xy output from YOLODetector, size: [1, 13, 13, 3, 2]
    box_wh: PyTorch tensor, box_wh output from YOLODetector, size: [1, 13, 13, 3, 2]
    input_shape: ? e.g. 416x416
    image_shape: ? e.g. 640x480
    """
    # [x, y] -> [y, x]
    box_yx = torch.stack((box_xy[..., 1], box_xy[..., 0]), dim=4)
    # [w, h] -> [h, w]
    box_hw = torch.stack((box_wh[..., 1], box_wh[..., 0]), dim=4)

    factor = torch.min((input_shape / image_shape))  # min(416./640., 416./480.)

    # New shape: round(640. * 416./640., 480. * 416./640.)
    new_shape = torch.round(image_shape * factor)

    # Compute offset: [0., (416.-312.)/(2*416.)] i.e. [0, 0.125]
    offset = (input_shape - new_shape) / (2. * input_shape)

    # Compute scale: [1., 416./312.] i.e. [1., 1.33]
    scale = input_shape / new_shape

    # Convert boxes from center (y,x) and (h, w) to (y_min, x_min) and (y_max, x_max)
    box_yx = (box_yx - offset) * scale  # [(x-0.)*1., (y-0.125)*1.33]
    box_hw = box_hw * scale  # [h*1, w*1.33]

    box_mins = box_yx - (box_hw / 2.)  # x_min = (x-0.)*1. - h/2, y_min = ...
    box_maxes = box_yx + (box_hw / 2.)  # x_max = (x-0.)*1. + h/2, y_max = ...

    # Stack box coordinates in proper order
    boxes = torch.stack([
        box_mins[..., 0],  # y_min
        box_mins[..., 1],  # x_min
        box_maxes[..., 0],  # y_max
        box_maxes[..., 1],  # x_max
    ], dim=4)  # size: [1, 13, 13, 3, 4]

    # Scale boxes back to original image shape
    boxes = boxes * torch.cat([image_shape, image_shape]).view(1, 1, 1, 1, 4)

    return boxes


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def horizontal_flip(img, bboxes):
    img = img.copy()
    bboxes = bboxes.copy()
    img_center = np.array(img.shape[:2])[::-1] / 2
    img_center = np.hstack((img_center, img_center))
    img = img[:, ::-1, :]
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + 2 * (img_center[[0, 2]] - bboxes[:, [0, 2]])
    box_w = abs(bboxes[:, 0] - bboxes[:, 2])
    bboxes[:, 0] -= box_w
    bboxes[:, 2] += box_w
    return img, bboxes


def get_random_augmented_data(annotation_line, input_shape, augment=True, hue=0.1, saturation=1.5, value=1.5,
                              max_boxes=25):
    """
    Random preprocessing for real-time data augmentation. The augmentations that can be applied 
    randomly are: (1) HSV distortions and (2) Horizontal flip; 

    Reference: https://github.com/qqwweee/keras-yolo3/blob/master/yolo3/utils.py

    Parameters
    ----------
    annotation_line: str
    input_shape: tuple
    augment: bool, 
    hue: float
    saturation: float
    max_boxes: int
    """
    # Extract data from annotation string
    line = annotation_line.split()

    # Bounding boxes, size: [num_boxes, 5]
    bbox = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    # Read image
    image = Image.open(line[0])
    img_w, img_h = image.size

    # Model input shape
    h, w = input_shape

    # Get scale for image/box resizing
    scale = min(w / img_w, h / img_h)

    # Compute new width and height of image
    new_img_w = int(img_w * scale)
    new_img_h = int(img_h * scale)

    # Compute upper left corner coordinates for pasting image
    dx = (w - new_img_w) // 2
    dy = (h - new_img_h) // 2

    # Resize image while keeping original aspect ratio
    image = image.resize(size=(new_img_w, new_img_h), resample=Image.BICUBIC)
    new_image = Image.new(mode='RGB', size=(w, h), color=(128, 128, 128))
    new_image.paste(im=image, box=(dx, dy))
    image = np.array(new_image)
    # image = np.array(new_image)/255.0  # RGB values in range [0, 1]

    # Correct bounding boxes to new image size
    bboxes = np.zeros((max_boxes, 5))

    if len(bbox) > 0:
        # Shuffle the boxes
        np.random.shuffle(bbox)

        # Scale the boxes to account for resized image
        bbox[:, [0, 2]] = bbox[:, [0, 2]] * scale + dx  # x_min and x_max
        bbox[:, [1, 3]] = bbox[:, [1, 3]] * scale + dy  # y_min and y_max

    # Apply random augmentations
    if augment:
        # Random HSV jitter
        hue_jitter = rand() < 0.5
        if hue_jitter:
            hue = rand(-hue, hue)
            saturation = rand(1, saturation) if rand() < 0.5 else 1 / rand(1, saturation)
            value = rand(1, value) if rand() < 0.5 else 1 / rand(1, value)

            # Convert RGB to HSV
            hsv_img = rgb_to_hsv(np.array(image) / 255.0)  # Values must be in the range [0, 1]
            hsv_img[..., 0] += hue
            hsv_img[..., 0][hsv_img[..., 0] > 1] -= 1
            hsv_img[..., 0][hsv_img[..., 0] < 0] += 1
            hsv_img[..., 1] *= saturation
            hsv_img[..., 2] *= value
            hsv_img[hsv_img > 1] = 1
            hsv_img[hsv_img < 0] = 0
            image = hsv_to_rgb(hsv_img)  # RGB values in range [0, 1]

        # Randomly flip images and boxes horizontally
        flip = rand() < 0.5
        if flip:
            image, bbox = horizontal_flip(img=image, bboxes=bbox)

    if len(bbox) > max_boxes:
        bbox = bbox[:max_boxes]

    bboxes[:len(bbox)] = bbox.clip(min=0)

    return image, bboxes


def non_maximum_suppression(boxes, thresh=0.3):
    """
    Reference: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    Parameters
    ----------
    boxes: NumPy array, size: [?, 5], where ? can be some int, and 5 specifies 
        x_min, y_min, x_max, y_max
    thresh: float

    Return
    ------
    keep: list of indices of boxes to keep
    """
    # Get the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]  # Confidence scores

    # Compute area of each bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort the bounding boxes by confidence score
    indices = scores.argsort()

    # Initialize a list for indices to keep
    keep = []

    while len(indices) > 0:
        # Grab the last index in the indices list and add the
        # index value to the keep list
        last = len(indices) - 1
        i = indices[last]
        keep.append(i)

        # Find the largest (x, y) coordinates for the start of the
        # bounding box and the smallest (x, y) coordinates for the
        # end of the bounding box
        xx1 = np.maximum(x1[i], x1[indices[:last]])
        yy1 = np.maximum(y1[i], y1[indices[:last]])
        xx2 = np.minimum(x2[i], x2[indices[:last]])
        yy2 = np.minimum(y2[i], y2[indices[:last]])

        # Compute the width and height of the bounding boxes
        w = np.maximum(0., xx2 - xx1 + 1)
        h = np.maximum(0., yy2 - yy1 + 1)

        # Compute the IoU
        inter_area = w * h
        iou = inter_area / (areas[i] + areas[indices[:last]] - inter_area)

        # Delete all the indices where
        indices = np.delete(indices, np.concatenate(([last], np.where(iou > thresh)[0])))

    return keep


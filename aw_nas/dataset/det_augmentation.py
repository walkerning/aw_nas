"""Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325

Ellis Brown, Max deGroot
"""

import math
import random

import numpy as np

import cv2
from aw_nas.utils.box_utils import matrix_iou


def _crop(image, boxes, labels):
    height, width, _ = image.shape

    if len(boxes) == 0:
        return image, boxes, labels

    while True:
        mode = random.choice((
            None,
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        ))

        if mode is None:
            return image, boxes, labels

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            scale = random.uniform(0.3, 1.)
            min_ratio = max(0.5, scale * scale)
            max_ratio = min(2, 1. / scale / scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            w = int(scale * ratio * width)
            h = int((scale / ratio) * height)

            l = random.randrange(width - w)
            t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))

            iou = matrix_iou(boxes, roi[np.newaxis])

            if not (min_iou <= iou.min() and iou.max() <= max_iou):
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                     .all(axis=1)
            boxes_t = boxes[mask].copy()
            labels_t = labels[mask].copy()
            if len(boxes_t) == 0:
                continue

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            return image_t, boxes_t, labels_t


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy().astype(np.float32)

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    return image


def _expand(image, boxes, fill, p):
    if random.random() > p:
        return image, boxes

    height, width, depth = image.shape
    for _ in range(50):
        scale = random.uniform(1, 4)

        min_ratio = max(0.5, 1./scale/scale)
        max_ratio = min(2, scale*scale)
        ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
        ws = scale*ratio
        hs = scale/ratio
        if ws < 1 or hs < 1:
            continue
        w = int(ws * width)
        h = int(hs * height)

        left = random.randint(0, w - width)
        top = random.randint(0, h - height)

        boxes_t = boxes.copy()
        boxes_t[:, :2] += (left, top)
        boxes_t[:, 2:] += (left, top)

        expand_image = np.empty(
            (h, w, depth),
            dtype=image.dtype)
        expand_image[:, :] = fill
        expand_image[top:top + height, left:left + width] = image
        image = expand_image

        return image, boxes_t


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def _elastic(image, p, alpha=None, sigma=None, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
     From:
     https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
    """
    if random.random() > p:
        return image
    if alpha is None:
        alpha = image.shape[0] * random.uniform(0.5, 2)
    if sigma is None:
        sigma = int(image.shape[0] * random.uniform(0.5, 1))
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape[:2]

    dx, dy = [cv2.GaussianBlur((random_state.rand(
        *shape) * 2 - 1) * alpha, (sigma | 1, sigma | 1), 0) for _ in range(2)]
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x, y = np.clip(
        x+dx, 0, shape[1]-1).astype(np.float32), np.clip(y+dy, 0, shape[0]-1).astype(np.float32)
    return cv2.remap(image, x, y, interpolation=cv2.INTER_LINEAR, borderValue=0, borderMode=cv2.BORDER_REFLECT)


def preproc_for_test(image, insize):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC,
                      cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, insize, interpolation=interp_method)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return image.transpose(2, 0, 1)


def draw_bbox(image, bbxs, color=(0, 255, 0)):
    img = image.copy()
    bbxs = np.array(bbxs).astype(np.int32)
    for bbx in bbxs:
        cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[2], bbx[3]), color, 5)
    return img


class Preproc(object):
    def __init__(self, resize, p, writer=None):
        self.resize = resize
        self.p = p
        self.writer = writer  # writer used for tensorboard visualization
        self.epoch = 0

    def __call__(self, image, boxes, labels):
        # some bugs
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = np.array(boxes)
        if self.p == -2:  # abs_test
            targets = np.zeros((1, 5))
            targets[0] = image.shape[0]
            targets[0] = image.shape[1]
            image = preproc_for_test(image, self.resize)
            return image, targets[:, :-1], targets[:, -1]

        if len(boxes) == 0:
            targets = np.zeros((1, 5))
            # some ground truth in coco do not have bounding box! weird!
            image = preproc_for_test(image, self.resize)
            return image, targets[:, :-1], targets[:, -1]
        if self.p == -1:  # eval
            height, width, _ = image.shape
            boxes[:, 0::2] /= width
            boxes[:, 1::2] /= height
            # boxes *= self.resize
            image = preproc_for_test(image, self.resize)
            return image, boxes, labels

        targets = np.concatenate([boxes, labels.reshape(-1, 1)], 1)
        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :-1]
        labels_o = targets_o[:, -1]
        boxes_o[:, 0::2] /= width_o
        boxes_o[:, 1::2] /= height_o
        # boxes_o *= self.resize
        # labels_o = np.expand_dims(labels_o, 1)

        if self.writer is not None:
            image_show = draw_bbox(image, boxes)
            self.writer.add_image('preprocess/input_image',
                                  image_show, self.epoch)

        image_t, boxes, labels = _crop(image, boxes, labels)
        if self.writer is not None:
            image_show = draw_bbox(image_t, boxes)
            self.writer.add_image('preprocess/crop_image',
                                  image_show, self.epoch)

        image_t = _distort(image_t)
        if self.writer is not None:
            image_show = draw_bbox(image_t, boxes)
            self.writer.add_image(
                'preprocess/distort_image', image_show, self.epoch)

        image_t, boxes = _expand(image_t, boxes, 0, self.p)
        if self.writer is not None:
            image_show = draw_bbox(image_t, boxes)
            self.writer.add_image(
                'preprocess/expand_image', image_show, self.epoch)

        image_t, boxes = _mirror(image_t, boxes)
        if self.writer is not None:
            image_show = draw_bbox(image_t, boxes)
            self.writer.add_image(
                'preprocess/mirror_image', image_show, self.epoch)

        if self.writer is not None:
            self.release_writer()

        height, width, _ = image_t.shape
        image_t = preproc_for_test(image_t, self.resize)
        boxes = boxes.copy()
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height
        b_w = (boxes[:, 2] - boxes[:, 0])*1.
        b_h = (boxes[:, 3] - boxes[:, 1])*1.
        mask_b = np.minimum(b_w, b_h) > 0.01
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()
        # boxes_t *= self.resize

        if len(boxes_t) == 0:
            image = preproc_for_test(image_o, self.resize)
            return image, boxes_o, labels_o

        return image_t, boxes_t, labels_t

    def add_writer(self, writer, epoch=None):
        self.writer = writer
        self.epoch = epoch if epoch is not None else self.epoch + 1

    def release_writer(self):
        self.writer = None

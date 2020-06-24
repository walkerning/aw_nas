import logging
import os
import pathlib
import pickle
import xml.etree.ElementTree as ET
from datetime import datetime

import cv2
import numpy as np
import six
from torchvision import datasets, transforms

from aw_nas.dataset.base import BaseDataset
from aw_nas.dataset.transform import *
from aw_nas.dataset.voc_eval import do_python_eval, write_voc_results_file
from aw_nas.utils import logger
from aw_nas.utils.box_utils import *
from aw_nas.utils.torch_utils import Cutout

CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor')


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [
            int(bbox.find('xmin').text) - 1,
            int(bbox.find('ymin').text) - 1,
            int(bbox.find('xmax').text) - 1,
            int(bbox.find('ymax').text) - 1
        ]
        objects.append(obj_struct)

    return objects


def write_voc_results_file(write_dir, all_boxes, dataset):
    for cls_ind, cls_name in enumerate(dataset.class_names[1:], 1):
        logger.info('Writing {:s} VOC results file'.format(cls_name))
        # filename = write_dir + '/' + str(cls_name)
        filename = write_dir + '/' + 'comp4_det_test' + '_{:s}.txt'.format(
            cls_name)
        with open(filename, 'wt') as f:
            for im_ind, (year, index) in enumerate(dataset.ids):
                dets = all_boxes[cls_ind].get(im_ind, [])
                if len(dets) == 0:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                        index, dets[k, -1], dets[k, 0] + 1, dets[k, 1] + 1,
                        dets[k, 2] + 1, dets[k, 3] + 1))


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                logger.info('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        logger.info('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {
            'bbox': bbox,
            'difficult': difficult,
            'det': det
        }

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) * (BBGT[:, 3] - BBGT[:, 1]) -
                       inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def do_python_eval(annopath,
                   imgsetpath,
                   class_names,
                   output_dir='output',
                   use_07=True):
    cachedir = os.path.join(output_dir, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    logger.info('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls_name in enumerate(class_names[1:]):
        # filename = output_dir + '/' + str(cls_name)
        filename = output_dir + '/' + 'comp4_det_test' + '_{:s}.txt'.format(
            cls_name)
        rec, prec, ap = voc_eval(filename,
                                 annopath,
                                 imgsetpath,
                                 cls_name,
                                 cachedir,
                                 ovthresh=0.5,
                                 use_07_metric=use_07_metric)
        aps += [ap]
        logger.info('AP for {} = {:.4f}'.format(cls_name, ap))
        with open(os.path.join(output_dir, cls_name + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

    logger.info('{:.3f}'.format(np.mean(aps)))
    logger.info('~~~~~~~~')
    logger.info('')
    logger.info(
        '--------------------------------------------------------------')
    logger.info('Results computed with the **unofficial** Python eval code.')
    logger.info(
        'Results should be very close to the official MATLAB eval code.')
    logger.info(
        '--------------------------------------------------------------')
    return aps


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(output_dir, box_list, dataset)
    return do_python_eval(dataset.anno_path % ("VOC2007", "%s"),
                          dataset.image_sets_file, dataset.class_names,
                          output_dir)


class VOCDataset(object):
    def __init__(self,
                 data_dir,
                 image_sets=[("VOC2007", "trainval"), ("VOC2012", "trainval")],
                 transform=None,
                 target_transform=None,
                 is_test=False,
                 keep_difficult=False,
                 label_file=None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.data_dir = data_dir
        self.root = pathlib.Path(self.data_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.anno_path = os.path.join(data_dir, '%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join(data_dir, '%s', 'JPEGImages', '%s.jpg')
        self.ids = []

        for sub_dir in image_sets:
            self.image_sets_file = self.root / ("%s/ImageSets/Main/%s.txt" %
                                                sub_dir)
            self.ids.extend(
                VOCDataset._read_image_ids(self.image_sets_file, sub_dir[0]))
        self.keep_difficult = keep_difficult
        self.is_test = is_test

        # if the labels file exists, read in the class names
        label_file_name = self.root / "labels.txt"

        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()

            # classes should be a comma separated list

            classes = class_string.split(',')
            classes = [elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("VOC Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default VOC classes.")
            self.class_names = CLASSES

        self.class_dict = {
            class_name: i
            for i, class_name in enumerate(self.class_names)
        }

        def collate_fn(batch):
            inputs = [b[0] for b in batch]
            targets = [b[1] for b in batch]
            inputs = torch.stack(inputs, 0)
            return inputs, targets

        self.kwargs = {"collate_fn": collate_fn}

    def __getitem__(self, index):
        image, boxes, labels, height, width = self._getitem(index)
        return image, (boxes, labels, index, height, width)

    def _getitem(self, index):
        img_id, (boxes, labels, is_difficult) = self.get_annotation(index)
        image = self._read_image(img_id)
        height, width, channels = image.shape

        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes, labels)

        if self.target_transform is not None:
            boxes, labels = self.target_transform(boxes, labels)

        image = torch.from_numpy(image).to(torch.float)
        boxes = torch.from_numpy(boxes).to(torch.float)
        labels = torch.from_numpy(labels).to(torch.long)
        return image, boxes, labels, height, width

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file, year):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append((year, line.rstrip()))
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.anno_path % (image_id)
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(
                    int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes,
                         dtype=np.float32), np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self._imgpath % image_id
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float)
        image /= 255.
        return image


class VOC(BaseDataset):
    NAME = "voc"

    def __init__(self,
                 load_train_only=False,
                 class_name_file=None,
                 random_choose=False,
                 random_seed=123,
                 train_sets=[("VOC2007", "trainval"), ("VOC2012", "trainval")],
                 test_sets=[("VOC2007", "test")],
                 train_crop_size=300,
                 test_crop_size=300,
                 image_mean=[0.485, 0.456, 0.406],
                 image_std=[0.229, 0.224, 0.225],
                 iou_threshold=0.5,
                 keep_difficult=False,
                 relative_dir=None):
        super(VOC, self).__init__(relative_dir)
        self.load_train_only = load_train_only
        self.train_data_dir = os.path.join(self.data_dir, "train")
        self.class_name_file = class_name_file

        train_transform = TrainAugmentation(train_crop_size,
                                            np.array(image_mean),
                                            np.array(image_std))
        test_transform = TestTransform(test_crop_size, np.array(image_mean),
                                       np.array(image_std))

        self.datasets = {}
        self.datasets['train'] = VOCDataset(self.train_data_dir, train_sets,
                                            train_transform)
        self.grouped_annotation = {}

        if not self.load_train_only:
            self.test_data_dir = os.path.join(self.data_dir, "test")
            self.datasets['test'] = VOCDataset(self.test_data_dir,
                                               test_sets,
                                               test_transform,
                                               is_test=True)

    def splits(self):
        return self.datasets

    @classmethod
    def data_type(cls):
        return "image"

    @staticmethod
    def _read_names_from_file(path):
        with open(path, "r") as f:
            return f.read().strip().split("\n")

    @staticmethod
    def _write_names_to_file(names, path):
        with open(path, "w") as f:
            f.write("\n".join(names))

    def evaluate_detections(self, box_list, output_dir):
        dataset = self.datasets["test"]
        write_voc_results_file(output_dir, box_list, dataset)
        return do_python_eval(dataset.anno_path % ("VOC2007", "%s"),
                              dataset.image_sets_file, dataset.class_names,
                              output_dir)

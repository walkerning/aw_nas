import os
import json
import pickle

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from aw_nas.dataset.base import BaseDataset
from aw_nas.dataset.det_transform import *

min_keypoints_per_image = 10


def _collate_fn(batch):
    inputs = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    inputs = torch.stack(inputs, 0)
    return inputs, targets


def collate_fn(batch):
    """
    For GroupSampler to stack images when `keep_ratio` is True.
    """
    inputs = [b.pop("img").data for b in batch]
    height = max([inp.shape[-2] for inp in inputs])
    width = max([inp.shape[-1] for inp in inputs])
    targets = [b for b in batch]
    inputs = torch.stack(
        [F.pad(i, [0, width - i.shape[-1], 0, height - i.shape[-2]]) for i in inputs], 0
    )
    return inputs, targets


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODetection(data.Dataset):
    """COCO Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to COCO folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(
        self,
        root,
        image_sets,
        transform=None,
        target_transform=None,
        is_test=False,
        max_images=None,
        remove_no_anno=False,
        has_background=True,
        remove_invalid_labels=False,
        dataset_name="COCO",
    ):
        self.root = root
        self.cache_path = os.path.join(self.root, "awnas_cache")
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.is_test = is_test
        self.has_background = has_background
        self.remove_invalid_labels = remove_invalid_labels

        self.img_paths = list()
        self.image_indexes = list()
        self.annotations = list()
        self._view_map = {
            "minival2014": "val2014",  # 5k val2014 subset
            "valminusminival2014": "val2014",  # val2014 \setminus minival2014
            "test-dev2015": "test2015",
        }
        # self.data_name = list()
        # self.data_len = list()
        for (year, image_set) in image_sets:
            coco_name = image_set + year
            data_name = (
                self._view_map[coco_name] if coco_name in self._view_map else coco_name
            )
            annofile = self._get_ann_file(coco_name)
            _COCO = COCO(annofile)
            self._COCO = _COCO
            self.coco_name = coco_name
            cats = _COCO.loadCats(_COCO.getCatIds())
            self._classes = tuple(
                [c["name"] for c in cats if not remove_invalid_labels or c["name"]]
            )
            if self.has_background:
                self._classes = ("__background__",) + self._classes
            self.num_classes = len(self._classes)
            self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))
            self._class_to_coco_cat_id = dict(
                zip([c["name"] for c in cats], _COCO.getCatIds())
            )
            indexes = _COCO.getImgIds()

            if image_set.find("test") != -1:
                print("test set will not load annotations!")
            else:
                annos, indexes = self._load_coco_annotations(
                    coco_name, indexes, _COCO, remove_no_anno=remove_no_anno
                )
                self.annotations.extend(annos)

            self.image_indexes.extend(indexes)
            self.img_paths.extend(self._load_coco_img_path(coco_name, indexes))

        self.image_indexes = self.image_indexes[:max_images]
        self.img_paths = self.img_paths[:max_images]

        self.kwargs = {"collate_fn": collate_fn}
        if not self.is_test:
            self.kwargs["group_sample"] = True

        self._set_group_flag()

    def image_path_from_index(self, name, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   images/train2014/COCO_train2014_000000119993.jpg
        # file_name = ('COCO_' + name + '_' +
        #              str(index).zfill(12) + '.jpg')
        # change Example image path for index=119993 to images/train2017/000000119993.jpg
        file_name = str(index).zfill(12) + ".jpg"
        image_path = os.path.join(self.root, name, file_name)
        assert os.path.exists(image_path), "Path does not exist: {}".format(image_path)
        return image_path

    def _get_ann_file(self, name):
        prefix = "instances" if name.find("test") == -1 else "image_info"
        return os.path.join(self.root, "annotations", prefix + "_" + name + ".json")

    def _set_group_flag(self):
        self.group_index = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self._COCO.loadImgs(self.image_indexes[i])[0]
            self.group_index[i] = img_info["width"] > img_info["height"]

    def _rand_another(self, idx):
        pool = np.where(self.grou_index == self.group_index[idx])[0]
        return np.random.choice(pool)

    def _load_coco_annotations(self, coco_name, indexes, _COCO, remove_no_anno=False):
        cache_file = os.path.join(self.cache_path, coco_name + "_gt_roidb.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fid:
                roidb, valid_indexes = pickle.load(fid)
            print("{} gt roidb loaded from {}".format(coco_name, cache_file))
            return roidb, valid_indexes

        print("parsing gt roidb for {}".format(coco_name))
        gt_roidb = [
            (index, self._annotation_from_index(index, _COCO, remove_no_anno))
            for index in indexes
        ]
        valid_indexes = [index for index, anno in gt_roidb if anno is not None]
        gt_roidb = [anno for index, anno in gt_roidb if anno is not None]
        with open(cache_file, "wb") as fid:
            pickle.dump([gt_roidb, valid_indexes], fid, pickle.HIGHEST_PROTOCOL)
        print("wrote gt roidb to {}".format(cache_file))
        return gt_roidb, valid_indexes

    def _load_coco_img_path(self, coco_name, indexes):
        cache_file = os.path.join(self.cache_path, coco_name + "_img_path.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fid:
                img_path = pickle.load(fid)
            print("{} img path loaded from {}".format(coco_name, cache_file))
            return img_path

        print("parsing img path for {}".format(coco_name))
        img_path = [self.image_path_from_index(coco_name, index) for index in indexes]
        with open(cache_file, "wb") as fid:
            pickle.dump(img_path, fid, pickle.HIGHEST_PROTOCOL)
        print("wrote img path to {}".format(cache_file))
        return img_path

    def _annotation_from_index(
        self, index, _COCO, remove_images_without_annotations=False
    ):
        """
        Loads COCO bounding-box instance annotations. Crowd instances are
        handled by marking their overlaps (with all categories) to -1. This
        overlap value means that crowd "instances" are excluded from training.
        """
        im_ann = _COCO.loadImgs(index)[0]
        width = im_ann["width"]
        height = im_ann["height"]

        annIds = _COCO.getAnnIds(imgIds=index, iscrowd=None)
        objs = _COCO.loadAnns(annIds)
        if remove_images_without_annotations and not has_valid_annotation(objs):
            return None
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = x1 + np.max((0, obj["bbox"][2]))
            y2 = y1 + np.max((0, obj["bbox"][3]))
            if obj["area"] > 0 and x2 - x1 >= 1 and y2 - y1 >= 1:
                if obj.get("iscrowd", False):
                    continue
                obj["clean_bbox"] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        res = np.zeros((num_objs, 6))

        # Lookup table to map from COCO category ids to our internal class
        # indices
        coco_cat_id_to_class_ind = dict(
            [
                (self._class_to_coco_cat_id[cls], self._class_to_ind[cls])
                for cls in self._classes[int(self.has_background) :]
            ]
        )

        # do not transform label 1~90 to 1~80
        for ix, (annId, obj) in enumerate(zip(annIds, objs)):
            cls = coco_cat_id_to_class_ind[obj["category_id"]]
            res[ix, :4] = obj["clean_bbox"]
            res[ix, 4] = cls
            res[ix, 5] = annId

        return res

    def __getitem__(self, index):
        """
        Return:
            dict of {
                ori_boxes: x, y, w, h
                boxes: x1, y1, x2, y2
                labels: list of int
                image_id: int
                anno_ids: list of int
                shape: [ori_w, ori_h]
            }
        """
        if self.is_test:
            return self._getitem(index)
        while True:
            data = self._getitem(index)
            if data is None:
                index = self._rand_another(index)
                continue
            return data
        # image, boxes, labels, height, width, ori_boxes, annIds = self._getitem(index)
        ori_boxes[:, 2] -= ori_boxes[:, 0]
        ori_boxes[:, 3] -= ori_boxes[:, 1]
        return (
            image,
            {
                "ori_boxes": ori_boxes,
                "boxes": boxes,
                "labels": labels,
                "image_id": self.image_indexes[index],
                "anno_ids": annIds,
                "shape": [height, width],
            },
        )

    def _getitem(self, index):
        img_path = self.img_paths[index]
        target = self.annotations[index]
        ori_boxes, labels, annIds = target[:, :4], target[:, 4], target[:, 5]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # .astype(np.float32)
        height, width, _ = img.shape

        if self.transform is not None:
            # img, boxes, labels = self.transform(img, ori_boxes.copy(), labels)
            results = self.transform(img, ori_boxes.copy(), labels)
            ori_boxes[:, 2] -= ori_boxes[:, 0]
            ori_boxes[:, 3] -= ori_boxes[:, 1]
            results.update(
                {
                    "ori_boxes": ori_boxes,
                    "anno_ids": annIds,
                    "shape": (height, width),
                    "ori_shape": (height, width, 3),
                    "labels": labels,
                    "image_id": self.image_indexes[index],
                }
            )
            return results

        img = self._to_tensor(img).to(torch.float)
        boxes = self._to_tensor(boxes).to(torch.float)
        labels = self._to_tensor(labels).to(torch.long)
        return img, boxes, labels, height, width, ori_boxes, annIds

    def _to_tensor(self, var):
        if isinstance(var, torch.Tensor):
            return var
        if isinstance(var, np.ndarray):
            return torch.from_numpy(var)
        return torch.tensor(var)

    def __len__(self):
        return len(self.img_paths)

    def pull_image(self, index):
        """Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        """
        img_path = self.img_paths[index]
        return cv2.imread(img_path, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        """Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        """
        anno = self.annotations[index]
        if self.target_transform is not None:
            anno = self.target_transform(anno)
        return anno

    def pull_tensor(self, index):
        """Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        """
        return torch.tensor(self.pull_image(index)).unsqueeze_(0)


class COCODataset(BaseDataset):
    NAME = "coco"

    def __init__(
        self,
        load_train_only=False,
        class_name_file=None,
        random_choose=False,
        train_sets=[("2017", "train")],
        test_sets=[("2017", "val")],
        train_crop_size=300,
        test_crop_size=300,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        image_norm_factor=255.0,
        image_bias=0.0,
        iou_threshold=0.5,
        keep_difficult=False,
        max_images=None,
        train_pipeline=None,
        test_pipeline=None,
        remove_invalid_labels=False,
        has_background=True,
        remove_no_anno=False,
    ):
        super(COCODataset, self).__init__()
        self.load_train_only = load_train_only
        self.train_data_dir = self.data_dir
        self.class_name_file = class_name_file

        self.iou_threshold = iou_threshold

        if isinstance(train_crop_size, int):
            train_crop_size = (train_crop_size,) * 2
        if isinstance(test_crop_size, int):
            test_crop_size = (test_crop_size,) * 2

        assert len(train_crop_size) == 2 and len(test_crop_size) == 2

        train_transform = TrainAugmentation(
            train_pipeline,
            train_crop_size,
            np.array(image_mean),
            np.array(image_std),
            image_norm_factor,
            image_bias,
        )
        test_transform = TestTransform(
            test_pipeline,
            test_crop_size,
            np.array(image_mean),
            np.array(image_std),
            image_norm_factor,
            image_bias,
        )

        self.datasets = {}
        self.datasets["train"] = COCODetection(
            self.train_data_dir,
            train_sets,
            train_transform,
            remove_no_anno=remove_no_anno,
            remove_invalid_labels=remove_invalid_labels,
            has_background=has_background,
        )

        self.datasets["train_testTransform"] = COCODetection(
            self.train_data_dir,
            train_sets,
            test_transform,
            remove_no_anno=remove_no_anno,
            remove_invalid_labels=remove_invalid_labels,
            has_background=has_background,
        )

        if not self.load_train_only:
            self.test_data_dir = self.data_dir
            self.datasets["test"] = COCODetection(
                self.test_data_dir,
                test_sets,
                test_transform,
                is_test=True,
                max_images=max_images,
                remove_no_anno=remove_no_anno,
                remove_invalid_labels=remove_invalid_labels,
                has_background=has_background,
            )

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

    def evaluate_detections(self, all_boxes, output_dir):
        dataset = self.datasets.get("test", self.datasets["train"])
        return dataset.evaluate_detections(all_boxes, output_dir, dataset.image_indexes)


class COCOMMDetDataset(BaseDataset):
    NAME = "coco_mmdet"

    def __init__(
        self,
        train_ann_file,
        test_ann_file,
        train_img_prefix,
        test_img_prefix,
        classes=None,
        max_images=None,
        train_pipeline=None,
        test_pipeline=None,
    ):
        super(COCOMMDetDataset, self).__init__()

        from mmdet.datasets.coco import CocoDataset

        self.datasets = {}
        self.datasets["train"] = CocoDataset(
            train_ann_file,
            train_pipeline,
            data_root=self.data_dir,
            img_prefix=train_img_prefix,
        )

        self.datasets["train_testTransform"] = CocoDataset(
            train_ann_file,
            test_pipeline,
            data_root=self.data_dir,
            img_prefix=train_img_prefix,
            test_mode=True,
        )

        self.datasets["test"] = CocoDataset(
            test_ann_file,
            test_pipeline,
            data_root=self.data_dir,
            img_prefix=test_img_prefix,
            test_mode=True,
        )

        for ds in self.datasets.values():
            ds.kwargs = {"collate_fn": collate_fn}
            if not ds.test_mode:
                ds.kwargs["group_sample"] = True
                ds.group_index = ds.flag

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

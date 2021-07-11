# pylint: disable=invalid-name
import os
import pickle

import numpy as np

from aw_nas.objective.detection_utils.base import Metrics
from aw_nas.utils import getLogger

_LOGGER = getLogger("det.metrics")

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as e:
    _LOGGER.warn(
        (
            "Cannot import pycocotools: {}\n" "Should install EXTRAS_REQUIRE `det`"
        ).format(e)
    )

    class COCODetectionMetrics(Metrics):
        NAME = "coco"

        def __new__(cls, *args, **kwargs):
            _LOGGER.error(
                "COCODetectionMetrics cannot be used. Install required dependencies!"
            )
            raise Exception()

        def __call__(self, boxes):
            pass


else:

    class COCODetectionMetrics(Metrics):
        NAME = "coco"

        def __init__(
            self,
            class_names=None,
            remove_invalid_labels=False,
            has_background=True,
            eval_dir=None,
            schedule_cfg=None,
        ):
            super(COCODetectionMetrics, self).__init__(eval_dir, schedule_cfg)

            self.remove_invalid_labels = remove_invalid_labels
            self.has_background = has_background

            self.gt_recs = {}
            self.mAP = None
            self.eval_ids = []
            self._COCO = COCO()
            self.class_names = class_names or (
                "__background__",
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "airplane",
                "bus",
                "train",
                "truck",
                "boat",
                "traffic light",
                "fire hydrant",
                "",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
                "",
                "backpack",
                "umbrella",
                "",
                "",
                "handbag",
                "tie",
                "suitcase",
                "frisbee",
                "skis",
                "snowboard",
                "sports ball",
                "kite",
                "baseball bat",
                "baseball glove",
                "skateboard",
                "surfboard",
                "tennis racket",
                "bottle",
                "",
                "wine glass",
                "cup",
                "fork",
                "knife",
                "spoon",
                "bowl",
                "banana",
                "apple",
                "sandwich",
                "orange",
                "broccoli",
                "carrot",
                "hot dog",
                "pizza",
                "donut",
                "cake",
                "chair",
                "couch",
                "potted plant",
                "bed",
                "",
                "dining table",
                "",
                "",
                "toilet",
                "",
                "tv",
                "laptop",
                "mouse",
                "remote",
                "keyboard",
                "cell phone",
                "microwave",
                "oven",
                "toaster",
                "sink",
                "refrigerator",
                "",
                "book",
                "clock",
                "vase",
                "scissors",
                "teddy bear",
                "hair drier",
                "toothbrush",
            )

            if remove_invalid_labels:
                self.index_mapping = {
                    i: j
                    for i, j in zip(
                        [i for i, n in enumerate(self.class_names) if n],
                        range(len(self.class_names)),
                    )
                }
                self.invert_mapping = {j: i for i, j in self.index_mapping.items()}
            else:
                self.index_mapping = {i: i for i in range(len(self.class_names))}
                self.invert_mapping = self.index_mapping

            self._init_gt_recs()

        def __call__(self, det_boxes):
            if self.mAP is not None and len(self.eval_ids) == 0:
                return self.mAP
            self.mAP = self._do_eval(
                det_boxes, self.gt_recs, self.class_names, self.eval_dir
            )
            self._init_gt_recs()
            return self.mAP

        def _init_gt_recs(self):
            self._COCO = COCO()
            self._COCO.dataset["categories"] = [
                {"id": i, "name": name}
                for i, name in enumerate(self.class_names)
                if name != "__background__"
            ]
            self._COCO.dataset.setdefault("images", [])
            self.eval_ids = []

        def target_to_gt_recs(self, targets):
            for info in targets:
                image_id = info["image_id"]
                bboxes = info["ori_boxes"]
                labels = info["labels"]
                anno_ids = info["anno_ids"]
                height, width = info["shape"]
                self.eval_ids += [image_id]
                bboxes = bboxes.cpu().numpy() if hasattr(bboxes, "cpu") else bboxes
                anns = [
                    {
                        "bbox": box,
                        "image_id": image_id,
                        "category_id": self.invert_mapping[
                            int(label) + int(not self.has_background)
                        ],
                        "id": anno_id,
                        "height": int(height),
                        "width": int(width),
                        "iscrowd": 0,
                        "area": float(box[2] * box[3]),
                    }
                    for box, label, anno_id in zip(bboxes, labels, anno_ids)
                ]
                self._COCO.dataset.setdefault("annotations", []).extend(anns)

        def _do_eval(self, det_boxes, gt_recs, class_names, output_dir=None):
            results = []
            for cls_ind, cls in enumerate(self.class_names):
                if cls == "__background__" or cls == "":
                    continue
                results.extend(
                    self._coco_results_one_category(
                        det_boxes[self.index_mapping[cls_ind] - 1], cls_ind
                    )
                )
            ann_type = "bbox"
            self._COCO.dataset["images"] = [
                {"id": image_id} for image_id in self.eval_ids
            ]
            self._COCO.createIndex()
            if len(results) == 0:
                return 0.0
            coco_dt = self._COCO.loadRes(results)
            coco_eval = COCOeval(self._COCO, coco_dt)
            if self.eval_ids is not None and len(self.eval_ids) > 0:
                coco_eval.params.imgIds = sorted(self.eval_ids)
            coco_eval.params.useSegm = ann_type == "segm"
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = coco_eval.stats
            # stats = self._print_detection_eval_metrics(coco_eval)
            self.logger.info(
                ", ".join(
                    [
                        "%s: %.3f" % x
                        for x in zip(
                            ["mAP", "AP50", "AP75", "AP_s", "AP_m", "AP_l"], stats
                        )
                    ]
                )
            )
            if output_dir:
                eval_file = os.path.join(output_dir, "detection_results.pkl")
                with open(eval_file, "wb") as fid:
                    pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
                self.logger.info("Wrote COCO eval results to: {}".format(eval_file))
            return stats[0]

        def _coco_results_one_category(self, boxes, cat_id):
            results = []
            for im_ind in self.eval_ids:
                dets = np.array(boxes.get(im_ind, [])).astype(np.float)
                if len(dets) == 0:
                    continue
                scores = dets[:, -1]
                xs = dets[:, 0]
                ys = dets[:, 1]
                ws = dets[:, 2] - xs
                hs = dets[:, 3] - ys
                results.extend(
                    [
                        {
                            "image_id": im_ind,
                            "category_id": cat_id,
                            "bbox": [xs[k], ys[k], ws[k], hs[k]],
                            "score": scores[k],
                        }
                        for k in range(dets.shape[0])
                    ]
                )
            return results


__all__ = ["VOCMetrics", "COCODetectionMetrics"]


class VOCMetrics(Metrics):
    NAME = "voc"

    def __init__(
        self, has_background=True, class_names=None, eval_dir=None, schedule_cfg=None
    ):
        super(VOCMetrics, self).__init__(schedule_cfg)
        self.class_names = class_names or (
            "__background__",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        )

        self.gt_recs = {}
        self.mAP = None
        self._init_gt_recs()

        self.has_background = has_background

    def _init_gt_recs(self):
        self.gt_recs = {cls_name: {} for cls_name in self.class_names}

    def target_to_gt_recs(self, targets):
        for info in targets:
            image_id = info["image_id"]
            bboxes = info["ori_boxes"]
            labels = info["labels"]
            is_difficult = info["is_difficult"]
            bboxes = bboxes.cpu() if hasattr(bboxes, "cpu") else bboxes
            for bbox, label, is_diff in zip(bboxes, labels, is_difficult):
                class_recs = self.gt_recs[
                    self.class_names[label + int(not self.has_background)]
                ]
                class_recs.setdefault(image_id, {})
                class_recs[image_id].setdefault("bbox", []).append(bbox)
                class_recs[image_id].setdefault("det", []).append(False)
                class_recs[image_id].setdefault("difficult", []).append(is_diff)

    def voc_ap(self, rec, prec, use_07_metric=True):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:True).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.0
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.0], rec, [1.0]))
            mpre = np.concatenate(([0.0], prec, [0.0]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def voc_eval(self, det_recs, gt_recs, class_id, ovthresh=0.5, use_07_metric=True):
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

        npos = 0
        class_recs = gt_recs[self.class_names[class_id + 1]]
        for image_id, objects in class_recs.items():
            difficult = objects["difficult"]
            npos = npos + sum(~np.asarray(difficult).astype(np.bool))

        det_rec = det_recs[class_id]
        image_ids, det_scores, det_bboxes = [], [], []
        for image_id, pred in det_rec.items():
            image_ids += [image_id] * pred.shape[0]
            bboxes = pred[:, :4]
            scores = pred[:, -1]
            det_scores += [scores]
            det_bboxes += [bboxes]

        if len(det_scores) > 0:
            det_scores = np.concatenate(det_scores, 0)
            det_bboxes = np.concatenate(det_bboxes, 0)

            confidence = np.asarray(det_scores).astype(np.float).reshape(-1)
            BB = np.asarray(det_bboxes)

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            # sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs.get(image_ids[d], {"bbox": []})
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                if len(R["bbox"]) > 0:
                    BBGT = np.concatenate(R["bbox"], 0).reshape(-1, 4).astype(float)
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.0)
                    ih = np.maximum(iymax - iymin, 0.0)
                    inters = iw * ih
                    uni = (
                        (bb[2] - bb[0]) * (bb[3] - bb[1])
                        + (BBGT[:, 2] - BBGT[:, 0]) * (BBGT[:, 3] - BBGT[:, 1])
                        - inters
                    )
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R["difficult"][jmax]:
                        if not R["det"][jmax]:
                            tp[d] = 1.0
                            R["det"][jmax] = 1
                        else:
                            fp[d] = 1.0
                else:
                    fp[d] = 1.0

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self.voc_ap(rec, prec, use_07_metric)
        else:
            rec = -1.0
            prec = -1.0
            ap = -1.0

        return rec, prec, ap

    def do_python_eval(self, det_recs, gt_recs, use_07=True):
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = use_07
        self.logger.info("VOC07 metric? " + ("Yes" if use_07_metric else "No"))

        for i, cls_name in enumerate(self.class_names[1:]):
            rec, prec, ap = self.voc_eval(
                det_recs, gt_recs, i, ovthresh=0.5, use_07_metric=use_07_metric
            )
            aps += [ap]
            self.logger.info("AP for {} = {:.4f}".format(cls_name, ap))

        self.logger.info("{:.3f}".format(np.mean(aps)))
        self.logger.info("~~~~~~~~")
        self.logger.info("")
        self.logger.info(
            "--------------------------------------------------------------"
        )
        self.logger.info("Results computed with the **unofficial** Python eval code.")
        self.logger.info(
            "Results should be very close to the official MATLAB eval code."
        )
        self.logger.info(
            "--------------------------------------------------------------"
        )
        return np.mean(aps)

    def __call__(self, det_boxes):
        if self.mAP is not None and all([len(gt) == 0 for gt in self.gt_recs.values()]):
            return self.mAP
        self.mAP = self.do_python_eval(det_boxes, self.gt_recs, True)
        self._init_gt_recs()
        return self.mAP

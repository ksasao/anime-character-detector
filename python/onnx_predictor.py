#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

__all__ = ["YoloxOnnxPredictor"]


def _letterbox(img: np.ndarray, input_size: Sequence[int], swap: Sequence[int] = (2, 0, 1)) -> Tuple[np.ndarray, float]:
    padded = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    ratio = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized = cv2.resize(
        img,
        (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
        interpolation=cv2.INTER_CUBIC,
    ).astype(np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    padded = padded.transpose(swap)
    padded = np.ascontiguousarray(padded, dtype=np.float32)
    return padded, ratio


def _nms(boxes: np.ndarray, scores: np.ndarray, nms_thr: float):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep


def _multiclass_nms(boxes: np.ndarray, scores: np.ndarray, nms_thr: float, score_thr: float):
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]
    mask = cls_scores > score_thr
    if mask.sum() == 0:
        return None
    valid_scores = cls_scores[mask]
    valid_boxes = boxes[mask]
    valid_cls_inds = cls_inds[mask]
    keep = _nms(valid_boxes, valid_scores, nms_thr)
    if not keep:
        return None
    return np.concatenate(
        [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
    )


def _demo_postprocess(outputs: np.ndarray, img_size: Sequence[int], p6: bool = False):
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]
    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]
    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))
    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs


class YoloxOnnxPredictor:
    def __init__(
        self,
        model_path: str,
        input_shape: Sequence[int],
        score_thr: float,
        nms_thr: float,
        class_names: Iterable[str],
        providers: Sequence[str] | None = None,
    ):
        self.input_shape = tuple(int(dim) for dim in input_shape)
        if len(self.input_shape) != 2:
            raise ValueError("input_shape must contain (height, width)")
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.class_names = list(class_names)
        if not self.class_names:
            raise ValueError("class_names must contain at least one entry")
        provider_list = list(providers) if providers else ort.get_available_providers()
        self.session = ort.InferenceSession(model_path, providers=provider_list)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image: np.ndarray):
        if image is None:
            raise ValueError("image must be a valid numpy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be an HxWx3 array")
        blob, ratio = _letterbox(image, self.input_shape)
        ort_inputs = {self.input_name: blob[None, :, :, :]}
        outputs = self.session.run(None, ort_inputs)
        predictions = _demo_postprocess(outputs[0], self.input_shape)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio
        dets = _multiclass_nms(
            boxes_xyxy,
            scores,
            nms_thr=self.nms_thr,
            score_thr=self.score_thr,
        )
        if dets is None:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        final_boxes = dets[:, :4]
        final_scores = dets[:, 4]
        final_cls_inds = dets[:, 5]
        return final_boxes, final_scores, final_cls_inds

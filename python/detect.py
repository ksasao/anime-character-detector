#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import os
from typing import List

import cv2
import numpy as np

from onnx_predictor import YoloxOnnxPredictor


DEFAULT_CLASSES = (
    "character",
)


COLOR_PALETTE = np.array(
    [
        1.000, 0.500, 0.000,
    ]
).astype(np.float32).reshape(-1, 3)


def mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_args():
    parser = argparse.ArgumentParser("YOLOX ONNX image reporter")
    parser.add_argument("image", help="Path to an input image")
    parser.add_argument(
        "--model",
        default="character.onnx",
        help="Path to the exported ONNX model (default: character.onnx)",
    )
    parser.add_argument(
        "--output-dir",
        default="onnx_reports",
        help="Directory where the annotated image and text report are saved",
    )
    parser.add_argument(
        "--input-shape",
        default="640,640",
        help="Model input size as height,width",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="Score threshold for filtering detections",
    )
    parser.add_argument(
        "--nms-thr",
        type=float,
        default=0.45,
        help="IoU threshold used by NMS",
    )
    parser.add_argument(
        "--class-names",
        default=None,
        help="Optional file with custom class names, one per line",
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="Optional list of ONNX Runtime execution providers",
    )
    return parser.parse_args()


def load_class_names(path: str | None) -> List[str]:
    if not path:
        return list(DEFAULT_CLASSES)
    with open(path, "r", encoding="utf-8") as handle:
        names = [line.strip() for line in handle.readlines()]
    names = [name for name in names if name]
    return names if names else list(DEFAULT_CLASSES)


def visualize(img, boxes, scores, cls_ids, conf, class_names):
    for i in range(len(boxes)):
        score = float(scores[i])
        if score < conf:
            continue
        cls_id = int(cls_ids[i])
        x0, y0, x1, y1 = boxes[i].astype(int)
        color = (COLOR_PALETTE[cls_id % len(COLOR_PALETTE)] * 255).astype(np.uint8)
        text = f"{class_names[cls_id]}:{score * 100:.1f}%" if cls_id < len(class_names) else f"cls_{cls_id}:{score * 100:.1f}%"
        txt_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        txt_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color.tolist(), 4)
        txt_bk_color = (color * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1,
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, font_scale, txt_color, font_thickness)
    return img


def save_text_report(path: str, detections: List[dict]):
    header = "label,score,x0,y0,x1,y1"
    rows = [
        f"{det['label']},{det['score']:.4f},{det['box'][0]:.1f},{det['box'][1]:.1f},{det['box'][2]:.1f},{det['box'][3]:.1f}"
        for det in detections
    ]
    content = header + "\n" + "\n".join(rows) + ("\n" if rows else "")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def main():
    args = parse_args()
    class_names = load_class_names(args.class_names)
    input_shape = tuple(map(int, args.input_shape.split(",")))
    predictor = YoloxOnnxPredictor(
        model_path=args.model,
        input_shape=input_shape,
        score_thr=args.score_thr,
        nms_thr=args.nms_thr,
        class_names=class_names,
        providers=args.providers,
    )
    origin_img = cv2.imread(args.image)
    if origin_img is None:
        raise FileNotFoundError(f"Unable to read image: {args.image}")
    boxes, scores, cls_ids = predictor.predict(origin_img)
    annotated = origin_img.copy()
    detections: List[dict] = []
    if len(boxes) > 0:
        annotated = visualize(
            annotated,
            boxes,
            scores,
            cls_ids,
            conf=args.score_thr,
            class_names=class_names,
        )
        for box, score, cls_id in zip(boxes, scores, cls_ids):
            idx = int(cls_id)
            label = class_names[idx] if idx < len(class_names) else f"cls_{idx}"
            detections.append(
                {
                    "label": label,
                    "score": float(score),
                    "box": [float(coord) for coord in box.tolist()],
                }
            )
    mkdir(args.output_dir)
    base = os.path.splitext(os.path.basename(args.image))[0]
    image_path = os.path.join(args.output_dir, f"{base}_annotated.jpg")
    text_path = os.path.join(args.output_dir, f"{base}_detections.txt")
    cv2.imwrite(image_path, annotated)
    save_text_report(text_path, detections)
    if detections:
        for det in detections:
            print(
                f"{det['label']}: {det['score']:.2f} bbox="
                f"({det['box'][0]:.1f},{det['box'][1]:.1f},{det['box'][2]:.1f},{det['box'][3]:.1f})"
            )
    else:
        print("No detections above the score threshold.")
    print(f"Annotated image saved to: {image_path}")
    print(f"Text report saved to: {text_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""装甲板检测评测脚本

用法:
    python3 tools/eval_detector.py --model models/armor_pose_sim.onnx \
        --images ~/datasets/simds/val/images --labels ~/datasets/simds/val/labels

指标: 在 IoU=0.5、置信度阈值 conf 下的 TP/FP/FN、Precision、Recall、F1，
以及单张推理耗时（含预处理与后处理，CPU，与 C++ 部署路径一致）。
标签格式: class cx cy w h  kx1 ky1 ... (归一化, YOLO pose)
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import onnxruntime as ort


def letterbox(img, size=640):
    """等比缩放到 size×size，空白填 114（与训练一致）"""
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    canvas = np.full((size, size, 3), 114, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh))
    return canvas, s, dx, dy


def detect(sess, img, conf_th=0.35, iou_th=0.45, size=640):
    canvas, s, dx, dy = letterbox(img, size)
    x = (canvas[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    scores = out[4]
    keep = np.where(scores >= conf_th)[0]
    if len(keep) == 0:
        return []
    boxes = out[0:4][:, keep].T  # cx cy w h
    rects = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                      boxes[:, 2], boxes[:, 3]], 1)
    idx = cv2.dnn.NMSBoxes(rects.tolist(), scores[keep].tolist(), conf_th, iou_th)
    res = []
    for i in np.array(idx).flatten():
        cx, cy, w, h = boxes[i]
        res.append([(cx - w / 2 - dx) / s, (cy - h / 2 - dy) / s, w / s, h / s,
                    float(scores[keep][i])])
    return res


def load_labels(path, w, h):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        p = line.split()
        if len(p) != 13:
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h])
    return out


def iou(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--conf', type=float, default=0.35)
    args = ap.parse_args()

    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get('ORT_THREADS', '4'))  # 与 C++ 端保持一致
    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                 if os.environ.get('ORT_GPU') else ['CPUExecutionProvider'])
    sess = ort.InferenceSession(args.model, so, providers=providers)
    # 预热：第一次推理包含内存分配与算子初始化，不计入统计
    size = int(sess.get_inputs()[0].shape[2])
    warm = np.zeros((1, 3, size, size), np.float32)
    for _ in range(3):
        sess.run(None, {sess.get_inputs()[0].name: warm})
    files = sorted(glob.glob(os.path.join(args.images, '*.jpg')))
    tp = fp = fn = 0
    times = []
    for f in files:
        img = cv2.imread(f)
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        dets = detect(sess, img, args.conf, size=size)
        times.append((time.perf_counter() - t0) * 1000)
        gts = load_labels(os.path.join(args.labels, os.path.splitext(os.path.basename(f))[0] + '.txt'), w, h)
        used = [False] * len(gts)
        for d in dets:
            best, bi = 0.0, -1
            for i, g in enumerate(gts):
                if used[i]:
                    continue
                v = iou(d, g)
                if v > best:
                    best, bi = v, i
            if best >= 0.5:
                used[bi] = True
                tp += 1
            else:
                fp += 1
        fn += used.count(False)

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    times = np.array(times)
    print(f'images={len(files)}  TP={tp} FP={fp} FN={fn}')
    print(f'Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}')
    print(f'latency mean={times.mean():.1f} ms  p95={np.percentile(times, 95):.1f} ms')


if __name__ == '__main__':
    main()

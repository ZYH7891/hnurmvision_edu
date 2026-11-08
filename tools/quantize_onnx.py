#!/usr/bin/env python3
"""把 ONNX 模型做动态 int8 量化，用于降低 CPU 推理延迟。

用法:
    python3 tools/quantize_onnx.py in.onnx out.onnx
"""
import sys

from onnxruntime.quantization import QuantType, quantize_dynamic


def main():
    src, dst = sys.argv[1], sys.argv[2]
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
    print(f'量化完成: {dst}')


if __name__ == '__main__':
    main()

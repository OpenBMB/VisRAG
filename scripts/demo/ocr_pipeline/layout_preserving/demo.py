#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np
import fastdeploy as fd
from PIL import Image
import io

# ----------------------- 硬编码常量 -----------------------

# 模型文件路径
DET_MODEL_DIR='/mnt/data/user/tc_agi/user/xubokai/mmtrain/document-convert-models-en/en_PP-OCRv3_det_infer'
REC_MODEL_DIR='/mnt/data/user/tc_agi/user/xubokai/mmtrain/document-convert-models-en/en_PP-OCRv3_rec_infer'
CLS_MODEL_DIR='/mnt/data/user/tc_agi/user/xubokai/mmtrain/document-convert-models-en/ch_ppocr_mobile_v2.0_cls_infer'
REC_LABEL_FILE='/mnt/data/user/tc_agi/user/xubokai/mmtrain/document-convert-models-en/en_PP-OCRv3_rec_infer/en_dict.txt'

# 输入图片路径
IMAGE_PATH = "xx"  # Write your image path here

# 推理设备配置
BACKEND = "gpu"  # 可选值："gpu" 或 "cpu"
DEVICE_ID = 0     # 如果使用GPU，请设置GPU设备ID

# 其他参数
MIN_SCORE = 0.6  # 识别得分阈值

# ----------------------- 函数定义 -----------------------

def decode_image(image_path):
    """
    从给定路径解码图像。
    """
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def calculate_spaces_and_newlines(current_box, previous_box, space_threshold=45, line_threshold=15):
    """计算两个文本框之间的空格和换行数量。"""
    spaces = 0
    newlines = 0
    
    # 检查文本框是否在同一行
    if abs(current_box[1] - previous_box[1]) < line_threshold:
        spaces = max(1, int(abs(current_box[0] - previous_box[0]) / space_threshold))
    else:
        newlines = max(1, int(abs(current_box[1] - previous_box[1]) / line_threshold))
    
    return spaces, newlines

def tostr_layout_preserving(result):
    """将OCR结果转换为保留布局的合并字符串。"""
    text_boxes = []
    for box, text, score in zip(result.boxes, result.text, result.rec_scores):
        if score >= MIN_SCORE:  # 仅包含识别得分 >= 0.6 的文本框
            coords = [(box[i], box[i + 1]) for i in range(0, len(box), 2)]
            center_x = (coords[0][0] + coords[2][0]) / 2
            center_y = (coords[0][1] + coords[2][1]) / 2
            text_boxes.append((center_x, center_y, text, coords))

    # 按从上到下、从左到右排序文本框
    text_boxes = sorted(text_boxes, key=lambda x: (x[1], x[0]))
    
    # 合并文本框
    merged_text = []
    previous_box = None
    for box in text_boxes:
        if previous_box is not None:
            spaces, newlines = calculate_spaces_and_newlines(box, previous_box)
            merged_text.append('\n' * newlines + ' ' * spaces)
        merged_text.append(box[2])
        previous_box = box

    res_text = ''.join(merged_text)
    return res_text

def build_option():
    """根据后台和设备构建FastDeploy运行选项。"""
    det_option = fd.RuntimeOption()
    cls_option = fd.RuntimeOption()
    rec_option = fd.RuntimeOption()

    if BACKEND.lower() == "gpu":
        det_option.use_gpu(DEVICE_ID)
        cls_option.use_gpu(DEVICE_ID)
        rec_option.use_gpu(DEVICE_ID)

        # 如果需要使用TensorRT，可以在此处启用
        # det_option.use_trt_backend()
        # cls_option.use_trt_backend()
        # rec_option.use_trt_backend()
    else:
        det_option.use_cpu()
        cls_option.use_cpu()
        rec_option.use_cpu()

    return det_option, cls_option, rec_option

# ----------------------- 主函数 -----------------------

def main():
    # 构建模型文件路径
    det_model_file = os.path.join(DET_MODEL_DIR, "inference.pdmodel")
    det_params_file = os.path.join(DET_MODEL_DIR, "inference.pdiparams")

    cls_model_file = os.path.join(CLS_MODEL_DIR, "inference.pdmodel")
    cls_params_file = os.path.join(CLS_MODEL_DIR, "inference.pdiparams")

    rec_model_file = os.path.join(REC_MODEL_DIR, "inference.pdmodel")
    rec_params_file = os.path.join(REC_MODEL_DIR, "inference.pdiparams")

    # 构建运行选项
    det_option, cls_option, rec_option = build_option()

    # 初始化模型
    det_model = fd.vision.ocr.DBDetector(
        det_model_file, det_params_file, runtime_option=det_option
    )

    cls_model = fd.vision.ocr.Classifier(
        cls_model_file, cls_params_file, runtime_option=cls_option
    )

    rec_model = fd.vision.ocr.Recognizer(
        rec_model_file, rec_params_file, REC_LABEL_FILE, runtime_option=rec_option
    )

    # 设置Det模型的预处理和后处理参数
    det_model.preprocessor.max_side_len = 960
    det_model.postprocessor.det_db_thresh = 0.3
    det_model.postprocessor.det_db_box_thresh = 0.6
    det_model.postprocessor.det_db_unclip_ratio = 1.5
    det_model.postprocessor.det_db_score_mode = "slow"
    det_model.postprocessor.use_dilation = False

    # 设置Cls模型的后处理参数
    cls_model.postprocessor.cls_thresh = 0.9

    # 创建PP-OCRv3实例
    ppocr_v3 = fd.vision.ocr.PPOCRv3(
        det_model=det_model, cls_model=cls_model, rec_model=rec_model
    )

    # 读取并处理图像
    image = decode_image(IMAGE_PATH)
    
    print("-------> 正在进行OCR预测")
    result = ppocr_v3.predict(image)
    print("-------> OCR预测完成")

    # generate the result
    text = tostr_layout_preserving(result)

    print("-------> OCR结果:")
    print(text)
    

if __name__ == "__main__":
    main()

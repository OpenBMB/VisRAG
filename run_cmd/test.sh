#!/bin/bash

echo "====================== CUDA信息检查 ======================"
echo "日期: $(date)"
echo "主机名: $(hostname)"

echo -e "\n---------- NVIDIA驱动和CUDA版本 ----------"
if command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi输出:"
    nvidia-smi
else
    echo "未找到nvidia-smi命令，NVIDIA驱动可能未安装"
fi

echo -e "\n---------- NVCC版本信息 ----------"
if command -v nvcc &> /dev/null; then
    echo "CUDA编译器版本:"
    nvcc --version
else
    echo "未找到nvcc命令，CUDA工具包可能未安装"
fi

echo -e "\n---------- GPU设备详细信息 ----------"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU数量和型号:"
    nvidia-smi --query-gpu=count,name,driver_version --format=csv
    echo -e "\nGPU详细信息:"
    nvidia-smi -q
else
    echo "无法获取GPU详细信息"
fi

echo -e "\n---------- PyTorch CUDA支持 ----------"
echo "执行PyTorch CUDA测试..."
if command -v python3 &> /dev/null; then
    python3 -c "
import torch
print('PyTorch版本:', torch.__version__)
print('CUDA是否可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('当前CUDA版本:', torch.version.cuda)
    print('可用GPU数量:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    # 简单的CUDA测试
    print('运行简单的CUDA测试...')
    try:
        x = torch.rand(5, 3).cuda()
        y = torch.rand(5, 3).cuda()
        z = x + y
        print('CUDA测试成功!')
    except Exception as e:
        print('CUDA测试失败:', str(e))
"
else
    echo "未找到Python3，无法测试PyTorch CUDA支持"
fi

echo -e "\n---------- 环境变量 ----------"
echo "CUDA相关环境变量:"
env | grep -E 'CUDA|NVIDIA'

echo "====================== 检查完成 ======================"
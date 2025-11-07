# Transformer核心组件实现与消融实验

本仓库包含Transformer模型的手工实现、训练代码及消融实验结果，基于IWSLT2017德英翻译数据集验证模型关键组件的有效性。

## 仓库结构
- `src/`：源代码（模型组件、数据处理、训练逻辑）
- `scripts/`：运行脚本
- `data/`：运行数据
- `results/`：训练结果曲线
- `requirements.txt`：依赖库清单

## 硬件要求
- CPU：至少4核（推荐8核）
- GPU：可选（NVIDIA GPU，显存≥4GB，加速训练）
- 内存：≥16GB（处理数据集）

## 环境配置
```bash
# 安装依赖
pip install -r requirements.txt

@echo off
rem Transformer 训练脚本（Windows 版本）
rem 固定随机种子以保证实验可复现

rem 设置随机种子
set SEED=42
set PYTHONHASHSEED=%SEED%
python -c "import torch; torch.manual_seed(%SEED%); print('PyTorch 随机种子已设置为', %SEED%)"

rem 运行训练代码（注意 Windows 路径用反斜杠 \）
python src\train_transformer.py ^
    --data_path ".\data\iwslt2017_de_en" ^
    --seed %SEED%

echo 训练完成！结果保存在 results\ 目录下
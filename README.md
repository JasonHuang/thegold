# 黄金价格预测系统

这是一个基于深度学习的黄金价格预测系统，使用LSTM网络分析历史价格数据，预测未来几天的黄金价格走势。

## 功能特点

- 自动获取最新黄金期货价格数据
- 使用LSTM模型进行时间序列预测
- 自动特征工程和预处理
- 生成可视化预测结果图表
- 模型自动重训练机制
- 预测结果合理性检查

## 安装与使用

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行

#### 快速生成预测图表（不重新训练模型）

```bash
python run.py --user-friendly --future 5
```

#### 强制重新训练模型

```bash
python run.py --force-retrain --future 5
```

#### 使用中文生成图表

```bash
python run.py --user-friendly --chinese
```

#### 使用样例数据（当没有训练好的模型时）

```bash
python run.py --user-friendly --sample
```

## 项目结构

```
thegold/
├── run.py               # 主入口文件
├── gold.py              # 原始单文件版本（已重构）
├── src/                 # 源代码目录
│   ├── data_utils.py    # 数据获取和预处理
│   ├── models.py        # 模型定义和管理
│   ├── evaluation.py    # 模型评估
│   ├── visualization.py # 可视化功能
│   ├── prediction.py    # 预测功能
│   └── main.py          # 主程序逻辑
├── models/              # 保存模型的目录
└── README.md            # 项目说明文档
```

## 参数说明

- `--force-retrain`: 强制重新训练模型
- `--retrain-period`: 模型重新训练周期（天），默认90天
- `--window`: 历史窗口大小（天数），默认60天
- `--future`: 预测未来天数，默认5天
- `--user-friendly`: 只生成面向普通用户的趋势图，不进行完整训练
- `--chinese`: 使用中文生成图表（默认英文）
- `--sample`: 使用样例数据生成趋势图（当没有训练好的模型时）

## 依赖包

主要依赖包：
- tensorflow
- pandas
- numpy
- akshare
- matplotlib
- seaborn
- scikit-learn
- joblib

## 修复说明

最近修复了预测结果过低的问题：
1. 改进了scaler的保存和加载
2. 修复了递归预测中特征更新的逻辑
3. 添加了预测结果合理性检查和纠正机制

#!/bin/bash
set -e  # 遇到错误立即终止（网页7）

# 初始化项目目录
PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
echo "项目路径: $PROJECT_DIR"

# 创建虚拟环境
python3 -m venv $PROJECT_DIR/gold-env
source $PROJECT_DIR/gold-env/bin/activate

# 使用清华源加速安装
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 安装核心依赖
pip install numpy pandas matplotlib scikit-learn

# 可选深度学习框架
read -p "是否安装TensorFlow？(y/n)" choice
if [ $choice == "y" ]; then
  pip install tensorflow
fi

echo "环境配置完成！激活命令：source $PROJECT_DIR/gold-env/bin/activate"
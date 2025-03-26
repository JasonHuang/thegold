#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型模块 - 包含模型构建和训练功能
"""

import os
import logging
import json
from pathlib import Path
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

logger = logging.getLogger(__name__)

def build_model(input_shape, future_days=5, learning_rate=0.001):
    """构建增强版LSTM模型
    
    Args:
        input_shape: 输入数据形状，元组(时间步数, 特征数)
        future_days: 预测未来的天数
        learning_rate: 学习率
        
    Returns:
        编译好的Keras模型
    """
    model = Sequential([
        # 减少复杂度
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dense(future_days)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    
    logger.info(f"模型构建完成: {model.summary()}")
    return model

def build_transformer_model(input_shape, future_days=5, head_size=256, num_heads=4):
    """构建基于Transformer的预测模型"""
    inputs = tf.keras.Input(shape=input_shape)
    
    # 位置编码
    x = inputs
    
    # Transformer编码器层
    x = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads)(x, x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    # 前馈网络
    x = tf.keras.layers.Conv1D(filters=head_size, kernel_size=1, activation='relu')(x)
    x = tf.keras.layers.LayerNormalization()(x)
    
    # 全局池化
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(future_days)(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

def save_model_metadata(model_path, metrics, data_info):
    """保存模型元数据，包括训练日期和性能指标
    
    Args:
        model_path: 模型文件路径
        metrics: 模型评估指标
        data_info: 训练数据信息
        
    Returns:
        元数据文件路径
    """
    metadata = {
        "model_path": model_path,
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "metrics": metrics,
        "data_info": data_info,
        "next_training_date": (datetime.datetime.now() + datetime.timedelta(days=90)).strftime("%Y-%m-%d")
    }
    
    # 创建元数据文件路径
    metadata_path = model_path.replace('.keras', '_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    
    # 同时更新"最新模型"的元数据文件
    model_dir = Path('models')
    latest_metadata_path = model_dir / 'latest_model_metadata.json'
    with open(latest_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    
    logger.info(f"模型元数据已保存到: {metadata_path}")
    return metadata_path

def should_retrain_model(retraining_period_days=90):
    """检查是否需要根据时间间隔重新训练模型
    
    Args:
        retraining_period_days: 重新训练周期（天）
        
    Returns:
        (是否需要重新训练, 原因说明)
    """
    model_dir = Path('models')
    latest_metadata_path = model_dir / 'latest_model_metadata.json'
    
    # 如果没有元数据文件，则需要训练新模型
    if not latest_metadata_path.exists():
        return True, "没有找到现有模型元数据，需要训练新模型"
    
    try:
        # 读取元数据
        with open(latest_metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 获取最后训练日期和计划下次训练日期
        last_training_date = datetime.datetime.strptime(metadata['training_date'], "%Y-%m-%d")
        next_training_date = metadata.get('next_training_date')
        
        if next_training_date:
            next_training_date = datetime.datetime.strptime(next_training_date, "%Y-%m-%d")
        else:
            # 如果没有下次训练日期，则基于最后训练日期和重新训练周期计算
            next_training_date = last_training_date + datetime.timedelta(days=retraining_period_days)
        
        # 检查是否到达重新训练时间
        current_date = datetime.datetime.now()
        if current_date >= next_training_date:
            days_since_last_training = (current_date - last_training_date).days
            return True, f"距离上次训练已经过去 {days_since_last_training} 天，超过了设定的 {retraining_period_days} 天重新训练周期"
        else:
            days_until_next_training = (next_training_date - current_date).days
            return False, f"距离下次计划训练还有 {days_until_next_training} 天，当前模型依然有效"
    
    except Exception as e:
        logger.warning(f"检查模型训练状态时出错: {str(e)}")
        return True, f"检查模型状态时发生错误，建议重新训练: {str(e)}"

def get_latest_model_path():
    """获取最新训练的模型路径
    
    Returns:
        最新模型路径，如果没有则返回None
    """
    model_dir = Path('models')
    latest_metadata_path = model_dir / 'latest_model_metadata.json'
    
    if not latest_metadata_path.exists():
        return None
    
    try:
        with open(latest_metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        model_path = metadata['model_path']
        if os.path.exists(model_path):
            return model_path
    except Exception as e:
        logger.warning(f"获取最新模型路径时出错: {str(e)}")
    
    return None 
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理模块 - 包含数据获取、预处理和特征工程功能
"""

import os
import numpy as np
import pandas as pd
import akshare as ak
import logging
from typing import Optional
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)

def fetch_gold_data(symbol: str = "AU0", 
                   start_date: Optional[str] = None, 
                   end_date: Optional[str] = None) -> pd.DataFrame:
    """获取黄金数据"""
    try:
        logger.info(f"正在获取{symbol}合约数据...")
        df = ak.futures_zh_daily_sina(symbol=symbol)
        df['date'] = pd.to_datetime(df['date'])
        
        # 数据筛选
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
            
        df = df.sort_values('date').set_index('date')
        
        # 基本数据清洗
        df = df.dropna(subset=['close'])
        
        logger.info(f"成功获取数据: {len(df)}行, 日期范围: {df.index.min().strftime('%Y-%m-%d')} 到 {df.index.max().strftime('%Y-%m-%d')}")
        return df[['close']].rename(columns={'close': 'price'})
    except Exception as e:
        logger.error(f"获取数据失败: {str(e)}")
        raise

def engineer_features(df):
    """对原始数据进行特征工程
    
    Args:
        df: 原始价格数据DataFrame
        
    Returns:
        添加特征后的DataFrame
    """
    df_feat = df.copy()
    
    # 保留基本特征
    df_feat['MA5'] = df_feat['price'].rolling(window=5).mean()
    df_feat['MA20'] = df_feat['price'].rolling(window=20).mean()
    df_feat['volatility_20'] = df_feat['price'].rolling(window=20).std()
    df_feat['price_change'] = df_feat['price'].pct_change()
    
    # 移除NaN值
    df_feat = df_feat.dropna()
    
    logger.info(f"完成特征工程，特征数量: {df_feat.shape[1]}")
    return df_feat

def preprocess_data(data, window=60, future=5, test_size=0.2, feature_columns=None):
    """创建时间序列样本"""
    if feature_columns is None:
        feature_columns = data.columns.tolist()
    
    # 选择特征
    features = data[feature_columns]
    
    # 创建专用于价格的scaler
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    price_data = data[['price']].values
    price_scaler.fit(price_data)
    
    # 对所有特征进行归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    
    # 记录特征数量和价格列索引
    n_features = len(feature_columns)
    price_idx = feature_columns.index('price')
    
    # 创建序列数据
    X, y = [], []
    for i in range(window, len(scaled_features) - future + 1):
        # 窗口序列 - 所有特征
        X.append(scaled_features[i-window:i])
        # 未来价格 - 只取价格列
        y.append(scaled_features[i:i+future, price_idx])
    
    X = np.array(X)
    y = np.array(y)
    
    # 输出序列的形状，用于调试
    logger.info(f"X.shape = {X.shape}, y.shape = {y.shape}")
    
    # 划分训练集和测试集
    split = int((1 - test_size) * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 在训练时保存scaler
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(price_scaler, os.path.join(model_dir, 'price_scaler.pkl'))
    logger.info(f"已保存价格scaler到 {os.path.join(model_dir, 'price_scaler.pkl')}")
    
    return X_train, y_train, X_test, y_test, price_scaler

def optimize_dataframe(df):
    """降低DataFrame内存使用"""
    for col in df.select_dtypes(include=['float']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df 
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估模块 - 包含模型评估功能
"""

import numpy as np
import logging
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test, scaler):
    """评估模型性能"""
    # 预测
    predictions = model.predict(X_test)
    
    # 反归一化 - 简化版本
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    pred = scaler.inverse_transform(predictions.reshape(-1, 1))
    
    # 计算评估指标
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    
    logger.info(f"模型评估: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
    
    return actual, pred, {'mae': mae, 'rmse': rmse, 'r2': r2}

def evaluate_forecast(actual, pred):
    """详细评估预测结果"""
    # 基本评估指标
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, pred)
    
    # 添加新指标
    mape = np.mean(np.abs((actual - pred) / actual)) * 100  # 平均绝对百分比误差
    
    # 方向准确性 - 预测价格变动方向是否正确
    actual_dir = np.sign(np.diff(actual.reshape(-1)))
    pred_dir = np.sign(np.diff(pred.reshape(-1)))
    dir_accuracy = np.mean(actual_dir == pred_dir)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'direction_accuracy': dir_accuracy
    }

def k_fold_validation(X, y, build_model_fn, scaler, n_splits=5):
    """K折交叉验证"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 构建并训练模型
        model = build_model_fn((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, 
                  epochs=50, batch_size=32, 
                  validation_data=(X_val, y_val),
                  verbose=0)
        
        # 评估模型
        _, _, metrics = evaluate_model(model, X_val, y_val, scaler)
        fold_metrics.append(metrics)
        logger.info(f"Fold {fold+1}/{n_splits} - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")
    
    # 计算平均指标
    avg_metrics = {
        'mae': np.mean([m['mae'] for m in fold_metrics]),
        'rmse': np.mean([m['rmse'] for m in fold_metrics]),
        'r2': np.mean([m['r2'] for m in fold_metrics])
    }
    
    logger.info(f"K折交叉验证平均结果 - MAE: {avg_metrics['mae']:.2f}, RMSE: {avg_metrics['rmse']:.2f}, R²: {avg_metrics['r2']:.4f}")
    return avg_metrics 
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预测模块 - 包含生成预测的功能
"""

import os
import logging
import numpy as np
import datetime
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

from src.data_utils import fetch_gold_data, engineer_features

logger = logging.getLogger(__name__)

def generate_predictions(days=30):
    """生成未来30天的预测"""
    model_dir = 'models'  # 定义model_dir变量以便后面使用
    try:
        model_path = 'models/gold_prediction_model_latest.keras'
        if not os.path.exists(model_path):
            # 如果最新模型不存在，尝试获取目录中最新的模型
            logger.warning(f"未找到最新模型文件 {model_path}，尝试查找其他可用模型")
            model_dir = Path('models')
            model_files = list(model_dir.glob('gold_prediction_model_*.keras'))
            if model_files:
                # 按修改时间排序，获取最新的模型文件
                model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                model_path = str(model_files[0])
                logger.info(f"使用替代模型: {model_path}")
            else:
                raise FileNotFoundError(f"在models目录中找不到任何可用的模型文件")
        
        model = load_model(model_path)
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise
    
    # 获取最新数据
    df = fetch_gold_data()
    df_features = engineer_features(df)
    
    # 准备预测输入 - 最新的窗口
    window = 60  # 添加默认窗口大小
    latest_data = df_features.iloc[-window:].values

    # 尝试加载训练时保存的scaler
    try:
        price_scaler = joblib.load(os.path.join(model_dir, 'price_scaler.pkl'))
        logger.info("成功加载训练时保存的price_scaler")
    except Exception as e:
        logger.warning(f"无法加载训练时的scaler: {str(e)}，将创建新的scaler")
        # 如果找不到保存的scaler，创建新的但基于全部历史数据
        price_scaler = MinMaxScaler()
        price_scaler.fit(df_features[['price']].values)
    
    # 记录最后一个实际价格，用于验证预测结果的合理性
    last_price = df['price'].iloc[-1]
    logger.info(f"最新实际金价: {last_price:.2f}")
    
    # 计算价格波动限制 - 基于历史数据
    # 获取过去30天的最大日波动百分比
    recent_df = df.iloc[-30:]
    daily_changes = recent_df['price'].pct_change().abs()
    max_daily_change = max(daily_changes.dropna()) * 1.5  # 放宽50%
    avg_daily_change = daily_changes.mean() * 2  # 平均变化的两倍
    
    # 限制最大变化，一般不超过3%
    max_allowed_change = min(max(max_daily_change, avg_daily_change), 0.03)
    logger.info(f"基于历史数据的最大允许日变化率: {max_allowed_change:.2%}")
    
    # 对所有特征进行归一化
    features_scaler = MinMaxScaler()
    latest_scaled = features_scaler.fit_transform(latest_data)
    X_pred = latest_scaled.reshape(1, window, df_features.shape[1])
    
    # 递归预测
    predictions = []
    current_sequence = X_pred.copy()
    
    # 存储最后一个时间步的原始特征值，用于更新派生特征
    last_features = df_features.iloc[-1].copy()
    
    # 存储前一天的预测价格，用于限制日变化幅度
    prev_price = last_price
    
    for i in range(days):
        # 预测下一个值
        next_pred = model.predict(current_sequence, verbose=0)[0]
        
        # 转换为实际价格
        next_price_raw = price_scaler.inverse_transform([[next_pred[0]]])[0][0]
        
        # 计算与前一天的价格变化百分比
        price_change = (next_price_raw - prev_price) / prev_price
        
        # 如果变化超过限制，则将其限制在合理范围内
        if abs(price_change) > max_allowed_change:
            # 保持变化方向，但限制幅度
            direction = 1 if price_change > 0 else -1
            limited_change = direction * max_allowed_change
            next_price_adjusted = prev_price * (1 + limited_change)
            logger.debug(f"限制价格变化: 原始{price_change:.2%} -> 调整后{limited_change:.2%}")
            
            # 转换回归一化值
            next_pred_adjusted = price_scaler.transform([[next_price_adjusted]])[0][0]
            next_pred[0] = next_pred_adjusted
        else:
            next_price_adjusted = next_price_raw
        
        predictions.append(next_pred[0])
        prev_price = next_price_adjusted
        
        # 更新序列 - 为所有特征创建合理的值
        next_input = np.zeros((1, 1, df_features.shape[1]))
        
        # 预测的价格是第一个特征
        next_price = next_pred[0]
        next_input[0, 0, 0] = next_price
        
        # 更新last_features以包含新预测的价格
        last_features['price'] = price_scaler.inverse_transform([[next_price]])[0][0]
        
        # 如果需要，可以在这里更新其他派生特征
        # 例如：更新移动平均线(简化计算)
        if i > 0:
            # 计算更新的MA5和MA20值
            if i < 5:
                # 还没有5天数据时，MA5使用可用数据的平均值
                predicted_prices = [df['price'].iloc[-5+j] for j in range(min(5, i))] + \
                               [price_scaler.inverse_transform([[predictions[j]]])[0][0] for j in range(i)]
                ma5 = sum(predicted_prices) / len(predicted_prices)
            else:
                # 有5天以上数据时，使用最近5天预测的平均值
                ma5 = sum([price_scaler.inverse_transform([[predictions[j]]])[0][0] for j in range(i-4, i+1)]) / 5
            
            # 更新MA5
            last_features['MA5'] = ma5
        
        # 归一化新特征
        next_feature_values = features_scaler.transform([last_features.values])[0]
        
        # 将归一化后的特征值放入next_input
        for j in range(df_features.shape[1]):
            next_input[0, 0, j] = next_feature_values[j]
        
        # 滚动窗口
        current_sequence = np.append(current_sequence[:, 1:, :], 
                                    next_input, axis=1)
    
    # 反归一化预测结果
    predictions = price_scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1))
    
    # 创建日期索引 - 从最新数据的下一天开始
    last_date = df.index[-1]
    future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(days)]
    
    # 实施全局合理性检查
    # 1. 检查预测结果是否整体偏低
    if predictions[0][0] < last_price * 0.97:  # 首日预测不低于当前价格的97%
        logger.warning(f"首日预测偏低: 当前价格 {last_price:.2f}，首日预测 {predictions[0][0]:.2f}")
        logger.info("应用首日修正因子")
        # 计算合理的首日预测值，略低于当前价格
        target_first_day = last_price * 0.99
        correction_factor = target_first_day / predictions[0][0]
        predictions = predictions * correction_factor
        logger.info(f"修正后的首日预测: {predictions[0][0]:.2f}")
    
    # 2. 检查整个预测序列的波动是否合理
    max_drop = (predictions.min() - last_price) / last_price
    if max_drop < -0.05:  # 如果预测未来有超过5%的下跌
        logger.warning(f"预测序列整体偏低: 最大跌幅 {max_drop:.2%}")
        
        # 应用渐进式修正，保持趋势但减小幅度
        min_pred = predictions.min()
        # 确保最低点不低于当前价格的95%
        min_threshold = last_price * 0.95
        
        if min_pred < min_threshold:
            # 计算需要提升的幅度
            lift_factor = (min_threshold - min_pred) / (last_price - min_pred)
            
            # 对每个预测值应用渐进式修正
            for i in range(len(predictions)):
                # 降低幅度但保持趋势
                if predictions[i][0] < last_price:
                    drop = last_price - predictions[i][0]
                    predictions[i][0] = last_price - drop * (1 - lift_factor)
            
            logger.info(f"应用渐进式修正后的价格范围: {predictions.min():.2f}-{predictions.max():.2f}")
    
    # 3. 检查是否存在不合理的剧烈波动
    for i in range(1, len(predictions)):
        prev_price = predictions[i-1][0]
        curr_price = predictions[i][0]
        daily_change = (curr_price - prev_price) / prev_price
        
        if abs(daily_change) > 0.02:  # 单日波动不超过2%
            # 平滑异常波动
            smooth_price = prev_price * (1 + (0.02 if daily_change > 0 else -0.02))
            predictions[i][0] = smooth_price
            logger.debug(f"平滑第{i+1}天的异常波动: {daily_change:.2%} -> {(smooth_price-prev_price)/prev_price:.2%}")
    
    logger.info(f"生成了未来{days}天的预测，价格范围: {predictions.min():.2f}-{predictions.max():.2f}")
    return predictions, future_dates, df

def generate_sample_predictions(days=5):
    """生成样例预测数据，用于没有模型时的演示"""
    df = fetch_gold_data()
    last_date = df.index[-1]
    future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(days)]
    
    # 生成一些模拟的预测结果 - 以最后一个价格为基准，添加一些随机波动
    last_price = df['price'].iloc[-1]
    # 使用随机种子确保每次生成相同结果
    np.random.seed(42)
    # 生成一个趋势增长或下降的序列，再加上随机波动
    trend = np.linspace(0, 0.05, days)  # 生成0到5%的线性增长
    noise = np.random.normal(0, 0.01, days)  # 添加1%左右的随机波动
    predictions = last_price * (1 + trend + noise)
    
    logger.warning("生成了样例预测数据，仅供参考")
    return predictions, future_dates, df 
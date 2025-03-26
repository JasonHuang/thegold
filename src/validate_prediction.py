#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证模块 - 用于对比预测结果和实际价格
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

def load_predictions():
    """加载预测结果"""
    try:
        with open('predictions.json', 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        return predictions
    except FileNotFoundError:
        logger.error("未找到预测结果文件 predictions.json")
        return None

def fetch_actual_prices():
    """获取实际价格数据"""
    from src.data_utils import fetch_gold_data
    df = fetch_gold_data()
    return df

def validate_predictions():
    """验证预测结果"""
    # 加载预测结果
    predictions = load_predictions()
    if not predictions:
        return
    
    # 获取实际价格
    df = fetch_actual_prices()
    
    # 获取预测日期
    prediction_dates = [datetime.strptime(date, '%Y-%m-%d') for date in predictions['dates']]
    predicted_prices = predictions['prices']
    
    # 获取实际价格
    actual_prices = []
    actual_dates = []
    
    for date in prediction_dates:
        if date in df.index:
            actual_prices.append(df.loc[date, 'price'])
            actual_dates.append(date)
    
    if not actual_prices:
        logger.warning("还没有实际价格数据可供对比")
        return
    
    # 计算误差
    errors = []
    for pred_price, act_price in zip(predicted_prices[:len(actual_prices)], actual_prices):
        error = (pred_price - act_price) / act_price * 100
        errors.append(error)
    
    # 输出结果
    logger.info("\n=== 预测验证结果 ===")
    logger.info(f"预测日期: {prediction_dates[0].strftime('%Y-%m-%d')}")
    logger.info(f"预测价格: {predicted_prices[0]:.2f}")
    logger.info(f"实际价格: {actual_prices[0]:.2f}")
    logger.info(f"预测误差: {errors[0]:.2f}%")
    
    if len(errors) > 1:
        logger.info("\n=== 5天预测统计 ===")
        logger.info(f"平均误差: {sum(errors) / len(errors):.2f}%")
        logger.info(f"最大误差: {max(errors):.2f}%")
        logger.info(f"最小误差: {min(errors):.2f}%")
    
    # 保存验证结果
    validation_results = {
        'validation_date': datetime.now().strftime('%Y-%m-%d'),
        'prediction_dates': [d.strftime('%Y-%m-%d') for d in prediction_dates],
        'predicted_prices': predicted_prices,
        'actual_dates': [d.strftime('%Y-%m-%d') for d in actual_dates],
        'actual_prices': actual_prices,
        'errors': errors
    }
    
    with open('validation_results.json', 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, ensure_ascii=False, indent=2)
    
    logger.info("\n验证结果已保存到 validation_results.json")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    validate_predictions() 
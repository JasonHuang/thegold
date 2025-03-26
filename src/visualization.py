#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化模块 - 包含所有绘图功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import datetime
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

# 配置中文字体支持
try:
    # 尝试设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS'] 
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
except Exception as e:
    logger.warning(f"设置中文字体失败: {e}，图表中的中文可能无法正确显示")

def plot_results(actual, pred, history=None, predictions_days=5, use_english=False):
    """可视化技术评估结果
    
    Args:
        actual: 实际值
        pred: 预测值
        history: 训练历史记录
        predictions_days: 预测的天数
        use_english: 是否使用英文标题
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 创建一个2x2的子图布局
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    
    # 设置标题语言
    if use_english:
        title1 = f"Gold Price Prediction (Next {predictions_days} days, MAE: {mean_absolute_error(actual, pred):.2f})"
        title2 = "Predicted vs Actual Values"
        title3 = "Prediction Error Distribution"
        title4 = "Training and Validation Loss"
        xlabel1 = "Samples"
        ylabel1 = "Price (CNY/g)"
        xlabel2 = "Actual Price"
        ylabel2 = "Predicted Price"
        xlabel3 = "Prediction Error"
        ylabel3 = "Frequency"
        xlabel4 = "Epoch"
        ylabel4 = "Loss"
        legend1 = ["Actual Price", "Predicted Price"]
        legend2 = ["Training Loss", "Validation Loss"]
    else:
        title1 = f"黄金价格预测 (未来{predictions_days}天 MAE: {mean_absolute_error(actual, pred):.2f}元)"
        title2 = "预测值 vs 实际值"
        title3 = "预测误差分布"
        title4 = "训练和验证损失"
        xlabel1 = "样本"
        ylabel1 = "价格 (元/克)"
        xlabel2 = "实际价格"
        ylabel2 = "预测价格"
        xlabel3 = "预测误差"
        xlabel4 = "Epoch"
        ylabel4 = "损失"
        legend1 = ["实际价格", "预测价格"]
        legend2 = ["训练损失", "验证损失"]
    
    # 1. 价格预测对比图
    axs[0, 0].plot(actual, label=legend1[0], alpha=0.7, linewidth=2)
    axs[0, 0].plot(pred, label=legend1[1], linestyle='--', linewidth=2)
    axs[0, 0].set_title(title1, fontsize=15)
    axs[0, 0].set_xlabel(xlabel1, fontsize=12)
    axs[0, 0].set_ylabel(ylabel1, fontsize=12)
    axs[0, 0].legend(fontsize=12)
    axs[0, 0].grid(True)
    
    # 2. 价格预测散点图
    axs[0, 1].scatter(actual, pred, alpha=0.5)
    min_val = min(actual.min(), pred.min())
    max_val = max(actual.max(), pred.max())
    axs[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
    axs[0, 1].set_title(title2, fontsize=15)
    axs[0, 1].set_xlabel(xlabel2, fontsize=12)
    axs[0, 1].set_ylabel(ylabel2, fontsize=12)
    
    # 3. 价格误差直方图
    error = actual - pred
    sns.histplot(error, bins=50, kde=True, ax=axs[1, 0])
    axs[1, 0].set_title(title3, fontsize=15)
    axs[1, 0].set_xlabel(xlabel3, fontsize=12)
    axs[1, 0].set_ylabel(ylabel3, fontsize=12)
    
    # 4. 训练和验证损失曲线 (如果提供了history)
    if history:
        axs[1, 1].plot(history.history['loss'], label=legend2[0])
        axs[1, 1].plot(history.history['val_loss'], label=legend2[1])
        axs[1, 1].set_title(title4, fontsize=15)
        axs[1, 1].set_xlabel(xlabel4, fontsize=12)
        axs[1, 1].set_ylabel(ylabel4, fontsize=12)
        axs[1, 1].legend(fontsize=12)
    else:
        axs[1, 1].set_visible(False)
    
    plt.tight_layout()
    language = "en" if use_english else "cn"
    plt.savefig(f'gold_price_prediction_results_{language}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_future_trend(predictions, future_dates, df, future_days=5, use_english=False, use_sample_data=False):
    """创建更直观的未来金价走势图
    
    Args:
        predictions: 预测值数组
        future_dates: 未来日期数组
        df: 历史数据DataFrame
        future_days: 预测的未来天数
        use_english: 是否使用英文
        use_sample_data: 是否为样例数据
    """
    # 获取历史数据作为参考
    recent_days = 30  # 显示最近30天的历史数据
    historical = df['price'].iloc[-recent_days:].copy()
    
    # 设置字体和样式
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(14, 8))
    
    # 设置所有图表为英文，避免中文字体问题
    title = "Gold Price Forecast (Next 5 Days)"
    subtitle = f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d')}"
    ylabel = "Price (CNY/gram)"
    trend_label = "Price Trend"
    history_label = "Historical Price"
    prediction_label = "Predicted Price"
    today_label = "Today"
    note_text = "Note: This forecast is based on historical data patterns and may vary due to market conditions."
    
    # 分析预测趋势
    if predictions[-1] > predictions[0]:
        trend_text = f"↑ Upward Trend: Price expected to increase by {predictions[-1]-predictions[0]:.2f} CNY/gram ({(predictions[-1]/predictions[0]-1)*100:.1f}%)"
        trend_color = 'green'
    elif predictions[-1] < predictions[0]:
        trend_text = f"↓ Downward Trend: Price expected to decrease by {predictions[0]-predictions[-1]:.2f} CNY/gram ({(1-predictions[-1]/predictions[0])*100:.1f}%)"
        trend_color = 'red'
    else:
        trend_text = "→ Stable Price: No significant change expected"
        trend_color = 'blue'
    
    # 添加样例数据警告
    if use_sample_data:
        sample_warning = "WARNING: SAMPLE DATA - FOR DEMONSTRATION ONLY"
    
    # 绘制历史数据
    plt.plot(historical.index, historical.values, 
             color='gray', alpha=0.7, linewidth=2, label=history_label)
    
    # 在图表上标记"今天"
    plt.axvline(x=df.index[-1], color='black', linestyle='--', alpha=0.7)
    plt.text(df.index[-1], historical.min() * 0.98, today_label, 
             ha='center', va='top', rotation=90, fontsize=10)
    
    # 绘制预测数据
    prediction_line = plt.plot(future_dates, predictions, 
                              color=trend_color, marker='o', markersize=8, 
                              linewidth=3, label=prediction_label)[0]
    
    # 填充预测区域，增强视觉效果
    plt.fill_between(future_dates, predictions, 
                     df['price'].iloc[-1], alpha=0.2, color=trend_color)
    
    # 为每个预测点添加价格标签
    for i, (date, price) in enumerate(zip(future_dates, predictions)):
        plt.annotate(f'{price:.1f}', (date, price), 
                     textcoords="offset points", 
                     xytext=(0,10), ha='center',
                     fontweight='bold', fontsize=12)
    
    # 添加趋势指示文本框
    plt.figtext(0.5, 0.01, trend_text, 
               ha='center', fontsize=14, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # 添加样例数据警告
    if use_sample_data:
        plt.figtext(0.5, 0.95, sample_warning, 
                   ha='center', fontsize=16, fontweight='bold', color='red',
                   bbox=dict(facecolor='yellow', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # 添加注释说明
    plt.figtext(0.5, -0.02, note_text, ha='center', fontsize=10, style='italic')
    
    # 设置图表标题和标签
    plt.title(title, fontsize=18, fontweight='bold')
    plt.suptitle(subtitle, fontsize=10, y=0.92)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 格式化x轴日期
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    
    # 调整y轴范围，确保有足够的空间显示注释
    y_min = min(historical.min(), predictions.min()) * 0.98
    y_max = max(historical.max(), predictions.max()) * 1.02
    plt.ylim(y_min, y_max)
    
    # 添加图例
    plt.legend(loc='upper left')
    
    # 确保布局正确
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    # 保存图表 - 无论是否use_english，都用英文保存
    language = "en"
    plt.savefig(f'gold_future_trend_{language}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_price_trend(df):
    """绘制原始数据价格走势图"""
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['price'])
    plt.title('黄金价格走势')
    plt.grid(True)
    plt.savefig('gold_price_trend.png')
    plt.close()

def plot_correlation_matrix(df_features):
    """绘制特征相关性矩阵"""
    corr = df_features.corr()
    plt.figure(figsize=(14,10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.savefig('feature_correlation.png')
    plt.close() 
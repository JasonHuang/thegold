#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
黄金价格预测系统 - 主入口文件
"""

import sys
import os
import json
import logging
import argparse
from datetime import datetime
from src.main import main, generate_user_friendly_chart
from src.data_utils import fetch_gold_data, engineer_features
from src.prediction import generate_predictions, generate_sample_predictions
from src.visualization import plot_predictions

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_predictions(predictions, dates):
    """保存预测结果到JSON文件"""
    predictions_data = {
        'prediction_date': datetime.now().strftime('%Y-%m-%d'),
        'dates': [date.strftime('%Y-%m-%d') for date in dates],
        'prices': [float(price) for price in predictions]
    }
    
    with open('predictions.json', 'w', encoding='utf-8') as f:
        json.dump(predictions_data, f, ensure_ascii=False, indent=2)
    
    logger.info("预测结果已保存到 predictions.json")

if __name__ == "__main__":
    # 添加当前目录到Python路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # 导入参数解析
    parser = argparse.ArgumentParser(description='黄金价格预测系统')
    parser.add_argument('--force-retrain', action='store_true', help='强制重新训练模型')
    parser.add_argument('--retrain-period', type=int, default=90, help='模型重新训练周期（天），默认90天')
    parser.add_argument('--window', type=int, default=60, help='历史窗口大小')
    parser.add_argument('--future', type=int, default=5, help='预测未来天数')
    parser.add_argument('--user-friendly', action='store_true', help='只生成面向普通用户的趋势图')
    parser.add_argument('--chinese', action='store_true', help='使用中文生成图表')
    parser.add_argument('--sample', action='store_true', help='使用样例数据生成趋势图（当没有训练好的模型时）')
    
    args = parser.parse_args()
    
    # 如果只需要生成用户友好的趋势图，则不进行完整的训练过程
    if args.user_friendly:
        use_english = not args.chinese
        try:
            # 获取数据
            logger.info("正在获取黄金价格数据...")
            df = fetch_gold_data()
            
            # 生成预测
            logger.info("正在生成预测...")
            predictions, future_dates, _ = generate_predictions(days=args.future)
            
            # 保存预测结果
            save_predictions(predictions.flatten(), future_dates)
            
            # 生成可视化
            logger.info("正在生成预测图表...")
            plot_predictions(predictions.flatten(), future_dates, df, future_days=args.future, use_english=use_english)
            
            # 输出预测结果
            logger.info("\n=== 预测结果 ===")
            for date, price in zip(future_dates, predictions.flatten()):
                logger.info(f"{date.strftime('%Y-%m-%d')}: {price:.2f}")
            
            logger.info("\n预测完成！预测图表已保存为 gold_future_trend_en.png")
            
        except Exception as e:
            logger.error(f"运行过程中出现错误: {str(e)}")
            if not args.sample:
                raise
            
            # 使用样例数据
            predictions, future_dates, df = generate_sample_predictions(args.future)
            plot_predictions(predictions, future_dates, df, future_days=args.future, use_english=use_english, use_sample_data=True)
    else:
        main(
            window=args.window,
            future_days=args.future,
            epochs=100,
            batch_size=32,
            force_retrain=args.force_retrain,
            retraining_period_days=args.retrain_period
        ) 
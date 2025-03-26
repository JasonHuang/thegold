#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主程序模块 - 包含主函数和程序入口点
"""

import os
import logging
import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from src.data_utils import fetch_gold_data, engineer_features, preprocess_data
from src.models import build_model, save_model_metadata, should_retrain_model, get_latest_model_path
from src.evaluation import evaluate_model
from src.visualization import plot_results, plot_future_trend, plot_price_trend, plot_correlation_matrix
from src.prediction import generate_predictions, generate_sample_predictions

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main(window=60, future_days=5, epochs=100, batch_size=32, force_retrain=False, retraining_period_days=90):
    """主函数
    
    Args:
        window: 时间窗口大小
        future_days: 预测未来的天数
        epochs: 训练轮数
        batch_size: 批量大小
        force_retrain: 是否强制重新训练模型
        retraining_period_days: 模型重新训练周期（天）
    """
    try:
        # 创建模型保存目录
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        
        # 检查是否需要重新训练模型
        should_train, reason = should_retrain_model(retraining_period_days)
        
        # 如果强制重新训练或者应该重新训练
        if force_retrain or should_train:
            logger.info(f"将重新训练模型: {reason}")
            
            # 获取当前时间，用于模型文件命名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(model_dir, f'gold_prediction_model_{timestamp}.keras')
            
            # 获取数据
            df = fetch_gold_data()
            print(f"最新数据日期：{df.index[-1].strftime('%Y-%m-%d')}")
            
            # 数据质量检查
            print(f"数据统计: \n{df.describe()}")
            
            # 绘制原始数据图表
            plot_price_trend(df)
            
            # 特征工程和相关性分析
            df_features = engineer_features(df)
            plot_correlation_matrix(df_features)
            
            # 数据预处理
            X_train, y_train, X_test, y_test, scaler = preprocess_data(
                df_features, 
                window=window, 
                future=future_days,
                test_size=0.2
            )
            
            # 学习率余弦退火
            from tensorflow.keras.optimizers.schedules import CosineDecay
            lr_schedule = CosineDecay(
                initial_learning_rate=0.001,
                decay_steps=epochs * (len(X_train) // batch_size),
                alpha=0.0001
            )
            
            # 构建模型
            model = build_model((X_train.shape[1], X_train.shape[2]), future_days)
            
            # 创建回调函数
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
            ]
            
            # 模型训练
            logger.info("开始训练模型...")
            history = model.fit(
                X_train, y_train,
                epochs=50,  # 减少初始epochs
                batch_size=16,  # 减少批量大小
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            # 评估模型
            logger.info("评估模型性能...")
            actual, pred, metrics = evaluate_model(model, X_test, y_test, scaler)
            
            # 打印评估指标
            print(f"模型评估指标:")
            print(f"MAE: {metrics['mae']:.2f} 元/克")
            print(f"RMSE: {metrics['rmse']:.2f} 元/克")
            print(f"R²: {metrics['r2']:.2%}")
            
            # 可视化结果 - 技术性图表
            logger.info("生成技术指标可视化图表...")
            plot_results(actual, pred, history, future_days, use_english=True)
            
            # 添加普通用户友好的趋势图表
            logger.info("生成面向普通用户的未来趋势图...")
            predictions, future_dates, df = generate_predictions(future_days)
            plot_future_trend(predictions.flatten(), future_dates, df, future_days, use_english=True)
            
            # 保存模型使用新格式
            model.save(model_path)
            logger.info(f"模型已保存到: {model_path}")
            
            # 同时保存一个latest模型副本，方便后续加载
            latest_model_path = os.path.join(model_dir, 'gold_prediction_model_latest.keras')
            model.save(latest_model_path)
            logger.info(f"同时保存了最新模型副本: {latest_model_path}")
            
            # 保存模型元数据
            data_info = {
                "data_range": {
                    "start": df.index.min().strftime('%Y-%m-%d'),
                    "end": df.index.max().strftime('%Y-%m-%d')
                },
                "data_count": len(df),
                "feature_count": df_features.shape[1]
            }
            save_model_metadata(model_path, metrics, data_info)
            
            return model, scaler, metrics
        
        else:
            # 加载现有模型进行预测
            logger.info(f"使用现有模型: {reason}")
            model_path = get_latest_model_path()
            
            if not model_path:
                logger.warning("无法找到现有模型，将重新训练")
                return main(window, future_days, epochs, batch_size, force_retrain=True)
            
            # 加载模型和元数据
            from tensorflow.keras.models import load_model
            import json
            
            model = load_model(model_path)
            
            with open(Path('models') / 'latest_model_metadata.json', 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            logger.info(f"加载现有模型成功，上次训练日期: {metadata['training_date']}")
            print(f"现有模型评估指标:")
            print(f"MAE: {metadata['metrics']['mae']:.2f} 元/克")
            print(f"RMSE: {metadata['metrics']['rmse']:.2f} 元/克")
            print(f"R²: {metadata['metrics']['r2']:.2%}")
            
            # 生成面向普通用户的未来趋势图
            logger.info("生成面向普通用户的未来趋势图...")
            predictions, future_dates, df = generate_predictions(future_days)
            plot_future_trend(predictions.flatten(), future_dates, df, future_days, use_english=True)
            
            return model, None, metadata['metrics']
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise

def generate_user_friendly_chart(future_days=5, use_english=True, use_sample_data=False):
    """生成用户友好的预测趋势图，无需进行完整训练"""
    try:
        # 尝试使用模型生成预测
        predictions, future_dates, df = generate_predictions(future_days)
        plot_future_trend(predictions.flatten(), future_dates, df, future_days, use_english=use_english)
    except Exception as e:
        logger.warning(f"使用模型预测失败: {str(e)}")
        if not use_sample_data:
            raise
        
        # 使用样例数据
        predictions, future_dates, df = generate_sample_predictions(future_days)
        plot_future_trend(predictions, future_dates, df, future_days, use_english=use_english, use_sample_data=True)

if __name__ == "__main__":
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
        generate_user_friendly_chart(future_days=args.future, use_english=use_english, use_sample_data=args.sample)
    else:
        main(
            window=args.window,
            future_days=args.future,
            epochs=100,
            batch_size=32,
            force_retrain=args.force_retrain,
            retraining_period_days=args.retrain_period
        ) 
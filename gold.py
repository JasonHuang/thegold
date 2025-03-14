# -*- coding: utf-8 -*-
# 黄金价格预测完整示例（2025年3月更新版）
import os
import numpy as np
import pandas as pd
import akshare as ak
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import logging
import datetime
import matplotlib.font_manager as fm
import tensorflow as tf
from sklearn.model_selection import KFold, TimeSeriesSplit
from typing import Optional, Dict, Any, Tuple
import json
import time
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 配置中文字体支持
try:
    # 尝试设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']  # 优先使用的字体列表
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    # 检查系统是否有可用的中文字体
    fonts = [f.name for f in fm.fontManager.ttflist]
    logger.info(f"可用字体: {[f for f in fonts if '黑' in f or 'Hei' in f or 'sans' in f.lower()][:5]}")
except Exception as e:
    logger.warning(f"设置中文字体失败: {e}，图表中的中文可能无法正确显示")

# 数据获取模块
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

# 数据预处理与特征工程模块
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

# 数据预处理模块
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
    
    return X_train, y_train, X_test, y_test, price_scaler

# 模型构建模块
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

# 评估模块
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

# 可视化模块
def plot_results(actual, pred, history=None, predictions_days=5, use_english=False):
    """可视化结果
    
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
    plt.show()

# 保存模型元数据函数
def save_model_metadata(model_path: str, metrics: Dict[str, Any], data_info: Dict[str, Any]) -> str:
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
    
    # 同时更新"最新模型"的元数据链接
    latest_metadata_path = os.path.join(os.path.dirname(model_path), 'latest_model_metadata.json')
    with open(latest_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    
    logger.info(f"模型元数据已保存到: {metadata_path}")
    return metadata_path

# 检查是否需要重新训练模型
def should_retrain_model(retraining_period_days: int = 90) -> Tuple[bool, str]:
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

# 获取最新模型路径
def get_latest_model_path() -> Optional[str]:
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

# 主程序
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
            
            # 绘制原始数据图表，检查异常值和趋势
            plt.figure(figsize=(12,6))
            plt.plot(df.index, df['price'])
            plt.title('黄金价格走势')
            plt.grid(True)
            plt.savefig('gold_price_trend.png')
            plt.close()
            
            # 特征相关性分析
            df_features = engineer_features(df)
            corr = df_features.corr()
            plt.figure(figsize=(14,10))
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.savefig('feature_correlation.png')
            plt.close()
            
            # 数据预处理
            X_train, y_train, X_test, y_test, scaler = preprocess_data(
                df_features, 
                window=window, 
                future=future_days,
                test_size=0.2
            )
            
            # 学习率余弦退火
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=0.001,
                decay_steps=epochs * (len(X_train) // batch_size),
                alpha=0.0001
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            
            # 构建模型
            model = build_model((X_train.shape[1], X_train.shape[2]), future_days)
            
            # 创建回调函数
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
            plot_future_trend(future_days=future_days, use_english=True)
            
            # 保存模型使用新格式
            model.save(model_path)
            logger.info(f"模型已保存到: {model_path}")
            
            # 同时保存一个latest模型副本，方便后续加载
            latest_model_path = os.path.join(model_dir, 'gold_prediction_model_latest.keras')
            model.save(latest_model_path)
            
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
            model = load_model(model_path)
            
            with open(Path('models') / 'latest_model_metadata.json', 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 获取最新的一些数据用于测试和预测
            df = fetch_gold_data()
            df_features = engineer_features(df)
            
            logger.info(f"加载现有模型成功，上次训练日期: {metadata['training_date']}")
            print(f"现有模型评估指标:")
            print(f"MAE: {metadata['metrics']['mae']:.2f} 元/克")
            print(f"RMSE: {metadata['metrics']['rmse']:.2f} 元/克")
            print(f"R²: {metadata['metrics']['r2']:.2%}")
            
            # 生成面向普通用户的未来趋势图
            logger.info("生成面向普通用户的未来趋势图...")
            plot_future_trend(future_days=future_days, use_english=True)
            
            # 如果需要，可以添加对模型进行简单评估的代码
            
            return model, None, metadata['metrics']
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise

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

def k_fold_validation(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = build_model((X_train.shape[1], X_train.shape[2]))
        # 训练模型
        history = model.fit(X_train, y_train, 
                          epochs=50, batch_size=32, 
                          validation_data=(X_val, y_val),
                          callbacks=[EarlyStopping(patience=10)],
                          verbose=0)
        # 评估模型
        metrics = evaluate_model(model, X_val, y_val, scaler)
        fold_metrics.append(metrics)
    
    return fold_metrics

def evaluate_forecast(actual, pred):
    # 已有评估指标
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

def analyze_feature_importance(model, X, feature_names):
    # 排列重要性
    perm_importance = []
    baseline_pred = model.predict(X)
    baseline_mae = mean_absolute_error(scaler.inverse_transform(y_test), 
                                       scaler.inverse_transform(baseline_pred))
    
    for i, feature in enumerate(feature_names):
        # 打乱特征值
        X_permuted = X.copy()
        X_permuted[:, :, i] = np.random.permutation(X_permuted[:, :, i])
        
        # 预测并计算误差
        perm_pred = model.predict(X_permuted)
        perm_mae = mean_absolute_error(scaler.inverse_transform(y_test), 
                                       scaler.inverse_transform(perm_pred))
        
        # 重要性 = 打乱后误差增加量
        importance = perm_mae - baseline_mae
        perm_importance.append((feature, importance))
    
    # 按重要性排序
    perm_importance.sort(key=lambda x: x[1], reverse=True)
    return perm_importance

def generate_predictions(days=30):
    """生成未来30天的预测"""
    model = load_model('models/gold_prediction_model_latest.keras')  # 更新文件扩展名为.keras
    
    # 获取最新数据
    df = fetch_gold_data()
    df_features = engineer_features(df)
    
    # 准备预测输入 - 最新的窗口
    window = 60  # 添加默认窗口大小
    latest_data = df_features.iloc[-window:].values
    scaler = MinMaxScaler()
    scaler.fit(df_features[['price']].values)  # 只对价格列进行拟合
    
    # 对所有特征进行归一化
    features_scaler = MinMaxScaler()
    latest_scaled = features_scaler.fit_transform(latest_data)
    X_pred = latest_scaled.reshape(1, window, df_features.shape[1])
    
    # 递归预测
    predictions = []
    current_sequence = X_pred.copy()
    
    for _ in range(days):
        # 预测下一个值
        next_pred = model.predict(current_sequence, verbose=0)[0]
        predictions.append(next_pred[0])
        
        # 更新序列
        next_input = np.zeros((1, 1, df_features.shape[1]))
        next_input[0, 0, 0] = next_pred[0]  # 假设price是第一个特征
        
        # 滚动窗口
        current_sequence = np.append(current_sequence[:, 1:, :], 
                                     next_input, axis=1)
    
    # 反归一化预测结果
    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1))
    
    # 创建日期索引 - 从最新数据的下一天开始
    last_date = df.index[-1]
    future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(days)]
    
    return predictions, future_dates

# 针对普通用户的直观可视化函数
def plot_future_trend(future_days=5, use_english=False, use_sample_data=False):
    """创建更直观的未来金价走势图
    
    Args:
        future_days: 预测的未来天数
        use_english: 是否使用英文
        use_sample_data: 当没有训练好的模型时，是否使用样例数据
    """
    try:
        # 获取预测数据
        predictions, future_dates = generate_predictions(future_days)
        predictions = predictions.flatten()
    except Exception as e:
        logger.warning(f"获取预测数据失败: {str(e)}，将使用样例数据")
        if not use_sample_data:
            raise
            
        # 使用样例数据创建示例图表
        df = fetch_gold_data()
        last_date = df.index[-1]
        future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(future_days)]
        
        # 生成一些模拟的预测结果 - 以最后一个价格为基准，添加一些随机波动
        last_price = df['price'].iloc[-1]
        # 使用随机种子确保每次生成相同结果
        np.random.seed(42)
        # 生成一个趋势增长或下降的序列，再加上随机波动
        trend = np.linspace(0, 0.05, future_days)  # 生成0到5%的线性增长
        noise = np.random.normal(0, 0.01, future_days)  # 添加1%左右的随机波动
        predictions = last_price * (1 + trend + noise)
        
        logger.warning("使用样例数据创建的示例图表，仅供参考")
    
    # 获取历史数据作为参考
    df = fetch_gold_data()
    recent_days = 30  # 显示最近30天的历史数据
    historical = df['price'].iloc[-recent_days:].copy()
    
    # 设置字体和样式
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(14, 8))
    
    # 设置语言
    if use_english:
        title = "Gold Price Forecast (Next 5 Days)"
        subtitle = f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d')}"
        ylabel = "Price (CNY/gram)"
        trend_label = "Price Trend"
        history_label = "Historical Price"
        prediction_label = "Predicted Price"
        today_label = "Today"
        note_text = "Note: This forecast is based on historical data patterns and may vary due to market conditions."
        if predictions[-1] > predictions[0]:
            trend_text = f"🔼 Upward Trend: Price expected to increase by {predictions[-1]-predictions[0]:.2f} CNY/gram ({(predictions[-1]/predictions[0]-1)*100:.1f}%)"
            trend_color = 'green'
        elif predictions[-1] < predictions[0]:
            trend_text = f"🔽 Downward Trend: Price expected to decrease by {predictions[0]-predictions[-1]:.2f} CNY/gram ({(1-predictions[-1]/predictions[0])*100:.1f}%)"
            trend_color = 'red'
        else:
            trend_text = "◀️▶️ Stable Price: No significant change expected"
            trend_color = 'blue'
        
        # 添加样例数据警告
        if use_sample_data:
            sample_warning = "⚠️ SAMPLE DATA - FOR DEMONSTRATION ONLY ⚠️"
    else:
        title = "黄金价格预测（未来5天）"
        subtitle = f"最后更新：{datetime.datetime.now().strftime('%Y-%m-%d')}"
        ylabel = "价格（元/克）"
        trend_label = "价格趋势"
        history_label = "历史价格"
        prediction_label = "预测价格"
        today_label = "今天"
        note_text = "注意：该预测基于历史数据模式，可能因市场条件而变化。"
        if predictions[-1] > predictions[0]:
            trend_text = f"🔼 上涨趋势：预计价格将上涨 {predictions[-1]-predictions[0]:.2f} 元/克 ({(predictions[-1]/predictions[0]-1)*100:.1f}%)"
            trend_color = 'red'  # 在中国文化中，红色通常代表上涨
        elif predictions[-1] < predictions[0]:
            trend_text = f"🔽 下跌趋势：预计价格将下跌 {predictions[0]-predictions[-1]:.2f} 元/克 ({(1-predictions[-1]/predictions[0])*100:.1f}%)"
            trend_color = 'green'  # 在中国文化中，绿色通常代表下跌
        else:
            trend_text = "◀️▶️ 价格稳定：预计无明显变化"
            trend_color = 'blue'
            
        # 添加样例数据警告
        if use_sample_data:
            sample_warning = "⚠️ 示例数据 - 仅用于演示 ⚠️"
    
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
    
    # 保存图表
    language = "en" if use_english else "cn"
    plt.savefig(f'gold_future_trend_{language}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 返回预测结果和日期，以便可能的进一步使用
    return predictions, future_dates

# 数据加载时进行内存优化
def optimize_dataframe(df):
    """降低DataFrame内存使用"""
    for col in df.select_dtypes(include=['float']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

if __name__ == "__main__":
    import argparse
    
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
        plot_future_trend(future_days=args.future, use_english=use_english, use_sample_data=args.sample)
    else:
        main(
            window=args.window,
            future_days=args.future,
            epochs=100,
            batch_size=32,
            force_retrain=args.force_retrain,
            retraining_period_days=args.retrain_period
        )
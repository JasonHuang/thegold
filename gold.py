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
from typing import Optional

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

# 主程序
def main(window=60, future_days=5, epochs=100, batch_size=32):
    """主函数
    
    Args:
        window: 时间窗口大小
        future_days: 预测未来的天数
        epochs: 训练轮数
        batch_size: 批量大小
    """
    try:
        # 创建模型保存目录
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        
        # 获取当前时间，用于模型文件命名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f'gold_prediction_model_{timestamp}.h5')
        
        # 获取数据
        df = fetch_gold_data()
        print(f"最新数据日期：{df.index[-1].strftime('%Y-%m-%d')}")
        
        # 数据质量检查
        print(f"数据统计: \n{df.describe()}")
        
        # 绘制原始数据图表，检查异常值和趋势
        plt.figure(figsize=(12,6))
        plt.plot(df.index, df['price'])
        plt.title('Gold Price Trend')
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
        
        # 可视化结果
        logger.info("生成可视化图表...")
        plot_results(actual, pred, history, future_days, use_english=True)
        
        # 保存模型使用新格式
        model.save(f'models/gold_prediction_model_{timestamp}.keras')
        
        logger.info(f"模型已保存到: {model_path}")
        
        return model, scaler, metrics
        
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
    model = load_model('models/gold_prediction_model_latest.h5')
    
    # 获取最新数据
    df = fetch_gold_data()
    df_features = engineer_features(df)
    
    # 准备预测输入 - 最新的窗口
    latest_data = df_features.iloc[-window:].values
    scaler = MinMaxScaler()
    latest_scaled = scaler.fit_transform(latest_data)
    X_pred = latest_scaled.reshape(1, window, df_features.shape[1])
    
    # 递归预测
    predictions = []
    current_sequence = X_pred.copy()
    
    for _ in range(days):
        # 预测下一个值
        next_pred = model.predict(current_sequence)[0]
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
    
    return predictions

# 数据加载时进行内存优化
def optimize_dataframe(df):
    """降低DataFrame内存使用"""
    for col in df.select_dtypes(include=['float']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

if __name__ == "__main__":
    main(window=60, future_days=5, epochs=100, batch_size=32)
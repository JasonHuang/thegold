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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 数据获取模块
def fetch_gold_data(symbol="AU0", start_date=None, end_date=None):
    """获取沪金主力合约数据
    
    Args:
        symbol: 合约代码，默认为"AU0"（沪金主力合约）
        start_date: 开始日期，格式为"YYYY-MM-DD"
        end_date: 结束日期，格式为"YYYY-MM-DD"
        
    Returns:
        处理后的DataFrame，包含日期索引和价格列
    """
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
    # 创建一个新的DataFrame
    df_feat = df.copy()
    
    # 添加滚动均值特征
    df_feat['MA5'] = df_feat['price'].rolling(window=5).mean()
    df_feat['MA20'] = df_feat['price'].rolling(window=20).mean()
    df_feat['MA60'] = df_feat['price'].rolling(window=60).mean()
    
    # 添加滚动标准差（波动率）
    df_feat['volatility_20'] = df_feat['price'].rolling(window=20).std()
    
    # 添加价格变化
    df_feat['price_change'] = df_feat['price'].pct_change()
    
    # 添加动量指标
    df_feat['momentum_5'] = df_feat['price'] - df_feat['price'].shift(5)
    df_feat['momentum_20'] = df_feat['price'] - df_feat['price'].shift(20)
    
    # 移除NaN值
    df_feat = df_feat.dropna()
    
    logger.info(f"完成特征工程，特征数量: {df_feat.shape[1]}")
    return df_feat

# 数据预处理模块
def preprocess_data(data, window=60, future=5, test_size=0.2, feature_columns=None):
    """创建时间序列样本
    
    Args:
        data: 特征工程后的DataFrame
        window: 时间窗口大小
        future: 预测未来的天数
        test_size: 测试集比例
        feature_columns: 要使用的特征列表，默认使用所有列
        
    Returns:
        训练集和测试集的特征和标签，以及用于反归一化的scaler
    """
    if feature_columns is None:
        feature_columns = data.columns.tolist()
    
    # 选择特征
    features = data[feature_columns]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    
    # 为价格单独创建一个scaler，用于后续转换
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    price_scaler.fit(data[['price']])
    
    X, y = [], []
    n_features = len(feature_columns)  # 特征数量
    
    for i in range(window, len(scaled_features)-future+1):
        # 添加所有特征的滑动窗口
        X.append(scaled_features[i-window:i])
        # 只预测价格
        y.append(scaled_features[i:i+future, 0])  # 假设price是第一列
    
    X = np.array(X)
    y = np.array(y)
    
    # 划分数据集
    split = int((1 - test_size) * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    logger.info(f"数据预处理完成: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
    logger.info(f"测试集: X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")
    
    return (X_train, y_train, X_test, y_test, price_scaler)

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
        # 第一层LSTM，使用双向LSTM提高性能
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        # 第二层LSTM
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        # 全连接层
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(future_days)
    ])
    
    # 使用Adam优化器并指定学习率
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    
    logger.info(f"模型构建完成: {model.summary()}")
    return model

# 评估模块
def evaluate_model(model, X_test, y_test, scaler):
    """评估模型性能
    
    Args:
        model: 训练好的模型
        X_test: 测试集特征
        y_test: 测试集标签
        scaler: 用于反归一化的scaler
        
    Returns:
        预测结果和评估指标
    """
    # 预测
    predictions = model.predict(X_test)
    
    # 反归一化
    # 创建一个临时数组，形状为 (样本数 * 预测天数, 1)
    temp_y_test = y_test.reshape(-1, 1)
    temp_pred = predictions.reshape(-1, 1)
    
    # 反归一化
    actual = scaler.inverse_transform(np.zeros((temp_y_test.shape[0], scaler.n_features_in_)))[:, 0].reshape(-1, 1)
    actual[:, 0] = scaler.inverse_transform(np.hstack([temp_y_test, np.zeros((temp_y_test.shape[0], scaler.n_features_in_-1))]))[:, 0]
    
    pred = scaler.inverse_transform(np.zeros((temp_pred.shape[0], scaler.n_features_in_)))[:, 0].reshape(-1, 1)
    pred[:, 0] = scaler.inverse_transform(np.hstack([temp_pred, np.zeros((temp_pred.shape[0], scaler.n_features_in_-1))]))[:, 0]
    
    # 计算评估指标
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    
    logger.info(f"模型评估: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
    
    return actual, pred, {'mae': mae, 'rmse': rmse, 'r2': r2}

# 可视化模块
def plot_results(actual, pred, history=None, predictions_days=5):
    """可视化结果
    
    Args:
        actual: 实际值
        pred: 预测值
        history: 训练历史记录
        predictions_days: 预测的天数
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 创建一个2x2的子图布局
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    
    # 1. 价格预测对比图
    axs[0, 0].plot(actual, label='实际价格', alpha=0.7, linewidth=2)
    axs[0, 0].plot(pred, label='预测价格', linestyle='--', linewidth=2)
    axs[0, 0].set_title(f"黄金价格预测 (未来{predictions_days}天 MAE: {mean_absolute_error(actual, pred):.2f}元)", fontsize=15)
    axs[0, 0].set_xlabel('样本', fontsize=12)
    axs[0, 0].set_ylabel('价格 (元/克)', fontsize=12)
    axs[0, 0].legend(fontsize=12)
    axs[0, 0].grid(True)
    
    # 2. 价格预测散点图
    axs[0, 1].scatter(actual, pred, alpha=0.5)
    min_val = min(actual.min(), pred.min())
    max_val = max(actual.max(), pred.max())
    axs[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
    axs[0, 1].set_title("预测值 vs 实际值", fontsize=15)
    axs[0, 1].set_xlabel('实际价格', fontsize=12)
    axs[0, 1].set_ylabel('预测价格', fontsize=12)
    
    # 3. 价格误差直方图
    error = actual - pred
    sns.histplot(error, bins=50, kde=True, ax=axs[1, 0])
    axs[1, 0].set_title("预测误差分布", fontsize=15)
    axs[1, 0].set_xlabel('预测误差', fontsize=12)
    axs[1, 0].set_ylabel('频率', fontsize=12)
    
    # 4. 训练和验证损失曲线 (如果提供了history)
    if history:
        axs[1, 1].plot(history.history['loss'], label='训练损失')
        axs[1, 1].plot(history.history['val_loss'], label='验证损失')
        axs[1, 1].set_title("训练和验证损失", fontsize=15)
        axs[1, 1].set_xlabel('Epoch', fontsize=12)
        axs[1, 1].set_ylabel('损失', fontsize=12)
        axs[1, 1].legend(fontsize=12)
    else:
        axs[1, 1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('gold_price_prediction_results.png', dpi=300, bbox_inches='tight')
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
        
        # 特征工程
        df_features = engineer_features(df)
        
        # 数据预处理
        X_train, y_train, X_test, y_test, scaler = preprocess_data(
            df_features, 
            window=window, 
            future=future_days,
            test_size=0.2
        )
        
        # 构建模型
        model = build_model((X_train.shape[1], X_train.shape[2]), future_days)
        
        # 创建回调函数
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        # 模型训练
        logger.info("开始训练模型...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
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
        plot_results(actual, pred, history, future_days)
        
        logger.info(f"模型已保存到: {model_path}")
        
        return model, scaler, metrics
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main(window=60, future_days=5, epochs=100, batch_size=32)
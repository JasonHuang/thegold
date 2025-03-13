# -*- coding: utf-8 -*-
# 黄金价格预测完整示例（2025年3月更新版）
import numpy as np
import pandas as pd
import akshare as ak
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 数据获取模块（网页2、网页5）
def fetch_gold_data():
    """获取沪金主力合约数据"""
    df = ak.futures_zh_daily_sina(symbol="AU0")  # 沪金主力合约
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    return df[['close']].rename(columns={'close': 'price'})

# 数据预处理模块（网页3、网页6）
def preprocess_data(data, window=60, future=5):
    """创建时间序列样本"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data.values.reshape(-1,1))
    
    X, y = [], []
    for i in range(window, len(scaled)-future+1):
        X.append(scaled[i-window:i])
        y.append(scaled[i:i+future])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, future)
    
    # 划分数据集
    split = int(0.8 * len(X))
    return (X[:split], y[:split], 
            X[split:], y[split:], scaler)

# 模型构建模块（网页1、网页6）
def build_model(input_shape, future_days=5):
    """构建LSTM模型"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(future_days)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 主程序
if __name__ == "__main__":
    # 获取数据（网页2）
    df = fetch_gold_data()
    print(f"最新数据日期：{df.index[-1].strftime('%Y-%m-%d')}")
    
    # 数据预处理
    X_train, y_train, X_test, y_test, scaler = preprocess_data(df, window=60, future=5)
    
    # 构建模型
    model = build_model((X_train.shape[1], 1))
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    
    # 模型训练（网页6）
    history = model.fit(X_train, y_train,
                       epochs=100,
                       batch_size=64,
                       validation_split=0.2,
                       callbacks=[early_stop],
                       verbose=1)
    
    # 预测与评估（网页3）
    predictions = model.predict(X_test)
    actual = scaler.inverse_transform(y_test.reshape(-1,1))
    pred = scaler.inverse_transform(predictions.reshape(-1,1))
    
    # 评估指标（网页7）
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    print(f"MAE: {mae:.2f} 元/克\nR²: {r2:.2%}")
    
    # 可视化（网页3）
    plt.figure(figsize=(15,6))
    plt.plot(actual, label='实际价格', alpha=0.7)
    plt.plot(pred, label='预测价格', linestyle='--')
    plt.title(f"黄金价格预测 (未来5天平均误差: {mae:.2f}元)")
    plt.legend()
    plt.grid(True)
    plt.show()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
黄金价格预测系统 - 主入口文件
"""

import sys
import os
from src.main import main, generate_user_friendly_chart

if __name__ == "__main__":
    # 添加当前目录到Python路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # 导入参数解析
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
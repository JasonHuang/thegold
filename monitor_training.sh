#!/bin/bash
cd ~/myprojects/thegold
echo "=== 模型训练状态检查 ==="
echo "最后训练时间："
grep "training_date" models/latest_model_metadata.json
echo "下次训练时间："
grep "next_training_date" models/latest_model_metadata.json
echo "模型性能指标："
grep -E "mae|rmse|r2" models/latest_model_metadata.json
echo "最新训练日志："
tail -n 5 retraining.log

#!/bin/bash
# 黄金价格预测模型自动重新训练脚本
# 可以设置为crontab定期执行

# 脚本位置
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# 记录日志的函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> retraining.log
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 定义日志文件
LOG_FILE="retraining.log"
touch $LOG_FILE

# 检查虚拟环境是否存在
if [ ! -d "gold-env" ]; then
    log "虚拟环境不存在，尝试创建..."
    bash setup.sh
    if [ $? -ne 0 ]; then
        log "设置环境失败，请检查setup.sh脚本"
        exit 1
    fi
fi

# 激活虚拟环境
log "激活虚拟环境..."
source gold-env/bin/activate

# 记录开始时间
START_TIME=$(date +%s)
log "开始执行模型重新训练检查"

# 运行Python脚本，检查是否需要重新训练
# --retrain-period参数可以根据实际需求调整
log "执行Python脚本..."
python run.py --retrain-period 90

# 检查执行结果
if [ $? -eq 0 ]; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    log "执行完成，耗时: ${DURATION} 秒"
else
    log "执行失败，请检查日志获取详细信息"
fi

# 汇总当前模型信息
if [ -f "models/latest_model_metadata.json" ]; then
    log "当前模型信息:"
    cat models/latest_model_metadata.json | grep -E "training_date|next_training_date|mae|rmse|r2" >> $LOG_FILE
else
    log "未找到模型元数据信息"
fi

# 清理旧的模型文件，保留最新的3个
log "清理旧的模型文件..."
ls -t models/gold_prediction_model_*.keras | tail -n +4 | xargs -I {} rm {} 2>/dev/null
ls -t models/gold_prediction_model_*_metadata.json | tail -n +4 | xargs -I {} rm {} 2>/dev/null

log "脚本执行完毕"

# 退出虚拟环境
deactivate 
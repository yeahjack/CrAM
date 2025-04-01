#!/usr/bin/env bash

# 启动 sglang 服务
./launch_sglang.sh &
SGLANG_PID=$!

# 等待几秒钟，让 sglang 服务完全启动
sleep 30

# 设置参数
UTILIZATION_THRESHOLD=10       # GPU 使用率阈值（百分比）
CHECK_INTERVAL=60              # 检查间隔（秒）
MAX_LOW_UTIL_TIME=600         # 最大低利用率时间（秒），例如30分钟

# 初始化低利用率计时器
low_util_timer=0

echo "监控 sglang 进程 (PID: $SGLANG_PID) 的 GPU 使用率..."

while true; do
    # 检查 sglang 进程是否还在运行
    if ! ps -p $SGLANG_PID > /dev/null; then
        echo "sglang 进程已终止，退出监控"
        exit 0
    fi
    
    # 查找运行 sglang::scheduler 的 GPU
    gpu_indices=$(nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader | grep "sglang::scheduler" | cut -d, -f1 | xargs -I{} nvidia-smi --query-gpu=index --format=csv,noheader --id={})
    
    # 如果找不到任何运行 sglang 的 GPU，尝试用进程 ID 查找
    if [ -z "$gpu_indices" ]; then
        gpu_indices=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$SGLANG_PID" | cut -d, -f1 | xargs -I{} nvidia-smi --query-gpu=index --format=csv,noheader --id={})
    fi
    
    # 如果仍然找不到，监控所有 GPU
    if [ -z "$gpu_indices" ]; then
        echo "未找到运行 sglang 的 GPU，监控所有 GPU"
        gpu_indices=$(nvidia-smi --query-gpu=index --format=csv,noheader)
    fi
    
    # 将 GPU 索引转换为逗号分隔的列表
    gpu_list=$(echo $gpu_indices | tr '\n' ',' | sed 's/,$//g')
    
    echo "监控以下 GPU: $gpu_list"
    
    # 获取这些 GPU 的平均利用率
    total_util=0
    gpu_count=0
    
    for gpu_id in $(echo $gpu_indices); do
        util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id)
        total_util=$(echo "$total_util + $util" | bc)
        gpu_count=$((gpu_count + 1))
    done
    
    # 计算平均利用率
    if [ $gpu_count -gt 0 ]; then
        avg_util=$(echo "scale=2; $total_util / $gpu_count" | bc)
    else
        avg_util=0
    fi
    
    echo "$(date): GPU 平均利用率: $avg_util%"
    
    # 检查 GPU 利用率是否低于阈值
    if (( $(echo "$avg_util < $UTILIZATION_THRESHOLD" | bc -l) )); then
        low_util_timer=$((low_util_timer + CHECK_INTERVAL))
        echo "低利用率持续时间: $low_util_timer 秒"
        
        # 如果低利用率时间超过最大允许时间，则终止进程
        if [ $low_util_timer -ge $MAX_LOW_UTIL_TIME ]; then
            echo "GPU 利用率持续低于 $UTILIZATION_THRESHOLD% 超过 $MAX_LOW_UTIL_TIME 秒，终止 sglang 进程"
            kill $SGLANG_PID
            echo "sglang 进程已终止"
            exit 0
        fi
    else
        # 重置低利用率计时器
        low_util_timer=0
    fi
    
    sleep $CHECK_INTERVAL
done
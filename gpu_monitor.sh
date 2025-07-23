#!/bin/bash

# GPU内存监控脚本
# 每10秒检查一次nvidia-smi输出，当内存使用低于5000MiB时执行自定义命令

# ===================配置区域===================
MEMORY_THRESHOLD=5000  # 内存阈值（MiB）
CHECK_INTERVAL=10      # 检查间隔（秒）
TARGET_TOTAL_MEMORY="24564MiB"  # 目标GPU总内存

# 🔧 在这里设置您的自定义命令
CUSTOM_COMMAND="echo 'GPU内存可用，开始执行任务!' && python train.py micro_batch_size_per_gpu=1 ppo_mini_batch_size=8 actor_rollout_ref.rollout.max_model_len=2048 actor_rollout_ref.rollout.response_length=128 > test-p.log"

# 日志开关
ENABLE_LOGGING=false
LOG_FILE="gpu_monitor.log"
# ===============================================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [ "$ENABLE_LOGGING" = true ]; then
        echo "[$timestamp] $message" >> "$LOG_FILE"
    fi
    echo -e "$message"
}

# 检查nvidia-smi是否可用
check_nvidia_smi() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_message "${RED}❌ 错误: nvidia-smi 命令未找到${NC}"
        exit 1
    fi
}

# 解析GPU内存使用情况
parse_gpu_memory() {
    local nvidia_output="$1"
    
    # 使用grep和sed提取内存使用信息
    # 匹配格式: "数字MiB /  24564MiB"
    local memory_info=$(echo "$nvidia_output" | grep -o '[0-9]\+MiB /  '"$TARGET_TOTAL_MEMORY" | head -1)
    
    if [ -n "$memory_info" ]; then
        # 提取使用的内存数值
        local used_memory=$(echo "$memory_info" | grep -o '^[0-9]\+')
        echo "$used_memory"
    else
        echo ""
    fi
}

# 执行自定义命令
execute_custom_command() {
    log_message "${GREEN}🚀 内存使用低于阈值，执行自定义命令...${NC}"
    
    # 执行自定义命令
    eval "$CUSTOM_COMMAND"
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_message "${GREEN}✅ 自定义命令执行成功${NC}"
    else
        log_message "${RED}❌ 自定义命令执行失败 (退出码: $exit_code)${NC}"
    fi
    
    return $exit_code
}

# 主监控循环
main_monitor_loop() {
    local consecutive_low_count=0
    local command_executed=false
    
    log_message "${BLUE}🔍 开始监控GPU内存使用情况...${NC}"
    log_message "${BLUE}📊 阈值: ${MEMORY_THRESHOLD}MiB, 检查间隔: ${CHECK_INTERVAL}秒${NC}"
    
    while true; do
        # 获取nvidia-smi输出
        local nvidia_output=$(nvidia-smi 2>/dev/null)
        local nvidia_exit_code=$?
        
        if [ $nvidia_exit_code -ne 0 ]; then
            log_message "${RED}⚠️  警告: nvidia-smi 执行失败${NC}"
            sleep "$CHECK_INTERVAL"
            continue
        fi
        
        # 解析内存使用
        local used_memory=$(parse_gpu_memory "$nvidia_output")
        
        if [ -n "$used_memory" ]; then
            local timestamp=$(date '+%H:%M:%S')
            
            if [ "$used_memory" -lt "$MEMORY_THRESHOLD" ]; then
                consecutive_low_count=$((consecutive_low_count + 1))
                log_message "${GREEN}[$timestamp] ✅ GPU内存: ${used_memory}MiB/${TARGET_TOTAL_MEMORY} (低于阈值 $consecutive_low_count 次)${NC}"
                
                # 防止重复执行，只在第一次低于阈值时执行
                if [ "$command_executed" = false ]; then
                    execute_custom_command
                    command_executed=true
                fi
            else
                if [ "$consecutive_low_count" -gt 0 ]; then
                    log_message "${YELLOW}[$timestamp] 📈 GPU内存: ${used_memory}MiB/${TARGET_TOTAL_MEMORY} (恢复到阈值以上)${NC}"
                    consecutive_low_count=0
                    command_executed=false  # 重置执行标志
                else
                    log_message "${YELLOW}[$timestamp] 📊 GPU内存: ${used_memory}MiB/${TARGET_TOTAL_MEMORY}${NC}"
                fi
            fi
        else
            log_message "${RED}⚠️  无法解析GPU内存信息${NC}"
        fi
        
        sleep "$CHECK_INTERVAL"
    done
}

# 信号处理
cleanup() {
    log_message "${BLUE}🛑 接收到退出信号，停止监控...${NC}"
    exit 0
}

# 设置信号处理
trap cleanup SIGINT SIGTERM

# 脚本使用说明
show_usage() {
    echo "GPU内存监控脚本"
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -t, --threshold NUM  设置内存阈值 (默认: 5000MiB)"
    echo "  -i, --interval NUM   设置检查间隔 (默认: 10秒)"
    echo "  -c, --command 'CMD'  设置自定义命令"
    echo ""
    echo "示例:"
    echo "  $0 -t 1500 -i 5 -c 'python train.py'"
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -t|--threshold)
                MEMORY_THRESHOLD="$2"
                shift 2
                ;;
            -i|--interval)
                CHECK_INTERVAL="$2"
                shift 2
                ;;
            -c|--command)
                CUSTOM_COMMAND="$2"
                shift 2
                ;;
            *)
                echo "未知参数: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# 主函数
main() {
    # 解析命令行参数
    parse_arguments "$@"
    
    # 检查nvidia-smi
    check_nvidia_smi
    
    # 显示配置信息
    log_message "${BLUE}🔧 配置信息:${NC}"
    log_message "${BLUE}   内存阈值: ${MEMORY_THRESHOLD}MiB${NC}"
    log_message "${BLUE}   检查间隔: ${CHECK_INTERVAL}秒${NC}"
    log_message "${BLUE}   自定义命令: ${CUSTOM_COMMAND}${NC}"
    log_message "${BLUE}   日志文件: ${LOG_FILE}${NC}"
    echo ""
    
    # 开始监控
    main_monitor_loop
}

# 执行主函数
main "$@"

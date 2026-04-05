#!/bin/bash

# 如果log文件夹不存在，则创建
if [ ! -d "log" ]; then
    mkdir log
fi


# 获取当前时间，格式为YYYY-MM-DD_HH-MM-SS
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
comment="MIF-B-DrugBank-hy-n-NTXenLoss-0.1-kimi"
file_name="log/$current_time-$comment.log"
export ALL_PROXY=socks5://192.168.0.107:7890
export HTTP_PROXY=http://192.168.0.107:7890
export HTTPS_PROXY=http://192.168.0.107:7890
# 运行nohup命令，输出日志到log目录下，文件名为当前时间sss
echo $comment > "$file_name"
nohup python -u main.py DrugBank -g 0 -m MIF-DTI-B  >> "$file_name" 2>&1 & 

# 实时显示日志文件内容
tail -f "$file_name"

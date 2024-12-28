#!/bin/bash

# 获取当前时间，格式为YYYY-MM-DD_HH-MM-SS
current_time=$(date +"%Y-%m-%d_%H-%M-%S")

# 运行nohup命令，输出日志到log目录下，文件名为当前时间
nohup python -u main.py DrugBank -g 1 > "log/$current_time.log" 2>&1 & 

# 实时显示日志文件内容
tail -f "log/$current_time.log"

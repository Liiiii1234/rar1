#!/bin/bash

#SBATCH --job-name=Li          # 作业名
#SBATCH --partition=aiaca800          # GPU 队列（和你模板保持一致）
#SBATCH --qos=2a800                   # QoS 配置
#SBATCH --nodes=1                     # 节点数量
#SBATCH --ntasks-per-node=1           # 每节点进程数
#SBATCH --cpus-per-task=8             # 给 8 核更稳
#SBATCH --gres=gpu:1                  # 1 块 GPU
#SBATCH --mem=32G                     # 给 32G 内存，训练更稳
#SBATCH --output=run_dpo_%j.log       # 输出日志
#SBATCH --error=run_dpo_%j.err        # 错误日志
#SBATCH --mail-type=END,FAIL          # 结束/失败发邮件
#SBATCH --mail-user=Jianuo.Li23@student.xjtlu.edu.cn

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# 进入作业提交目录
cd $SLURM_SUBMIT_DIR

# 你的 DPO 训练命令（已优化显存，HPC 上完美跑）
python trl/scripts/dpo.py \
--dataset_name trl-lib/ultrafeedback_binarized \
--model_name_or_path Qwen/Qwen2-0.5B-Instruct \
--learning_rate 5.0e-6 \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--max_steps 1000 \
--gradient_accumulation_steps 16 \
--eval_strategy steps \
--eval_steps 50 \
--output_dir Qwen2-0.5B-DPO \
--no_remove_unused_columns \
--use_peft \
--lora_r 32 \
--lora_alpha 16 \
--fp16 \
--gradient_checkpointing


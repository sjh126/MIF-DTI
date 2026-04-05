# -*- coding:utf-8 -*-
import os
import optuna
import torch
from model import MIFDTI
from RunModel import run_MIF_model  # 你提供的主训练逻辑
from LossFunction import HybridLoss

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===== 目标函数 =====
def objective(trial, SEED=42, DATASET="Davis", K_FOLD=3):

    # ---- 离散搜索空间定义 ----
    alpha_focal = trial.suggest_categorical("alpha_focal", [0.25, 0.5, 0.75, 1.0])
    gamma = trial.suggest_categorical("gamma", [1.0, 1.5, 2.0, 3.0])
    lambda_focal = trial.suggest_categorical("lambda_focal", [0.3, 0.5, 0.6, 0.7])
    ce_weight_type = trial.suggest_categorical("ce_weight", ["none", "dataset"])

    print(f"🔍 当前尝试组合: alpha_focal={alpha_focal}, gamma={gamma}, lambda_focal={lambda_focal}, ce_weight={ce_weight_type}")

    # ---- 加载数据权重 ----
    if DATASET == "Davis":
        weight_loss = torch.FloatTensor([0.3, 0.7]).to(DEVICE)
    elif DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.2, 0.8]).to(DEVICE)
    else:
        weight_loss = None
    if ce_weight_type == "none":
        weight_loss = None

    # ---- 定义Loss函数 ----
    Loss = HybridLoss(
        weight_ce=weight_loss,
        alpha_focal=alpha_focal,
        gamma=gamma,
        lambda_focal=lambda_focal,
        DEVICE=DEVICE
    )

    # ---- 创建模型并执行一次K-Fold训练 ----
    try:
        auc_mean = run_MIF_model(
            SEED=SEED,
            DATASET=DATASET,
            MODEL=MIFDTI,
            K_Fold=K_FOLD,
            LOSS=Loss,
            device=DEVICE
        )

    except Exception as e:
        print(f"❌ Trial failed due to: {e}")
        return None

    print(f"✅ 当前组合平均AUC: {auc_mean:.4f}")
    return auc_mean


# ===== 主函数 =====
def main():
    SEED = 42
    DATASET = "Davis"
    N_TRIALS = 20  # 搜索次数
    K_FOLD = 3

    study_name = f"MIFDTI_LossSearch_{DATASET}"
    storage_name = f"sqlite:///{study_name}.db"

    print(f"⚙️ 启动Optuna搜索任务: {study_name}")
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage_name,
        load_if_exists=True
    )

    study.optimize(lambda trial: objective(trial, SEED, DATASET, K_FOLD), n_trials=N_TRIALS)

    print("\n🎯 搜索完成！")
    print("最佳参数:")
    print(study.best_params)
    print("最佳AUC:", study.best_value)

    # 保存搜索结果
    study.trials_dataframe().to_csv(f"{study_name}_trials.csv", index=False)


if __name__ == "__main__":
    main()

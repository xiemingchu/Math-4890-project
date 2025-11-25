import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import (
    train_test_split,
    KFold,
    LeaveOneOut,
    cross_val_score,
    RandomizedSearchCV,
    cross_val_predict,         # ✅ 新增：用来画 CV 预测散点
)

from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt   # ✅ 画图
import os

# ============================================================
# 0. 图像输出路径
# ============================================================

# 把图都存到这个文件夹里，你可以按需要改路径
plot_dir = r"C:\Users\Lenovo\Desktop\MATH 4890\stage2_plots"
os.makedirs(plot_dir, exist_ok=True)

# 一个小工具函数：画散点并保存
def plot_scatter(y_true, y_pred, title, filename):
    """
    y_true: 一维真实值
    y_pred: 一维预测值
    title: 图标题
    filename: 保存文件名（不含路径）
    """
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    # 画 y=x 参考线，R² 正时点会更多贴近这条线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), dpi=300)
    plt.close()

# ============================================================
# 1. 路径 & 目标设置
# ============================================================

# 主数据
data_path = r"C:\Users\Lenovo\Desktop\MATH 4890\变量筛选_cleaned.xlsx"

# 阶段一得到的“重合变量”文件（自评焦虑 / 自评抑郁）
feat_path_anx = r"C:\Users\Lenovo\Desktop\MATH 4890\Top30_overlap_a_promis_anx_bline.xlsx"
feat_path_dep = r"C:\Users\Lenovo\Desktop\MATH 4890\Top30_overlap_a_promis_dep_bline.xlsx"

# 所有 PROMIS 目标变量（防止泄漏，要从 X 中剔除）
PROMIS_TARGETS = [
    "p_promis_anx_bline",
    "p_promis_dep_bline",
    "a_promis_anx_bline",
    "a_promis_dep_bline"
]

# 只分析自评的两个目标
SELF_TARGETS = [
    "a_promis_anx_bline",
    "a_promis_dep_bline"
]

FEATURE_FILES = {
    "a_promis_anx_bline": feat_path_anx,
    "a_promis_dep_bline": feat_path_dep
}

# 输出结果
output_path = r"C:\Users\Lenovo\Desktop\MATH 4890\阶段二_自评焦虑抑郁_模型结果.xlsx"

# ============================================================
# 2. 读入主数据并预处理
# ============================================================

df = pd.read_excel(data_path)

# 尽量转成数值
df = df.apply(pd.to_numeric, errors="coerce")

# 数值列按列均值填补
df = df.fillna(df.select_dtypes(include=[np.number]).mean())
# 若仍有 NaN，填 0
df = df.fillna(0)

print("数据维度：", df.shape)

# ============================================================
# 3. 工具函数：评估 + 画图（Train/Test + 5-fold CV + LOOCV-RMSE）
# ============================================================

def evaluate_and_plot(model, model_name, target_name,
                      X_train, X_test, y_train, y_test):
    """
    拟合模型、计算各种指标，并画三张图:
    1）Train: y_true vs y_pred_train
    2）Test : y_true vs y_pred_test
    3）CV   : y_true (train) vs y_pred_cv (5-fold cross_val_predict)
    """

    # ------------ 拟合 ------------
    model.fit(X_train, y_train)

    # ------------ Train/Test 预测 ------------
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Train / Test 指标
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # ------------ 5-fold CV 指标（和之前一样）------------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_cv = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2").mean()
    rmse_cv = np.sqrt(
        -cross_val_score(
            model, X_train, y_train,
            cv=kf, scoring="neg_mean_squared_error"
        ).mean()
    )

    # 👉 为了画“CV 散点图”，我们用 cross_val_predict 得到
    #    每个训练样本在它的验证折上的预测值
    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=kf)
    # 这组 (y_train, y_pred_cv) 就对应 R2_CV 的效果
    plot_scatter(
        y_train.values,
        y_pred_cv,
        title=f"{target_name} - {model_name} (5-fold CV)",
        filename=f"{target_name}_{model_name}_CV_scatter.png"
    )

    # ------------ LOOCV RMSE ------------
    loo = LeaveOneOut()
    rmse_loo = np.sqrt(
        -cross_val_score(
            model, X_train, y_train,
            cv=loo, scoring="neg_mean_squared_error"
        ).mean()
    )

    # ------------ Train/Test 散点图 ------------
    # R² 为正时：点会集中在对角线附近；
    # R² 为负时：点会更像一团云，斜率接近 0。
    plot_scatter(
        y_train.values,
        y_pred_train,
        title=f"{target_name} - {model_name} (Train)",
        filename=f"{target_name}_{model_name}_Train_scatter.png"
    )

    plot_scatter(
        y_test.values,
        y_pred_test,
        title=f"{target_name} - {model_name} (Test)",
        filename=f"{target_name}_{model_name}_Test_scatter.png"
    )

    return r2_train, r2_test, rmse_train, rmse_test, r2_cv, rmse_cv, rmse_loo

# ============================================================
# 4. 主循环：分别对 自评焦虑 / 自评抑郁 建模
# ============================================================

results = []

for target in SELF_TARGETS:
    print("\n====================================================")
    print(f"🎯 开始建模：{target}")
    print("====================================================")

    # ---------- 4.1 读取该 target 的“重合变量列表” ----------
    feat_file = FEATURE_FILES[target]
    feat_df = pd.read_excel(feat_file)

    # 假设文件第一列就是变量名
    feat_list = (
        feat_df.iloc[:, 0]
        .dropna()
        .astype(str)
        .tolist()
    )

    print("📌 从 Excel 读到的重合变量个数：", len(feat_list))

    # ---------- 4.2 构造 X, y ----------
    # 从自变量中剔除所有 PROMIS 目标变量
    available_cols = [c for c in df.columns if c not in PROMIS_TARGETS]

    # 实际可用的特征 = feat_list ∩ available_cols
    final_feats = [f for f in feat_list if f in available_cols]

    print("✅ 实际可用特征个数：", len(final_feats))
    print("✅ 用这些特征建模：", final_feats)

    X = df[final_feats]
    y = df[target]

    # 标准化
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    # Train/Test 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # --------------------------------------------------------
    # 模型 1：ElasticNet（带 CV 调参）
    # --------------------------------------------------------
    print("\n🔧 模型 1：ElasticNet")

    enet = ElasticNetCV(
        alphas=np.logspace(-4, 2, 100),
        l1_ratio=np.linspace(0.1, 1.0, 10),
        cv=5,
        random_state=42
    )

    try:
        scores = evaluate_and_plot(enet, "ElasticNet", target,
                                   X_train, X_test, y_train, y_test)
        results.append([target, "ElasticNet", *scores])
    except Exception as e:
        print("ElasticNet 出错：", e)

    # --------------------------------------------------------
    # 模型 2：XGBoost（RandomizedSearchCV 调参）
    # --------------------------------------------------------
    print("\n🔧 模型 2：XGBoost（RandomizedSearchCV 调参）")

    xgb_base = XGBRegressor(
        random_state=42,
        tree_method="hist"
    )

    param_dist_xgb = {
        "n_estimators": randint(200, 800),
        "max_depth": randint(2, 8),
        "learning_rate": uniform(0.01, 0.2),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4)
    }

    xgb_search = RandomizedSearchCV(
        xgb_base,
        param_distributions=param_dist_xgb,
        n_iter=30,
        scoring="r2",
        cv=3,
        random_state=42,
        n_jobs=-1
    )

    try:
        xgb_search.fit(X_train, y_train)
        best_xgb = xgb_search.best_estimator_
        print("✅ XGBoost 最优参数：", xgb_search.best_params_)

        scores = evaluate_and_plot(best_xgb, "XGBoost", target,
                                   X_train, X_test, y_train, y_test)
        results.append([target, "XGBoost", *scores])
    except Exception as e:
        print("XGBoost 出错：", e)

    # --------------------------------------------------------
    # 模型 3：SVR（RandomizedSearchCV 调参）
    # --------------------------------------------------------
    print("\n🔧 模型 3：SVR")

    svr_base = SVR()

    param_dist_svr = {
        "C": uniform(0.1, 10),
        "gamma": uniform(0.001, 0.1),
        "epsilon": uniform(0.01, 0.1)
    }

    svr_search = RandomizedSearchCV(
        svr_base,
        param_distributions=param_dist_svr,
        n_iter=30,
        scoring="r2",
        cv=3,
        random_state=42,
        n_jobs=-1
    )

    try:
        svr_search.fit(X_train, y_train)
        best_svr = svr_search.best_estimator_
        print("✅ SVR 最优参数：", svr_search.best_params_)

        scores = evaluate_and_plot(best_svr, "SVR", target,
                                   X_train, X_test, y_train, y_test)
        results.append([target, "SVR", *scores])
    except Exception as e:
        print("SVR 出错：", e)

    # --------------------------------------------------------
    # 模型 4：NeuralNet（MLP，收缩网络 + 稳定调参）
    # --------------------------------------------------------
    print("\n🔧 模型 4：NeuralNet (MLP)")

    # 较小网络 + 更强正则 + 提前停止
    mlp_base = MLPRegressor(
        hidden_layer_sizes=(32,),   # 小一些的网络，降低过拟合风险
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        alpha=0.001,                # 基础 L2 正则
        max_iter=3000,
        early_stopping=True,
        n_iter_no_change=30,
        validation_fraction=0.2,
        random_state=42
    )

    # 调参范围：偏保守，优先稳定性
    param_dist_mlp = {
        "hidden_layer_sizes": [(16,), (32,), (48,), (32, 16)],
        "alpha": uniform(1e-4, 5e-3),            # L2 正则略强
        "learning_rate_init": uniform(5e-4, 5e-3)
    }

    # scoring 用 neg_mean_squared_error，更关注整体误差而不是盲目拉高 R²
    mlp_search = RandomizedSearchCV(
        mlp_base,
        param_distributions=param_dist_mlp,
        n_iter=30,
        scoring="neg_mean_squared_error",
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    try:
        mlp_search.fit(X_train, y_train)
        best_mlp = mlp_search.best_estimator_
        print("✅ NeuralNet 最优参数：", mlp_search.best_params_)

        scores = evaluate_and_plot(best_mlp, "NeuralNet", target,
                                   X_train, X_test, y_train, y_test)
        results.append([target, "NeuralNet", *scores])
    except Exception as e:
        print("NeuralNet 出错：", e)

# ============================================================
# 5. 汇总 & 导出结果
# ============================================================

df_results = pd.DataFrame(
    results,
    columns=[
        "Target", "Model",
        "R2_Train", "R2_Test",
        "RMSE_Train", "RMSE_Test",
        "R2_CV", "RMSE_CV",
        "RMSE_LOOCV"
    ]
)

df_results.to_excel(output_path, index=False)

print("\n====================================================")
print("🎉 阶段二（自评焦虑 + 自评抑郁）建模完成！")
print("📁 结果已保存到：", output_path)
print("📊 散点图保存在：", plot_dir)
print("====================================================")

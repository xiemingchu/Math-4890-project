import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os

# ==========================
# 参数设置（和原脚本保持一致）
# ==========================

TOP_K = 30

# 这里只画“自评”焦虑 / 抑郁，对应你阶段二使用的两个标签
TARGETS_FOR_PLOT = [
    "a_promis_anx_bline",
    "a_promis_dep_bline",
]

# 如果你之后想对所有四个 target 都画图，可以把上面列表改成原来那四个

file_path = r"C:\Users\Lenovo\Desktop\MATH 4890\变量筛选_cleaned.xlsx"
output_dir = r"C:\Users\Lenovo\Desktop\MATH 4890"

os.makedirs(output_dir, exist_ok=True)

# ==========================
# 读取 & 预处理（和原脚本相同）
# ==========================

df = pd.read_excel(file_path)

# 全部转成数值，非数值转 NaN
df = df.apply(pd.to_numeric, errors="coerce")

# 数值列用均值填补，再把残余 NaN 填 0
df = df.fillna(df.select_dtypes(include=[np.number]).mean())
df = df.fillna(0)


# ==========================
# 核心函数：给定 target，输出两个模型的完整排名
# ==========================

def get_rankings_for_target(df, target, top_k=TOP_K):
    """
    返回两个 DataFrame:
    enet_rank_df, xgb_rank_df
    每个都包含: Rank, Feature, Importance, Importance_norm
    """

    # 避免信息泄露：排除其他 target 列
    all_targets = [
        "p_promis_anx_bline",
        "p_promis_dep_bline",
        "a_promis_anx_bline",
        "a_promis_dep_bline",
    ]
    exclude_vars = [t for t in all_targets if t != target]

    X = df.drop(columns=[target] + exclude_vars, errors="ignore")
    y = df[target]

    # 标准化
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # ---------- ElasticNet ----------
    enet = ElasticNetCV(
        alphas=np.logspace(-4, 2, 100),
        l1_ratio=np.linspace(0.1, 1.0, 10),
        cv=5,
        random_state=42,
    ).fit(X_scaled, y)

    en_importance = pd.Series(np.abs(enet.coef_), index=X.columns)
    en_ranked = en_importance.sort_values(ascending=False).head(top_k)

    enet_rank_df = (
        en_ranked.reset_index()
        .rename(columns={"index": "Feature", 0: "Importance"})
    )
    enet_rank_df["Rank"] = np.arange(1, len(enet_rank_df) + 1)
    # 归一化成百分比（相对最大值）
    enet_rank_df["Importance_norm"] = (
        enet_rank_df["Importance"] / enet_rank_df["Importance"].max() * 100
    )

    # ---------- XGBoost ----------
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ).fit(X_scaled, y)

    xgb_importance = pd.Series(xgb.feature_importances_, index=X.columns)
    xgb_ranked = xgb_importance.sort_values(ascending=False).head(top_k)

    xgb_rank_df = (
        xgb_ranked.reset_index()
        .rename(columns={"index": "Feature", 0: "Importance"})
    )
    xgb_rank_df["Rank"] = np.arange(1, len(xgb_rank_df) + 1)
    xgb_rank_df["Importance_norm"] = (
        xgb_rank_df["Importance"] / xgb_rank_df["Importance"].max() * 100
    )

    return enet_rank_df, xgb_rank_df


# ==========================
# 画单个子图的辅助函数
# ==========================

def plot_rank_ax(ax, rank_df, title):
    """
    在给定的 ax 上画横向条形图：
    y 轴：Rank 1-30
    x 轴：Importance_norm (%)
    条旁边写变量名
    """

    # 保证 Rank 1 在最上面
    rank_df = rank_df.sort_values("Rank")

    y_pos = rank_df["Rank"].values
    x_val = rank_df["Importance_norm"].values
    labels = rank_df["Feature"].values

    ax.barh(y_pos, x_val)
    ax.set_ylim(0.5, max(y_pos) + 0.5)
    ax.invert_yaxis()  # Rank 1 在最上面
    ax.set_xlabel("Relative importance (%)")
    ax.set_ylabel("Rank")
    ax.set_title(title, fontsize=11)

    # 在每个条形旁边写变量名
    for y, x, lbl in zip(y_pos, x_val, labels):
        ax.text(
            x + 1,       # 稍微偏右一点
            y,
            lbl,
            va="center",
            fontsize=7,
        )


# ==========================
# 主流程：对两个 target 计算排名、保存 Excel、画 2×2 图
# ==========================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

nice_names_target = {
    "a_promis_anx_bline": "Self-reported anxiety",
    "a_promis_dep_bline": "Self-reported depression",
}
nice_names_model = {
    "ElasticNet": "Elastic Net",
    "XGBoost": "XGBoost",
}

ax_idx = 0

for target in TARGETS_FOR_PLOT:
    print(f"\n===============================")
    print(f"处理目标变量: {target}")
    print(f"===============================")

    enet_rank_df, xgb_rank_df = get_rankings_for_target(df, target, top_k=TOP_K)

    # ---- 保存到 Excel ----
    out_excel = os.path.join(
        output_dir, f"Top30_rank_{target}.xlsx"
    )
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        enet_rank_df.to_excel(writer, sheet_name="ElasticNet", index=False)
        xgb_rank_df.to_excel(writer, sheet_name="XGBoost", index=False)

    print(f"Top30 排名已保存到: {out_excel}")

    # ---- 画图：ElasticNet ----
    ax = axes[ax_idx]
    title = f"{nice_names_model['ElasticNet']} - {nice_names_target[target]}"
    plot_rank_ax(ax, enet_rank_df, title)
    ax_idx += 1

    # ---- 画图：XGBoost ----
    ax = axes[ax_idx]
    title = f"{nice_names_model['XGBoost']} - {nice_names_target[target]}"
    plot_rank_ax(ax, xgb_rank_df, title)
    ax_idx += 1

plt.tight_layout()
fig_path = os.path.join(output_dir, "Top30_rank_plots_anx_dep.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()

print("\n🎉 所有排名图已生成：", fig_path)
print("✅ Excel 排名文件也已保存完毕。")


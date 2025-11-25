import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from xgboost import XGBRegressor

# ==========================
# 参数设置
# ==========================

TARGETS = [
    'p_promis_anx_bline',
    'p_promis_dep_bline',
    'a_promis_anx_bline',
    'a_promis_dep_bline'
]

TOP_K_LIST = [10, 20, 30, 50]  # 想看的 K 值

file_path = r"C:\Users\Lenovo\Desktop\MATH 4890\变量筛选_cleaned.xlsx"
output_dir = r"C:\Users\Lenovo\Desktop\MATH 4890"

# ==========================
# 读取并预处理（完全照你原来的来）
# ==========================

df = pd.read_excel(file_path)

# 全部转成数值，非数值转 NaN
df = df.apply(pd.to_numeric, errors='coerce')

# 数值列用均值填补
df = df.fillna(df.select_dtypes(include=[np.number]).mean())
# 剩余 NaN（极少）填 0
df = df.fillna(0)

# ==========================
# 对单个目标变量：算 Top-K 重合率 + 导出 Top30 重合变量
# ==========================

def compute_overlap_for_target(df, target, top_k_list=TOP_K_LIST, output_dir=output_dir):
    print("\n====================================")
    print(f"🎯 处理目标变量：{target}")
    print("====================================")

    exclude_vars = [t for t in TARGETS if t != target]
    X = df.drop(columns=[target] + exclude_vars)
    y = df[target]

    # 标准化
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # ElasticNet
    print("🔧 训练 ElasticNet（带CV调参）...")
    enet = ElasticNetCV(
        alphas=np.logspace(-4, 2, 100),
        l1_ratio=np.linspace(0.1, 1.0, 10),
        cv=5,
        random_state=42
    ).fit(X_scaled, y)

    enet_importance = pd.Series(np.abs(enet.coef_), index=X.columns)
    enet_ranked = enet_importance.sort_values(ascending=False)
    en_features_sorted = list(enet_ranked.index)

    # XGBoost
    print("🔧 训练 XGBoost（固定参数，用于筛特征）...")
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ).fit(X_scaled, y)

    xgb_importance = pd.Series(xgb.feature_importances_, index=X.columns)
    xgb_ranked = xgb_importance.sort_values(ascending=False)
    xgb_features_sorted = list(xgb_ranked.index)

    # 计算不同 K 下的重合
    records = []
    for K in top_k_list:
        en_topK = set(en_features_sorted[:K])
        xgb_topK = set(xgb_features_sorted[:K])
        inter = en_topK & xgb_topK
        overlap_n = len(inter)
        overlap_rate = overlap_n / K * 100

        records.append({
            "Top_K": K,
            "Overlap_n": overlap_n,
            "Overlap_percent": overlap_rate
        })

        print(f"Top{K}: 重合 {overlap_n} 个 ({overlap_rate:.1f}%)")

        # 如果是 Top30，顺便像原来一样导出交集变量列表
        if K == 30:
            overlap_sorted = sorted(list(inter))
            df_out = pd.DataFrame({"Feature": overlap_sorted})
            safe_target = target.replace(":", "_").replace("/", "_").replace("\\", "_")
            output_path = fr"{output_dir}\Top30_overlap_{safe_target}.xlsx"
            df_out.to_excel(output_path, index=False)
            print(f"💾 Top30 重合变量已保存到：{output_path}")

    # 保存 TopK 重合率表
    df_overlap = pd.DataFrame(records)
    safe_target = target.replace(":", "_").replace("/", "_").replace("\\", "_")
    overlap_table_path = fr"{output_dir}\TopK_overlap_{safe_target}.xlsx"
    df_overlap.to_excel(overlap_table_path, index=False)
    print(f"💾 Top-K 重合率结果已保存到：{overlap_table_path}")

    return df_overlap

# ==========================
# 主循环
# ==========================

all_overlap_tables = {}

for target in TARGETS:
    df_ov = compute_overlap_for_target(df, target)
    all_overlap_tables[target] = df_ov

print("\n🎉 所有目标变量的 Top-K 重合率计算完成！")

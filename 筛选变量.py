import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from xgboost import XGBRegressor

# ==========================
# 参数设置
# ==========================

TOP_K = 30  # 固定使用 Top30
TARGETS = [
    'p_promis_anx_bline',
    'p_promis_dep_bline',
    'a_promis_anx_bline',
    'a_promis_dep_bline'
]

# 数据路径
file_path = r"C:\Users\Lenovo\Desktop\MATH 4890\变量筛选_cleaned.xlsx"
output_dir = r"C:\Users\Lenovo\Desktop\MATH 4890"

# ==========================
# 读取并预处理数据
# ==========================

df = pd.read_excel(file_path)

# 全部转成数值，非数值转 NaN
df = df.apply(pd.to_numeric, errors='coerce')

# 只对数值列用均值填补
df = df.fillna(df.select_dtypes(include=[np.number]).mean())
# 剩余 NaN（极小概率）再填 0
df = df.fillna(0)


# ==========================
# 核心函数：对单个目标变量，提取 Top30 重合变量并导出 Excel
# ==========================

def extract_top30_overlap_for_target(df, target, top_k=TOP_K, output_dir=output_dir):
    print("\n====================================")
    print(f"🎯 处理目标变量：{target}")
    print("====================================")

    # 1. 构建 X / y，排除其他目标变量，避免信息泄漏
    exclude_vars = [t for t in TARGETS if t != target]
    X = df.drop(columns=[target] + exclude_vars)
    y = df[target]

    # 2. 标准化特征
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 3. ElasticNet 模型 & 重要性（绝对系数）
    print("🔧 训练 ElasticNet（带CV调参）...")
    enet = ElasticNetCV(
        alphas=np.logspace(-4, 2, 100),
        l1_ratio=np.linspace(0.1, 1.0, 10),
        cv=5,
        random_state=42
    ).fit(X_scaled, y)

    enet_importance = pd.Series(np.abs(enet.coef_), index=X.columns)
    enet_ranked = enet_importance.sort_values(ascending=False)
    en_topk = set(enet_ranked.head(top_k).index)

    # 4. XGBoost 模型 & 重要性
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
    xgb_topk = set(xgb_ranked.head(top_k).index)

    # 5. 计算重合（Top30 交集）
    overlap = sorted(en_topk.intersection(xgb_topk))

    print(f"📊 ElasticNet Top{top_k} 个变量数：{len(en_topk)}")
    print(f"📊 XGBoost   Top{top_k} 个变量数：{len(xgb_topk)}")
    print(f"✅ Top{top_k} 重合变量数：{len(overlap)}")
    print("✅ 重合变量列表：", overlap)

    # 6. 保存到单独的 Excel：一列，一个变量一行
    df_out = pd.DataFrame({"Feature": overlap})

    # 文件名中替换冒号等不安全字符（一般没有，但以防万一）
    safe_target = target.replace(":", "_").replace("/", "_").replace("\\", "_")

    output_path = fr"{output_dir}\Top30_overlap_{safe_target}.xlsx"
    df_out.to_excel(output_path, index=False)

    print(f"💾 已保存到：{output_path}")

    return overlap


# ==========================
# 主循环：对四个目标变量分别执行
# ==========================

all_overlaps = {}

for target in TARGETS:
    overlap_vars = extract_top30_overlap_for_target(df, target)
    all_overlaps[target] = overlap_vars

print("\n🎉 所有目标变量的 Top30 重合特征提取完毕！")

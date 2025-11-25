import numpy as np
import matplotlib.pyplot as plt

outcomes = ["Anxiety", "Depression"]
topk_labels = ["Top 10", "Top 20", "Top 30", "Top 50"]

anx_rates = [30.0, 30.0, 36.7, 36.0]
dep_rates = [20.0, 40.0, 46.7, 44.0]

top10 = [anx_rates[0], dep_rates[0]]
top20 = [anx_rates[1], dep_rates[1]]
top30 = [anx_rates[2], dep_rates[2]]
top50 = [anx_rates[3], dep_rates[3]]

topk_data = [top10, top20, top30, top50]

x = np.arange(len(outcomes))
width = 0.18

fig, ax = plt.subplots(figsize=(6, 4))

bars_all = []  # 用来存所有 bar 对象

for i, (label, values) in enumerate(zip(topk_labels, topk_data)):
    bars = ax.bar(x + (i - 1.5) * width, values, width, label=label)
    bars_all.append(bars)

# 给每根柱子加数值标签
for bars in bars_all:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,                # 往上抬一点点
            f"{height:.1f}%",          # 显示一位小数 + %
            ha="center",
            va="bottom",
            fontsize=8
        )

ax.set_xticks(x)
ax.set_xticklabels(outcomes)
ax.set_ylabel("Overlap rate (%)")
ax.set_ylim(0, 60)
ax.set_title("Overlap rates of top-K predictors for anxiety and depression")
ax.legend(title="Top-K")

plt.tight_layout()
plt.savefig("overlap_rates_topK_with_labels.png", dpi=300, bbox_inches="tight")
plt.show()

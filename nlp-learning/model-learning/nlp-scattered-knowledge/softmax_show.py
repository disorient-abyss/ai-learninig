import matplotlib.pyplot as plt
import torch


def demo_softmax_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    users = ["User A", "User B", "User C", "User D"]

    # 生成随机评分 logits (均值 0，标准差 2 的正态分布，放大差异度)
    logits = torch.randn(4) * 2
    # 计算 Softmax 概率
    probs = torch.softmax(logits, dim=0)

    # 1. 左侧：条形图展示原始随机得分
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2"]
    bars = axes[0].bar(users, logits.numpy(), color=colors, edgecolor="black")
    axes[0].set_title("Random Raw Scores (Logits)")
    axes[0].set_ylabel("Score")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # 在条形图上标注具体数值
    for bar in bars:
        yval = bar.get_height()
        va = "bottom" if yval >= 0 else "top"
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"{yval:.2f}",
            ha="center",
            va=va,
        )

    # 2. 右侧：饼状图展示概率分布
    # autopct='%.1f%%' 自动显示百分比，startangle 控制起始角度
    axes[1].pie(
        probs.numpy(),
        labels=users,
        autopct="%.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops={"edgecolor": "black", "linewidth": 1},
    )
    axes[1].set_title(f"Softmax Probabilities (Sum={probs.sum():.1f})")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_softmax_distribution()
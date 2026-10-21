import matplotlib.pyplot as plt
import numpy as np

# 设置支持中文的字体（防止乱码）
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用黑体显示中文
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

# 1. 构造平滑的损失函数曲线数据
x = np.linspace(0.2, 9.5, 500)
# 使用多项式和正弦函数拟合出类似截图中“先陡峭、再平坦、过鞍点、入局部极小值、最后下滑”的曲线
y = 10 / (x + 0.3) + 2 * np.sin(x) - 0.5 * x + 4

# 2. 创建画布
plt.figure(figsize=(11, 6), dpi=150)
plt.plot(x, y, color="#A07828", linewidth=3, zorder=1)

# 3. 定义四个关键点（小球的坐标）
balls = [
    {"x": 0.8, "name": "起点", "box": None},
    {"x": 2.5, "name": "平坦区域", "text": "平坦区域下\n降很慢", "color": "#74C640"},
    {
        "x": 4.6,
        "name": "鞍点",
        "text": "鞍点：导数\n为0，但不\n是极值点",
        "color": "#3B9CFF",
    },
    {"x": 7.0, "name": "局部极小值", "text": "局部极小值", "color": "#3B9CFF"},
]

# 4. 绘制小球、虚线和红色投影点
for b in balls:
    # 找到曲线上对应的精确 y 值
    idx = np.abs(x - b["x"]).argmin()
    bx, by = x[idx], y[idx]

    # 画灰色小球
    plt.scatter(bx, by, color="#5A5A5A", s=700, zorder=3)
    # 画向下垂直的黑色虚线
    plt.plot([bx, bx], [0, by], color="black", linestyle=":", linewidth=1.5, zorder=2)
    # 画横轴上的红色投影圆点
    plt.scatter(bx, 0, color="red", s=80, zorder=4)

# 5. 绘制红色的前进箭头（在第一个和第二个点之间）
plt.annotate(
    "",
    xy=(1.8, 0.3),
    xytext=(1.0, 0.3),
    arrowprops={"arrowstyle": "->", "color": "red", "lw": 3},
)

# 6. 添加彩色文本指示框
# 平坦区域黄色公式框
plt.text(
    2.6,
    1.5,
    r"$\frac{\partial L}{\partial w} \approx 0$",
    fontsize=16,
    fontweight="bold",
    bbox={"facecolor": "#FFFFC0", "edgecolor": "none", "pad": 8},
)

# 平坦区域绿色提示框
plt.annotate(
    "平坦区域下\n降很慢",
    xy=(2.5, 3.8),
    xytext=(2.2, 7.5),
    fontsize=13,
    color="white",
    fontweight="bold",
    bbox={"boxstyle": "round,pad=0.5", "facecolor": "#74C640", "edgecolor": "none"},
    arrowprops={"arrowstyle": "->", "color": "blue", "lw": 1.5},
)

# 鞍点蓝色提示框
plt.annotate(
    "鞍点：导数\n为0，但不\n是极值点",
    xy=(4.6, 3.3),
    xytext=(4.6, 7.0),
    fontsize=13,
    color="white",
    fontweight="bold",
    bbox={"boxstyle": "square,pad=0.5", "facecolor": "#3B9CFF", "edgecolor": "none"},
    arrowprops={"arrowstyle": "->", "color": "blue", "lw": 1.5},
)

# 局部极小值蓝色提示框
plt.annotate(
    "局部极小值",
    xy=(7.0, 2.2),
    xytext=(6.3, 4.5),
    fontsize=13,
    color="white",
    fontweight="bold",
    bbox={"boxstyle": "square,pad=0.5", "facecolor": "#3B9CFF", "edgecolor": "none"},
    arrowprops={"arrowstyle": "->", "color": "blue", "lw": 1.5},
)

# 7. 装饰坐标轴
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)

# 画坐标轴箭头
plt.annotate(
    "", xy=(0, 11), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": "black", "lw": 2}
)
plt.annotate(
    "", xy=(10, 0), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": "black", "lw": 2}
)

# 设置坐标轴标签
plt.text(-0.2, 10.5, "Loss", fontsize=15, fontweight="bold", ha="right")
plt.xlabel("The value of the parameter w", fontsize=14, labelpad=10)

# 隐藏刻度
plt.xticks([])
plt.yticks([])
plt.xlim(-0.5, 10.2)
plt.ylim(-0.5, 11.2)

# 显示图像
plt.tight_layout()
plt.show()

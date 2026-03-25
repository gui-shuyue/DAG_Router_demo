import json
import pandas as pd
import matplotlib.pyplot as plt

# 读取你的实验记录 JSON 文件
# 如果脚本不在同级目录，可以直接使用绝对路径，例如:
file_path = r'e:\code\321\DAG_Router_demo-gui\record\l_per_question_metrics_train_50.json'
# file_path = 'l_per_question_metrics_train_50.json'

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取每道题是否正确的列表 (1 表示正确, 0 表示错误)
is_correct_list = [item['is_correct'] for item in data['per_question']]

# 转换为 pandas Series 以便快速计算移动平均和累积平均
results_series = pd.Series(is_correct_list)

# 计算累积准确率 (Cumulative)
cumulative_accuracy = results_series.expanding().mean()

# 计算滚动平均准确率 (Rolling Avg, 窗口大小为20)
# min_periods=1 确保曲线从第1个点开始绘制，与参考图保持一致
rolling_accuracy = results_series.rolling(window=20, min_periods=1).mean()

# 生成 X 轴数据（测试实例数量，从 1 到 500）
x_axis = range(1, len(is_correct_list) + 1)

# 绘制图表
fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

# 绘制累积准确率实线 (使用标准的 Matplotlib 蓝色)
ax.plot(
    x_axis, 
    cumulative_accuracy, 
    label='l_per_question_metrics (Cumulative)', 
    color='#1f77b4', 
    linestyle='-', 
    linewidth=1.8
)

# 绘制滚动平均虚线 (使用相同蓝色，但降低透明度并设置为虚线)
ax.plot(
    x_axis, 
    rolling_accuracy, 
    label='l_per_question_metrics (Rolling Avg, w=20)', 
    color='#1f77b4', 
    linestyle='--', 
    alpha=0.5, 
    linewidth=1.5
)

# 样式设置，复现原图风格
ax.set_title("Online Learning Algorithm Accuracy Trend", fontsize=12)
ax.set_xlabel("Number of Test Instances", fontsize=10)
ax.set_ylabel("Accuracy", fontsize=10)

# 设置 Y 轴范围为稍微超出 0 到 1 的区间，保持视觉留白
ax.set_ylim([-0.05, 1.05])

# 设置浅灰色虚线网格
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='lightgray')

# 将图例放置在中间靠右的位置
ax.legend(loc='center right', fontsize=9)

# 调整布局并展示
plt.tight_layout()
plt.show()
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_correctness_data(json_path):
    """
    从 per_question_metrics JSON 文件中提取 is_correct 序列 (1 或 0)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "per_question" in data:
            records = data["per_question"]
        else:
            records = data 
            
        correctness = [int(item.get("is_correct", 0)) for item in records]
        return np.array(correctness)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return np.array([])

def plot_single_accuracy_trend(json_path, label="LinUCB", window_size=20):
    """
    绘制单条算法的累计准确率与滑动平均准确率趋势图
    """
    correct_seq = load_correctness_data(json_path)
    if len(correct_seq) == 0:
        print("没有找到有效数据，无法绘图。")
        return
        
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    
    # 1. 计算累计准确率和滑动平均
    cumulative_acc = np.cumsum(correct_seq) / np.arange(1, len(correct_seq) + 1)
    rolling_acc = pd.Series(correct_seq).rolling(window=window_size, min_periods=1).mean()
    x_axis = np.arange(1, len(correct_seq) + 1)
    
    # 2. 绘制累计准确率 (实线)
    ax.plot(x_axis, cumulative_acc, 
            label=f'{label} (Cumulative)', 
            color='tab:blue', 
            linestyle='-', 
            linewidth=2)
    
    # 3. 绘制滑动平均准确率 (虚线，使用相同颜色但降低透明度)
    ax.plot(x_axis, rolling_acc, 
            label=f'{label} (Rolling Avg, w={window_size})', 
            color='tab:blue', 
            linestyle='--', 
            linewidth=1.5,
            alpha=0.5)

    # === 设置图表视觉细节 ===
    ax.set_title('Online Learning Algorithm Accuracy Trend', fontsize=14, pad=10)
    ax.set_xlabel('Number of Test Instances', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    
    # 固定 Y 轴范围在 0 到 1 之间
    ax.set_ylim(-0.05, 1.05)
    
    # 浅灰色点状网格线
    ax.grid(True, linestyle=':', color='lightgray', linewidth=1)
    
    # 图例
    ax.legend(loc='center right', framealpha=0.9, fontsize=10)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 指向你刚才跑出来的那个报告文件
    target_file = "record/l_execution_records_train_50.json"
    
    plot_single_accuracy_trend(target_file, label="LinUCB", window_size=20)
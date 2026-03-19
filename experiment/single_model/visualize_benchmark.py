import json
import matplotlib.pyplot as plt
import glob
import os

def load_all_reports():
    report_files = glob.glob("report_*.json")
    
    if not report_files:
        print("❌ 找不到任何 report_*.json 文件，请确保评测脚本已生成报告。")
        return []
    
    print(f"✅ 发现 {len(report_files)} 个模型报告文件，正在解析...")
    
    aggregated_data = []
    for file in report_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            aggregated_data.append({
                'model': data['model_name'].split('/')[-1], 
                'accuracy': data['summary']['accuracy'] * 100,
                'model_cost': data['summary']['avg_model_cost_usd'],
                'judge_cost': data['summary']['avg_judge_cost_usd']
            })
            
    return aggregated_data

def plot_accuracy(data):
    data_sorted = sorted(data, key=lambda x: x['accuracy'], reverse=False)
    models = [item['model'] for item in data_sorted]
    accuracies = [item['accuracy'] for item in data_sorted]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(models, accuracies, color='#4C72B0')
    plt.title('Model Accuracy Leaderboard', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Accuracy (%)', fontsize=12)
    plt.xlim(0, max(accuracies) * 1.15 if max(accuracies) > 0 else 100)

    for bar in bars:
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                 f'{bar.get_width():.1f}%', 
                 va='center', ha='left', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('01_benchmark_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📸 已生成: 01_benchmark_accuracy.png (准确率)")

def plot_pure_model_cost(data):
    data_sorted = sorted(data, key=lambda x: x['model_cost'], reverse=False)
    models = [item['model'] for item in data_sorted]
    model_costs = [item['model_cost'] for item in data_sorted]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(models, model_costs, color='#55A868')
    plt.title('Average Model Generation Cost per Question', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Model Cost (USD)', fontsize=12)
    plt.xlim(0, max(model_costs) * 1.2 if max(model_costs) > 0 else 0.1)

    for bar in bars:
        plt.text(bar.get_width() + (max(model_costs)*0.01), bar.get_y() + bar.get_height()/2, 
                 f'${bar.get_width():.6f}', 
                 va='center', ha='left', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('02_benchmark_model_cost.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📸 已生成: 02_benchmark_model_cost.png (纯模型生成花费)")

def plot_tradeoff(data):
    models = [item['model'] for item in data]
    accuracies = [item['accuracy'] for item in data]
    costs = [item['model_cost'] for item in data]

    plt.figure(figsize=(10, 7))
    plt.scatter(costs, accuracies, color='#8172B3', s=200, alpha=0.8, edgecolors='black', linewidth=1.5)
    
    plt.title('Accuracy vs. Pure Model Cost Trade-off', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Average Model Cost per Question (USD)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, model in enumerate(models):
        plt.annotate(model, 
                     (costs[i], accuracies[i]),
                     xytext=(10, -5), textcoords='offset points',
                     fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

    plt.text(0.05, 0.95, 'Optimal Zone\n(High Acc, Low Model Cost)', 
             transform=plt.gca().transAxes, fontsize=12, color='green', 
             fontweight='bold', va='top', ha='left',
             bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="green", alpha=0.6))

    plt.tight_layout()
    plt.savefig('03_benchmark_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📸 已生成: 03_benchmark_tradeoff.png (真实业务性价比)")

def plot_judge_cost(data):
    data_sorted = sorted(data, key=lambda x: x['judge_cost'], reverse=False)
    models = [item['model'] for item in data_sorted]
    judge_costs = [item['judge_cost'] for item in data_sorted]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(models, judge_costs, color='#C44E52')
    plt.title('Average Evaluation (Judge) Cost per Question', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Judge Cost (USD)', fontsize=12)
    plt.xlim(0, max(judge_costs) * 1.2 if max(judge_costs) > 0 else 0.1)

    for bar in bars:
        plt.text(bar.get_width() + (max(judge_costs)*0.01), bar.get_y() + bar.get_height()/2, 
                 f'${bar.get_width():.6f}', 
                 va='center', ha='left', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('04_benchmark_judge_cost.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📸 已生成: 04_benchmark_judge_cost.png (裁判评估花费)")

def generate_table_image(data):
    # ================= 核心修改：行列完全交换 =================
    # 按准确率从高到低排序
    data_sorted = sorted(data, key=lambda x: x['accuracy'], reverse=True)
    
    # 构建表头 (列名)
    headers = ["Model", "Accuracy (%)", "Model Cost ($)", "Judge Cost ($)"]
    
    # 构建数据行 (每一行是一个模型的数据)
    cell_text = []
    for item in data_sorted:
        row = [
            item['model'],
            f"{item['accuracy']:.2f}%",
            f"${item['model_cost']:.6f}",
            f"${item['judge_cost']:.6f}"
        ]
        cell_text.append(row)

    # 动态计算画布尺寸：宽度固定 12 左右，高度随模型数量增加
    fig_width = 12
    fig_height = max(3.5, len(data_sorted) * 0.7 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight')
    ax.axis('off')

    # 绘制表格
    table = ax.table(cellText=cell_text, colLabels=headers, loc='center', cellLoc='center')
    
    # 美化表格
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5) # 调整单元格宽高比

    # 单元格颜色配置
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            # 第一行 (表头)
            cell.set_text_props(weight='bold', color='white', fontsize=13)
            cell.set_facecolor('#4C72B0')
        elif col == 0:
            # 第一列 (模型名)
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#EAEAF2')
        else:
            # 普通单元格
            cell.set_edgecolor('#B0B0B0')

    plt.title('Benchmark Results Data Table', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('05_benchmark_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📸 已生成: 05_benchmark_table.png (行列翻转标准排版表格)")

if __name__ == "__main__":
    print("⏳ 正在扫描文件夹并生成 5 张高清分析图表...")
    report_data = load_all_reports()
    if report_data:
        plot_accuracy(report_data)
        plot_pure_model_cost(report_data)
        plot_tradeoff(report_data)
        plot_judge_cost(report_data)
        generate_table_image(report_data)
        
        print("🎉 所有 5 张可视化图片已全部渲染完毕！快去文件夹里看吧！")
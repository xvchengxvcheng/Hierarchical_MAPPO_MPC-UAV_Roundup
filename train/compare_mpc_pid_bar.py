"""
对比MPC和PID方法的成功率与平均步数柱状图
Compare success rate and average steps between MPC and PID methods using bar charts
"""

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体支持 / Set Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_collision_record_file(filepath):
    """
    解析撞毁记录文件（支持JSON格式） / Parse collision record file (supports JSON format)
    Args:
        filepath: 记录文件路径 / Record file path
    Returns:
        episodes: episode编号列表 / List of episode numbers
        collided: 是否撞毁列表（布尔值） / List of collision flags (boolean)
        success: 是否成功列表（布尔值） / List of success flags (boolean)
        steps: 步数列表 / List of steps
    """
    episodes = []
    collided = []
    success = []
    steps = []
    
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return episodes, collided, success, steps
    
    try:
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext == '.json':
            # JSON格式（JSONL：每行一个JSON对象） / JSON format (JSONL: one JSON object per line)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        episodes.append(int(record.get('episode', 0)))
                        collided.append(bool(record.get('collided', False)))
                        success.append(bool(record.get('success', False)))
                        steps.append(int(record.get('steps', 0)))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse JSON at line {line_num} in {filepath}: {e}")
                        continue
        else:
            print(f"Warning: Unsupported file format: {filepath}")
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
    
    return episodes, collided, success, steps


def calculate_metrics(episodes, collided, success, steps):
    """
    计算成功率和平均步数，以及标准差 / Calculate success rate, average steps, and standard deviations
    Returns:
        success_rate: 成功率（百分比） / Success rate (percentage)
        avg_steps: 平均步数 / Average steps
        success_rate_std: 成功率标准差（百分比） / Success rate standard deviation (percentage)
        avg_steps_std: 平均步数标准差 / Average steps standard deviation
    """
    if len(episodes) == 0:
        return None, None, None, None
    
    success_rate = sum(success) / len(success) * 100
    # 平均步数不考虑撞毁回合：只统计未撞毁（collided=False）的 episodes
    # 注意：仍然不按 success 过滤
    collided_arr = np.asarray(collided, dtype=bool)
    all_steps = np.asarray(steps, dtype=float)
    valid_steps = all_steps[~collided_arr]
    if len(valid_steps) == 0:
        # 如果全部回合都撞毁，则无法计算“未撞毁平均步数”
        avg_steps = float('nan')
        avg_steps_std = float('nan')
    else:
        avg_steps = float(np.mean(valid_steps))
        avg_steps_std = float(np.std(valid_steps, ddof=1)) if len(valid_steps) > 1 else 0.0
    
    # 计算标准差
    # 对于成功率，使用二项分布的标准差：sqrt(p(1-p)/n) * 100
    p = sum(success) / len(success)
    success_rate_std = np.sqrt(p * (1 - p) / len(success)) * 100
    
    # avg_steps_std 已在上面按未撞毁 valid_steps 计算
    
    return success_rate, avg_steps, success_rate_std, avg_steps_std


def plot_success_rate_comparison(mpc_data, pid_data, run_order=None, save_path=None, show_plot=True):
    """
    绘制成功率对比柱状图（带误差条） / Plot success rate comparison bar chart with error bars
    Args:
        mpc_data: MPC数据字典，格式为 {run_id: {'success_rate': float, 'success_rate_std': float}}
        pid_data: PID数据字典，格式同上
        run_order: run_id的顺序列表，如果为None则使用sorted排序 / Order of run_ids, if None use sorted
        save_path: 保存路径（可选） / Save path (optional)
        show_plot: 是否显示图表 / Whether to show plot
    """
    # 提取数据，使用指定的顺序
    if run_order is None:
        mpc_runs = sorted(mpc_data.keys())
        pid_runs = sorted(pid_data.keys())
    else:
        # 只保留在数据中存在的run_id，并保持顺序
        mpc_runs = [run for run in run_order if run in mpc_data]
        pid_runs = [run for run in run_order if run in pid_data]
    
    mpc_success_rates = [mpc_data[run]['success_rate'] for run in mpc_runs]
    mpc_success_stds = [mpc_data[run]['success_rate_std'] for run in mpc_runs]
    
    pid_success_rates = [pid_data[run]['success_rate'] for run in pid_runs]
    pid_success_stds = [pid_data[run]['success_rate_std'] for run in pid_runs]
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 设置x轴位置
    x = np.arange(len(mpc_runs))
    width = 0.35  # 柱状图宽度
    
    # 绘制柱状图（不绘制成功率的方差/误差条）
    bars_mpc = ax.bar(x - width/2, mpc_success_rates, width,
                      label='MPC-MARL', color='#37BEDC', alpha=0.8)
    bars_pid = ax.bar(x + width/2, pid_success_rates, width,
                      label='PID-MARL', color='#FE6500', alpha=0.8)
    
    # ax.set_xlabel('Run ID', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=26)
    # ax.set_title('Success Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    # ax.set_xticklabels(mpc_runs)
    ax.set_xticklabels(['S1', 'S2', 'S3', 'S4'], fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3, axis='y')
    max_val = max(max(mpc_success_rates), max(pid_success_rates))
    ax.set_ylim([0, max_val * 1.15])
    
    # 在柱状图上添加数值标签
    for bars in [bars_mpc, bars_pid]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=18)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Success rate comparison chart saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_avg_steps_comparison(mpc_data, pid_data, run_order=None, save_path=None, show_plot=True):
    """
    绘制平均步数对比柱状图（带误差条） / Plot average steps comparison bar chart with error bars
    Args:
        mpc_data: MPC数据字典，格式为 {run_id: {'avg_steps': float, 'avg_steps_std': float}}
        pid_data: PID数据字典，格式同上
        run_order: run_id的顺序列表，如果为None则使用sorted排序 / Order of run_ids, if None use sorted
        save_path: 保存路径（可选） / Save path (optional)
        show_plot: 是否显示图表 / Whether to show plot
    """
    # 提取数据，使用指定的顺序
    if run_order is None:
        mpc_runs = sorted(mpc_data.keys())
        pid_runs = sorted(pid_data.keys())
    else:
        # 只保留在数据中存在的run_id，并保持顺序
        mpc_runs = [run for run in run_order if run in mpc_data]
        pid_runs = [run for run in run_order if run in pid_data]
    
    mpc_avg_steps = [mpc_data[run]['avg_steps'] for run in mpc_runs]
    mpc_steps_stds = [mpc_data[run]['avg_steps_std'] for run in mpc_runs]
    
    pid_avg_steps = [pid_data[run]['avg_steps'] for run in pid_runs]
    pid_steps_stds = [pid_data[run]['avg_steps_std'] for run in pid_runs]
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 设置x轴位置
    x = np.arange(len(mpc_runs))
    width = 0.35  # 柱状图宽度
    
    # 绘制柱状图（带误差条）
    bars_mpc = ax.bar(x - width/2, mpc_avg_steps, width, 
                      yerr=mpc_steps_stds, capsize=5,
                      label='MPC-MARL', color='#37BEDC', alpha=0.8,
                      error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    bars_pid = ax.bar(x + width/2, pid_avg_steps, width, 
                      yerr=pid_steps_stds, capsize=5,
                      label='PID-MARL', color='#FE6500', alpha=0.8,
                      error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    
    # ax.set_xlabel('Run ID', fontsize=12)
    ax.set_ylabel('Average Steps', fontsize=26)
    # ax.set_title('Average Steps Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    # ax.set_xticklabels(mpc_runs)
    ax.set_xticklabels(['S1', 'S2', 'S3', 'S4'], fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3, axis='y')
    max_val = max(max(mpc_avg_steps), max(pid_avg_steps))
    max_err = max(max(mpc_steps_stds), max(pid_steps_stds))
    ax.set_ylim([0, (max_val + max_err) * 1.15])
    
    # 在柱状图上添加数值标签
    for bars in [bars_mpc, bars_pid]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=18)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Average steps comparison chart saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    # 文件列表
    base_dir = "results_collision"
    
    mpc_files = [
        f"{base_dir}/collision_records_MPC_76.json",
        f"{base_dir}/collision_records_MPC_75.json",
        f"{base_dir}/collision_records_MPC_77.json",
        f"{base_dir}/collision_records_MPC_78.json",
    ]
    
    pid_files = [
        f"{base_dir}/collision_records_PID_76.json",
        f"{base_dir}/collision_records_PID_75.json",
        f"{base_dir}/collision_records_PID_77.json",
        f"{base_dir}/collision_records_PID_78.json",
    ]
    
    # 读取MPC数据，记录顺序
    mpc_data = {}
    mpc_run_order = []  # 记录run_id的顺序
    print("Reading MPC files...")
    for filepath in mpc_files:
        # 从文件名中提取数字部分作为 run_id（例如：collision_records_MPC_71.json -> 71）
        filename = os.path.basename(filepath)
        match = re.search(r'_(\d+)\.json$', filename)
        if match:
            run_id = match.group(1)
        else:
            # 如果无法提取数字，使用整个文件名（去掉前缀和后缀）
            run_id = filename.replace('collision_records_', '').replace('.json', '')
        print(f"  Reading: {filepath}")
        episodes, collided, success, steps = parse_collision_record_file(filepath)
        
        if len(episodes) == 0:
            print(f"    Warning: No data found in {filepath}")
            continue
        
        success_rate, avg_steps, success_rate_std, avg_steps_std = calculate_metrics(episodes, collided, success, steps)
        mpc_data[run_id] = {
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'success_rate_std': success_rate_std,
            'avg_steps_std': avg_steps_std
        }
        mpc_run_order.append(run_id)  # 记录顺序
        print(f"    Loaded {len(episodes)} episodes, Success rate: {success_rate:.2f}% ± {success_rate_std:.2f}%, Avg steps: {avg_steps:.2f} ± {avg_steps_std:.2f}")
    
    # 读取PID数据，记录顺序
    pid_data = {}
    pid_run_order = []  # 记录run_id的顺序
    print("\nReading PID files...")
    for filepath in pid_files:
        # 从文件名中提取数字部分作为 run_id（例如：collision_records_PID_71.json -> 71）
        filename = os.path.basename(filepath)
        match = re.search(r'_(\d+)\.json$', filename)
        if match:
            run_id = match.group(1)
        else:
            # 如果无法提取数字，使用整个文件名（去掉前缀和后缀）
            run_id = filename.replace('collision_records_', '').replace('.json', '')
        print(f"  Reading: {filepath}")
        episodes, collided, success, steps = parse_collision_record_file(filepath)
        
        if len(episodes) == 0:
            print(f"    Warning: No data found in {filepath}")
            continue
        
        success_rate, avg_steps, success_rate_std, avg_steps_std = calculate_metrics(episodes, collided, success, steps)
        pid_data[run_id] = {
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'success_rate_std': success_rate_std,
            'avg_steps_std': avg_steps_std
        }
        pid_run_order.append(run_id)  # 记录顺序
        print(f"    Loaded {len(episodes)} episodes, Success rate: {success_rate:.2f}% ± {success_rate_std:.2f}%, Avg steps: {avg_steps:.2f} ± {avg_steps_std:.2f}")
    
    # 检查数据
    if len(mpc_data) == 0 or len(pid_data) == 0:
        print("\nError: No valid data found. Please check file paths.")
        return
    
    # 确保MPC和PID的run_id一致，但保持文件输入的顺序
    mpc_runs_set = set(mpc_data.keys())
    pid_runs_set = set(pid_data.keys())
    common_runs_set = mpc_runs_set & pid_runs_set
    
    if len(common_runs_set) == 0:
        print("\nError: No common run IDs between MPC and PID data.")
        return
    
    # 保持MPC文件输入的顺序，只保留共同的run_id
    common_runs = [run for run in mpc_run_order if run in common_runs_set]
    
    # 只保留共同的run_id
    mpc_data_filtered = {run: mpc_data[run] for run in common_runs}
    pid_data_filtered = {run: pid_data[run] for run in common_runs}
    
    print(f"\nFound {len(common_runs)} common runs (in order): {common_runs}")
    
    # 分别绘制并保存两张对比图，使用文件输入的顺序
    success_rate_path = f"{base_dir}/mpc_pid_success_rate_comparison.png"
    avg_steps_path = f"{base_dir}/mpc_pid_avg_steps_comparison.png"
    
    plot_success_rate_comparison(mpc_data_filtered, pid_data_filtered, 
                                 run_order=common_runs,
                                 save_path=success_rate_path, show_plot=True)
    plot_avg_steps_comparison(mpc_data_filtered, pid_data_filtered, 
                              run_order=common_runs,
                              save_path=avg_steps_path, show_plot=True)
    
    # 打印汇总统计
    print("\n=== Summary Statistics ===")
    print(f"MPC - Average Success Rate: {np.mean([mpc_data_filtered[r]['success_rate'] for r in common_runs]):.2f}%")
    print(f"PID - Average Success Rate: {np.mean([pid_data_filtered[r]['success_rate'] for r in common_runs]):.2f}%")
    print(f"MPC - Average Steps: {np.mean([mpc_data_filtered[r]['avg_steps'] for r in common_runs]):.2f}")
    print(f"PID - Average Steps: {np.mean([pid_data_filtered[r]['avg_steps'] for r in common_runs]):.2f}")


if __name__ == "__main__":
    main()

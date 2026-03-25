"""
读取并绘制撞毁记录数据的脚本 / Script to read and plot collision record data
支持多种统计图表 / Supports multiple statistical plots
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse

# 设置中文字体支持 / Set Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 25,
    'axes.titlesize': 25,
    'axes.labelsize': 30,
    'xtick.labelsize': 23,
    'ytick.labelsize': 23,
    'legend.fontsize': 21,
})


def parse_collision_record_file(filepath):
    """
    解析撞毁记录文件（支持JSON和旧文本格式） / Parse collision record file (supports JSON and old text format)
    Args:
        filepath: 记录文件路径 / Record file path
    Returns:
        episodes: episode编号列表 / List of episode numbers
        collided: 是否撞毁列表（布尔值） / List of collision flags (boolean)
        success: 是否成功列表（布尔值） / List of success flags (boolean)
        steps: 步数列表 / List of steps
    """
    import json
    
    episodes = []
    collided = []
    success = []
    steps = []
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return episodes, collided, success, steps
    
    try:
        # 根据文件扩展名判断格式 / Determine format by file extension
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
                        print(f"Warning: Failed to parse JSON at line {line_num}: {e}")
                        continue
        else:
            # 旧文本格式（向后兼容） / Old text format (backward compatible)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析格式: Episode {episode}: collided={True/False}, success={True/False}, steps={步数}
                    # Parse format: Episode {episode}: collided={True/False}, success={True/False}, steps={steps}
                    match = re.match(r'Episode (\d+): collided=(True|False), success=(True|False), steps=(\d+)', line)
                    if match:
                        ep = int(match.group(1))
                        col = match.group(2) == 'True'
                        suc = match.group(3) == 'True'
                        stp = int(match.group(4))
                        
                        episodes.append(ep)
                        collided.append(col)
                        success.append(suc)
                        steps.append(stp)
    except Exception as e:
        print(f"Error reading file: {e}")
    
    return episodes, collided, success, steps


def calculate_statistics(episodes, collided, success, steps):
    """
    计算统计信息 / Calculate statistics
    Returns:
        stats: 统计字典 / Statistics dictionary
    """
    if len(episodes) == 0:
        return None
    
    total_episodes = len(episodes)
    collision_count = sum(collided)
    success_count = sum(success)
    collision_rate = collision_count / total_episodes * 100
    success_rate = success_count / total_episodes * 100
    
    avg_steps = np.mean(steps)
    std_steps = np.std(steps)
    min_steps = np.min(steps)
    max_steps = np.max(steps)

    stats = {
        'total_episodes': total_episodes,
        'collision_count': collision_count,
        'success_count': success_count,
        'collision_rate': collision_rate,
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'std_steps': std_steps,
        'min_steps': min_steps,
        'max_steps': max_steps,
    }
    
    return stats


def plot_collision_statistics(episodes, collided, success, steps, save_path=None, show_plot=True):
    """
    绘制撞毁统计图表 / Plot collision statistics
    Args:
        episodes: episode编号列表 / List of episode numbers
        collided: 是否撞毁列表 / List of collision flags
        success: 是否成功列表 / List of success flags
        steps: 步数列表 / List of steps
        save_path: 保存路径（可选） / Save path (optional)
        show_plot: 是否显示图表 / Whether to show plot
    """
    if len(episodes) == 0:
        print("No data to plot")
        return
    
    # 计算统计信息 / Calculate statistics
    stats = calculate_statistics(episodes, collided, success, steps)
    
    # 创建图表 / Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 撞毁率随时间变化 / Collision rate over time
    ax1 = plt.subplot(2, 3, 1)
    window_size = min(50, len(episodes) // 10) if len(episodes) > 10 else len(episodes)
    if window_size > 1:
        collision_rates = []
        episode_windows = []
        for i in range(0, len(episodes) - window_size + 1, max(1, window_size // 5)):
            window_collided = collided[i:i+window_size]
            rate = sum(window_collided) / len(window_collided) * 100
            collision_rates.append(rate)
            episode_windows.append(episodes[i + window_size // 2])
        
        min_collision_rate = min(collision_rates)
        ax1.plot(episode_windows, collision_rates, 'r-', linewidth=2, label='Collision Rate')
        ax1.axhline(y=min_collision_rate, color='b', linestyle='--', alpha=0.5, label=f'Min: {min_collision_rate:.2f}%')
    else:
        collision_rates_single = [100 if c else 0 for c in collided]
        min_collision_rate = min(collision_rates_single)
        ax1.plot(episodes, collision_rates_single, 'r.', markersize=2, alpha=0.5)
        ax1.axhline(y=min_collision_rate, color='b', linestyle='--', alpha=0.5, label=f'Min: {min_collision_rate:.2f}%')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Collision Rate (%)')
    ax1.set_title('Collision Rate Over Episodes')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. 成功率随时间变化 / Success rate over time
    ax2 = plt.subplot(2, 3, 2)
    if window_size > 1:
        success_rates = []
        episode_windows = []
        for i in range(0, len(episodes) - window_size + 1, max(1, window_size // 5)):
            window_success = success[i:i+window_size]
            rate = sum(window_success) / len(window_success) * 100
            success_rates.append(rate)
            episode_windows.append(episodes[i + window_size // 2])
        
        max_success_rate = max(success_rates)
        ax2.plot(episode_windows, success_rates, 'g-', linewidth=2, label='Success Rate')
        ax2.axhline(y=max_success_rate, color='b', linestyle='--', alpha=0.5, label=f'Max: {max_success_rate:.2f}%')
    else:
        success_rates_single = [100 if s else 0 for s in success]
        max_success_rate = max(success_rates_single)
        ax2.plot(episodes, success_rates_single, 'g.', markersize=2, alpha=0.5)
        ax2.axhline(y=max_success_rate, color='b', linestyle='--', alpha=0.5, label=f'Max: {max_success_rate:.2f}%')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Success Rate Over Episodes')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. 步数分布 / Steps distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(steps, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax3.axvline(x=stats['avg_steps'], color='r', linestyle='--', linewidth=2, label=f'Mean: {stats["avg_steps"]:.1f}')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Number of Episodes')
    ax3.set_title('Steps Distribution')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. 步数随时间变化 / Steps over time
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(episodes, steps, c=[1 if c else 0 for c in collided], cmap='RdYlGn', 
                s=10, alpha=0.6, edgecolors='black', linewidths=0.5)
    ax4.axhline(y=stats['avg_steps'], color='b', linestyle='--', alpha=0.5, label=f'Average: {stats["avg_steps"]:.1f}')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Steps')
    ax4.set_title('Steps per Episode (Red=Collided, Green=No Collision)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. 总体统计饼图 / Overall statistics pie chart
    ax5 = plt.subplot(2, 3, 5)
    labels = ['Collided', 'No Collision']
    sizes = [stats['collision_count'], stats['total_episodes'] - stats['collision_count']]
    colors = ['red', 'green']
    explode = (0.1, 0)
    ax5.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax5.set_title('Collision Statistics')
    
    # 6. 统计信息文本 / Statistics text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = f"""
    Statistics Summary:
    
    Total Episodes: {stats['total_episodes']}
    Collision Count: {stats['collision_count']}
    Success Count: {stats['success_count']}
    
    Collision Rate: {stats['collision_rate']:.2f}%
    Success Rate: {stats['success_rate']:.2f}%
    
    Average Steps: {stats['avg_steps']:.1f}
    Std Steps: {stats['std_steps']:.1f}
    Min Steps: {stats['min_steps']}
    Max Steps: {stats['max_steps']}
    """
    ax6.text(0.1, 0.5, stats_text, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_simple_collision_rate(episodes, collided, save_path=None, show_plot=True):
    """
    绘制简单的撞毁率曲线 / Plot simple collision rate curve
    """
    if len(episodes) == 0:
        print("No data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 计算滑动窗口撞毁率 / Calculate sliding window collision rate
    window_size = min(50, len(episodes) // 10) if len(episodes) > 10 else len(episodes)
    if window_size > 1:
        collision_rates = []
        episode_windows = []
        for i in range(0, len(episodes) - window_size + 1, max(1, window_size // 5)):
            window_collided = collided[i:i+window_size]
            rate = sum(window_collided) / len(window_collided) * 100
            collision_rates.append(rate)
            episode_windows.append(episodes[i + window_size // 2])
        
        ax.plot(episode_windows, collision_rates, 'r-', linewidth=2, label='Collision Rate (%)')
    else:
        # 如果数据太少，直接显示每个episode的状态 / If too little data, show each episode's status
        ax.plot(episodes, [100 if c else 0 for c in collided], 'r.', markersize=3, alpha=0.6, label='Collided')
    
    total_collision_rate = sum(collided) / len(collided) * 100
    ax.axhline(y=total_collision_rate, color='b', linestyle='--', linewidth=2, 
               label=f'Average: {total_collision_rate:.2f}%')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Collision Rate (%)')
    ax.set_title('Collision Rate Over Episodes')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_compare_simple_collision_rate(episodes1, collided1,
                                       episodes2, collided2,
                                       label1='File 1', label2='File 2',
                                       save_path=None, show_plot=True):
    """
    对比两个实验的简单撞毁率曲线 / Compare simple collision rate curves of two runs
    """
    if len(episodes1) == 0 or len(episodes2) == 0:
        print("Not enough data to compare (one of the files is empty)")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # 为两个文件分别计算滑动窗口撞毁率 / Sliding-window collision rate for each file
    def compute_curve(episodes, collided):
        window_size = min(400, len(episodes) // 10) if len(episodes) > 10 else len(episodes)
        if window_size > 1:
            collision_rates = []
            episode_windows = []
            for i in range(0, len(episodes) - window_size + 1, max(1, window_size // 5)):
                window_collided = collided[i:i+window_size]
                rate = sum(window_collided) / len(window_collided) * 100
                collision_rates.append(rate)
                episode_windows.append(episodes[i + window_size // 2])
            return episode_windows, collision_rates
        else:
            # 数据太少时直接返回每个 episode 的状态 / For very few data points, just return per-episode status
            return episodes, [100 if c else 0 for c in collided]

    x1, y1 = compute_curve(episodes1, collided1)
    x2, y2 = compute_curve(episodes2, collided2)

    ax.plot(x1, y1, 'r-', linewidth=2, label=f'MPC-MARL')
    ax.plot(x2, y2, 'g-', linewidth=2, label=f'PID-MARL')

    total_rate1 = sum(collided1) / len(collided1) * 100
    total_rate2 = sum(collided2) / len(collided2) * 100
    ax.axhline(y=total_rate1, color='r', linestyle='--', alpha=0.5,
               label=f'MPC-MARL: mean = {total_rate1:.2f}%')
    ax.axhline(y=total_rate2, color='g', linestyle='--', alpha=0.5,
               label=f'PID-MARL: mean = {total_rate2:.2f}%')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Collision Rate (%)')
    # ax.set_title('Collision Rate Comparison', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_compare_success_rate(episodes1, success1,
                              episodes2, success2,
                              label1='File 1', label2='File 2',
                              save_path=None, show_plot=True):
    """
    对比两个实验的成功率随时间变化 / Compare success rate over episodes for two runs
    """
    if len(episodes1) == 0 or len(episodes2) == 0:
        print("Not enough data to compare success rate (one of the files is empty)")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    def compute_curve(episodes, success):
        window_size = min(400, len(episodes) // 10) if len(episodes) > 10 else len(episodes)
        if window_size > 1:
            success_rates = []
            episode_windows = []
            for i in range(0, len(episodes) - window_size + 1, max(1, window_size // 5)):
                window_success = success[i:i + window_size]
                rate = sum(window_success) / len(window_success) * 100
                success_rates.append(rate)
                episode_windows.append(episodes[i + window_size // 2])
            return episode_windows, success_rates, window_size
        else:
            return episodes, [100 if s else 0 for s in success], 1

    x1, y1, w1 = compute_curve(episodes1, success1)
    x2, y2, w2 = compute_curve(episodes2, success2)

    ax.plot(x1, y1, 'g-', linewidth=2, label='MPC-MARL')
    ax.plot(x2, y2, 'b-', linewidth=2, label='PID-MARL')

    final_rate1 = sum(success1[-w1:]) / len(success1[-w1:]) * 100
    final_rate2 = sum(success2[-w2:]) / len(success2[-w2:]) * 100
    ax.axhline(y=final_rate1, color='g', linestyle='--', alpha=0.5,
               label=f'MPC-MARL: final = {final_rate1:.2f}%')
    ax.axhline(y=final_rate2, color='b', linestyle='--', alpha=0.5,
               label=f'PID-MARL: final = {final_rate2:.2f}%')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    # ax.set_title('Success Rate Comparison', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Success comparison plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_compare_steps_distribution(steps1, steps2,
                                    label1='File 1', label2='File 2',
                                    save_path=None, show_plot=True):
    """
    对比两个实验的步数分布直方图 / Compare steps distribution (histogram) of two runs
    """
    if len(steps1) == 0 or len(steps2) == 0:
        print("Not enough data to compare steps distribution (one of the files is empty)")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # 统一 bin 范围，便于对比 / Use common bins for fair comparison
    all_steps = np.concatenate([np.array(steps1), np.array(steps2)])
    bins = np.histogram_bin_edges(all_steps, bins=30)

    ax.hist(steps1, bins=bins, alpha=0.6, edgecolor='black',
            label=f'MPC-MARL', color='skyblue')
    ax.hist(steps2, bins=bins, alpha=0.6, edgecolor='black',
            label=f'PID-MARL', color='orange')

    ax.set_xlabel('Steps')
    ax.set_ylabel('Number of Episodes')
    # ax.set_title('Steps Distribution Comparison', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Steps distribution comparison plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot collision records')
    parser.add_argument('--file', type=str,
                        # default= "results_collision/collision_records_66.json",
                        default="results_collision/collision_records_mpc_5.json",
                       help='Path to collision record file (e.g., results_collision/collision_records_65.txt)')
    parser.add_argument('--file2', type=str,
                        # default=None,
                        default="results_collision/collision_records_66.json",
                       help='Second collision record file to compare (JSON or TXT)')
    parser.add_argument('--test_time', type=int, default=None,
                       help='Test time number to automatically find file (e.g., 52)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save the plot (e.g., collision_analysis.png)')
    parser.add_argument('--simple', action='store_true',
                       help='Plot simple collision rate only')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not show plot (only save)')
    
    args = parser.parse_args()
    
    # 确定第一个文件路径 / Determine first file path
    if args.file:
        filepath = args.file
    elif args.test_time:
        # 优先查找JSON格式，如果没有则查找旧格式 / Prefer JSON format, fallback to old format
        json_path = f"results_collision/collision_records_{args.test_time}.json"
        txt_path = f"results_collision/collision_records_{args.test_time}.txt"
        if os.path.exists(json_path):
            filepath = json_path
        elif os.path.exists(txt_path):
            filepath = txt_path
            print(f"Warning: Using old text format file. Consider migrating to JSON format.")
        else:
            print(f"Error: No collision record file found for test_time={args.test_time}")
            return
    else:
        # 查找最新的记录文件（优先JSON格式） / Find latest record file (prefer JSON format)
        results_dir = "results_collision"
        if os.path.exists(results_dir):
            # 先查找JSON文件 / First find JSON files
            json_files = [f for f in os.listdir(results_dir) 
                         if f.startswith('collision_records_') and f.endswith('.json')]
            txt_files = [f for f in os.listdir(results_dir) 
                        if f.startswith('collision_records_') and f.endswith('.txt')]
            
            files = json_files + txt_files  # JSON优先 / JSON first
            
            if files:
                # 按修改时间排序，取最新的 / Sort by modification time, take latest
                files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
                filepath = os.path.join(results_dir, files[0])
                print(f"Using latest file: {filepath}")
            else:
                print("Error: No collision record files found in results_collision/")
                return
        else:
            print("Error: results_collision/ directory not found")
            return
    
    # 读取第一个文件数据 / Read first file data
    print(f"Reading file: {filepath}")
    episodes, collided, success, steps = parse_collision_record_file(filepath)
    
    if len(episodes) == 0:
        print("No data found in file")
        return
    
    print(f"Loaded {len(episodes)} episodes (file 1)")
    print(f"Collision rate (file 1): {sum(collided) / len(collided) * 100:.2f}%")
    print(f"Success rate   (file 1): {sum(success) / len(success) * 100:.2f}%")
    
    # 绘制图表 / Plot
    show_plot = not args.no_show
    save_path = args.save
    
    # 如果未指定保存路径，则根据原始文件名自动生成 PNG 路径
    # When save path is not provided, generate a PNG path based on original file name
    # 如果提供了第二个文件，则进行对比绘图 / If a second file is provided, do comparison plotting
    if args.file2:
        filepath2 = args.file2
        print(f"Reading second file: {filepath2}")
        episodes2, collided2, success2, steps2 = parse_collision_record_file(filepath2)

        if len(episodes2) == 0:
            print("No data found in second file, falling back to single-file plotting")
            args.file2 = None  # 标记失败，下面走单文件逻辑 / mark as failed and fall back
        else:
            print(f"Loaded {len(episodes2)} episodes (file 2)")
            print(f"Collision rate (file 2): {sum(collided2) / len(collided2) * 100:.2f}%")
            print(f"Success rate   (file 2): {sum(success2) / len(success2) * 100:.2f}%")

            base1, _ = os.path.splitext(filepath)
            base2 = os.path.splitext(os.path.basename(filepath2))[0]

            # 输出文件名前缀 / Output filename prefix
            if save_path:
                out_base, _ = os.path.splitext(save_path)
            else:
                out_base = f"{base1}_VS_{base2}"

            label1 = os.path.basename(filepath)
            label2 = os.path.basename(filepath2)

            if args.simple:
                # 简单模式：只画撞毁率对比 / Simple mode: only collision-rate comparison
                collision_path = out_base + '_collision_cmp.png'
                plot_compare_simple_collision_rate(
                    episodes, collided,
                    episodes2, collided2,
                    label1=label1, label2=label2,
                    save_path=collision_path, show_plot=show_plot
                )
            else:
                # 非 simple 模式：分别画三张图，并单独保存
                collision_path = out_base + '_collision_cmp.png'
                success_path = out_base + '_success_cmp.png'
                steps_path = out_base + '_steps_cmp.png'

                plot_compare_simple_collision_rate(
                    episodes, collided,
                    episodes2, collided2,
                    label1=label1, label2=label2,
                    save_path=collision_path, show_plot=show_plot
                )
                plot_compare_success_rate(
                    episodes, success,
                    episodes2, success2,
                    label1=label1, label2=label2,
                    save_path=success_path, show_plot=show_plot
                )
                plot_compare_steps_distribution(
                    steps, steps2,
                    label1=label1, label2=label2,
                    save_path=steps_path, show_plot=show_plot
                )
            return

    # 单文件绘图逻辑 / Single-file plotting logic
    if args.simple:
        if not save_path:
            base, _ = os.path.splitext(filepath)
            save_path = base + '_simple.png'
        plot_simple_collision_rate(episodes, collided, save_path=save_path, show_plot=show_plot)
    else:
        if not save_path:
            base, _ = os.path.splitext(filepath)
            save_path = base + '_analysis.png'
        plot_collision_statistics(episodes, collided, success, steps, save_path=save_path, show_plot=show_plot)


if __name__ == "__main__":
    main()


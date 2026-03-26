"""
从 TensorBoard 日志中读取并绘制训练奖励曲线
Read and plot training reward curves from TensorBoard logs
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    USE_TF = False
except ImportError:
    try:
        from tensorflow.python.summary.summary_iterator import summary_iterator
        import tensorflow as tf
        USE_TF = True
        EventAccumulator = None
    except ImportError:
        print("请安装 tensorboard 或 tensorflow: pip install tensorboard")
        exit(1)


def load_tensorboard_data(log_dir, scalar_tags=None):
    """
    从 TensorBoard 日志目录加载标量数据
    
    Args:
        log_dir: TensorBoard 日志目录路径（相对或绝对路径）
        scalar_tags: 要加载的标量标签列表，如果为 None 则加载所有
    
    Returns:
        dict: {tag: {'steps': [...], 'values': [...]}}
    """
    # 转换为 Path 对象并解析为绝对路径
    log_dir = Path(log_dir).resolve()
    
    if not log_dir.exists():
        print(f"错误: 日志目录不存在: {log_dir}")
        print(f"  当前工作目录: {os.getcwd()}")
        # 尝试从项目根目录查找
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        alt_path = project_root / log_dir if not log_dir.is_absolute() else log_dir
        if alt_path.exists() and alt_path != log_dir:
            print(f"  尝试使用: {alt_path}")
            log_dir = alt_path
        else:
            return {}
    
    if not log_dir.is_dir():
        print(f"错误: 路径不是目录: {log_dir}")
        return {}
    
    # 查找所有 events 文件（包括嵌套目录中的）
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        print(f"警告: 在 {log_dir} 中未找到 TensorBoard 事件文件")
        return {}
    
    print(f"找到 {len(event_files)} 个事件文件在: {log_dir}")
    # 检查文件大小
    for event_file in event_files:
        file_size = event_file.stat().st_size
        print(f"  文件: {event_file.relative_to(log_dir)}, 大小: {file_size} 字节")
        if file_size == 0:
            print(f"    警告: 文件为空!")
    
    data = {}
    
    if USE_TF and EventAccumulator is None:
        # 使用 tensorflow 直接读取
        all_tags_found = set()
        for event_file in event_files:
            try:
                for event in tf.compat.v1.train.summary_iterator(str(event_file)):
                    if event.summary:
                        for value in event.summary.value:
                            tag = value.tag
                            all_tags_found.add(tag)
                            # 检查是否匹配请求的标签（包括嵌套标签）
                            should_load = False
                            if scalar_tags is None:
                                should_load = True
                            else:
                                # 直接匹配
                                if tag in scalar_tags:
                                    should_load = True
                                # 嵌套标签匹配（如 average_episode_rewards/average_episode_rewards）
                                elif any(f"{req_tag}/{req_tag}" == tag for req_tag in scalar_tags):
                                    should_load = True
                                # 模糊匹配（包含该标签名）
                                elif any(req_tag in tag for req_tag in scalar_tags):
                                    should_load = True
                            
                            if should_load:
                                # 使用原始标签名作为键（如果是嵌套标签，提取主标签名）
                                if '/' in tag and tag.count('/') == 1:
                                    parts = tag.split('/')
                                    if parts[0] == parts[1]:
                                        key = parts[0]  # 嵌套标签如 a/a -> a
                                    else:
                                        key = tag
                                else:
                                    key = tag
                                
                                if key not in data:
                                    data[key] = {'steps': [], 'values': []}
                                data[key]['steps'].append(event.step)
                                data[key]['values'].append(value.simple_value)
            except Exception as e:
                print(f"读取 {event_file} 时出错: {e}")
        print(f"可用的标量标签: {sorted(all_tags_found)}")
        if scalar_tags:
            found_tags = [tag for tag in scalar_tags if tag in all_tags_found or f"{tag}/{tag}" in all_tags_found]
            missing_tags = [tag for tag in scalar_tags if tag not in found_tags]
            if missing_tags:
                print(f"警告: 以下标签未找到: {missing_tags}")
            print(f"找到的标签: {found_tags}")
    else:
        # 使用 EventAccumulator（推荐方法）
        try:
            # 首先尝试从根目录读取
            ea = EventAccumulator(str(log_dir))
            ea.Reload()
            
            # 显示所有可用的标签类型
            all_tags = ea.Tags()
            print(f"所有可用的标签类型: {list(all_tags.keys())}")
            for tag_type, tags in all_tags.items():
                if tags:
                    print(f"  {tag_type}: {tags}")
            
            available_tags = all_tags.get('scalars', [])
            print(f"\n根目录可用的标量标签: {available_tags}")
            
            # 如果根目录没有找到标签，尝试从嵌套目录读取
            if not available_tags or (scalar_tags and not any(tag in available_tags or f"{tag}/{tag}" in available_tags for tag in scalar_tags)):
                print("\n根目录未找到所需标签，尝试从嵌套目录读取...")
                # 查找所有嵌套的事件文件目录
                nested_dirs = []
                for event_file in event_files:
                    # 获取事件文件所在的目录
                    event_dir = event_file.parent
                    # 检查是否是嵌套结构（目录名和父目录名相同）
                    # 例如: logs/average_episode_rewards/average_episode_rewards/
                    if event_dir.parent != log_dir and event_dir.name == event_dir.parent.name:
                        nested_dirs.append(event_dir)
                    # 也检查直接子目录（如果事件文件直接在子目录中）
                    elif event_dir.parent == log_dir and event_dir != log_dir:
                        # 检查子目录中是否有同名子目录
                        subdir = event_dir / event_dir.name
                        if subdir.exists() and (subdir / event_file.name).exists():
                            nested_dirs.append(subdir)
                
                # 去重并排序
                nested_dirs = sorted(list(set(nested_dirs)), key=lambda x: x.name)
                print(f"找到 {len(nested_dirs)} 个嵌套目录")
                
                # 从每个嵌套目录读取数据
                for nested_dir in nested_dirs:
                    tag_name = nested_dir.parent.name if nested_dir.parent != log_dir else nested_dir.name  # 标签名
                    print(f"  读取嵌套目录: {nested_dir.relative_to(log_dir)} (标签: {tag_name})")
                    
                    try:
                        ea_nested = EventAccumulator(str(nested_dir))
                        ea_nested.Reload()
                        nested_tags = ea_nested.Tags().get('scalars', [])
                        print(f"    找到标签: {nested_tags}")
                        
                        # 如果指定了要加载的标签，检查是否匹配
                        should_load = scalar_tags is None
                        if scalar_tags:
                            # 检查标签名是否匹配（支持部分匹配，如 'average_episode_rewards' 匹配 'eval_average_episode_rewards'）
                            for req_tag in scalar_tags:
                                if req_tag in tag_name or tag_name in req_tag:
                                    should_load = True
                                    break
                        
                        if should_load:
                            # 读取所有标量数据
                            for tag in nested_tags:
                                try:
                                    scalar_events = ea_nested.Scalars(tag)
                                    steps = [e.step for e in scalar_events]
                                    values = [e.value for e in scalar_events]
                                    # 使用标签名作为键
                                    key = tag_name
                                    if key not in data:
                                        data[key] = {'steps': [], 'values': []}
                                    data[key]['steps'].extend(steps)
                                    data[key]['values'].extend(values)
                                    print(f"    加载标签 '{tag}' -> '{key}': {len(steps)} 个数据点")
                                except Exception as e:
                                    print(f"    加载标签 '{tag}' 时出错: {e}")
                    except Exception as e:
                        print(f"  读取嵌套目录 {nested_dir} 时出错: {e}")
                        import traceback
                        traceback.print_exc()
            
            # 如果根目录有标签，也尝试加载
            if available_tags:
                tags_to_load = []
                if scalar_tags:
                    for requested_tag in scalar_tags:
                        if requested_tag in available_tags:
                            tags_to_load.append(requested_tag)
                        else:
                            # 尝试查找嵌套标签（如 average_episode_rewards/average_episode_rewards）
                            nested_tag = f"{requested_tag}/{requested_tag}"
                            if nested_tag in available_tags:
                                print(f"  找到嵌套标签: {nested_tag} (替代 {requested_tag})")
                                tags_to_load.append(nested_tag)
                            else:
                                # 尝试模糊匹配（包含该标签名的所有标签）
                                matching_tags = [tag for tag in available_tags if requested_tag in tag]
                                if matching_tags:
                                    print(f"  找到匹配标签: {matching_tags} (替代 {requested_tag})")
                                    tags_to_load.extend(matching_tags)
                else:
                    tags_to_load = available_tags
                
                for tag in tags_to_load:
                    try:
                        scalar_events = ea.Scalars(tag)
                        steps = [e.step for e in scalar_events]
                        values = [e.value for e in scalar_events]
                        # 使用原始标签名作为键（如果是嵌套标签，提取主标签名）
                        if '/' in tag and tag.count('/') == 1:
                            parts = tag.split('/')
                            if parts[0] == parts[1]:
                                key = parts[0]  # 嵌套标签如 a/a -> a
                            else:
                                key = tag
                        else:
                            key = tag
                        
                        if key not in data:
                            data[key] = {'steps': [], 'values': []}
                        data[key]['steps'].extend(steps)
                        data[key]['values'].extend(values)
                        print(f"  加载标签 '{tag}' -> '{key}': {len(steps)} 个数据点")
                    except Exception as e:
                        print(f"  加载标签 '{tag}' 时出错: {e}")
                        
        except Exception as e:
            print(f"读取日志时出错: {e}")
            import traceback
            traceback.print_exc()
    
    return data


def plot_single_metric(data_dict, metric_name, metric_key, title, ylabel, 
                       save_path=None, show_plot=True, smooth_window=1,
                       tick_fontsize=20):
    """
    绘制单个指标的对比图
    
    Args:
        data_dict: 数据字典，格式为 {run_name: {tag: {'steps': [...], 'values': [...]}}}
        metric_name: 指标名称（用于文件名）
        metric_key: 数据字典中的键名
        title: 图表标题
        ylabel: Y轴标签
        save_path: 保存图片的路径，如果为 None 则不保存
        show_plot: 是否显示图表
        smooth_window: 平滑窗口大小（移动平均）
        tick_fontsize: 坐标轴刻度数字大小（同时用于科学计数法偏移文字，如 1e6）
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    
    # 检查是否有该指标的数据
    has_data = any(metric_key in data for data in data_dict.values())
    if not has_data:
        print(f"警告: 未找到 {metric_key} 数据，跳过绘制")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    # colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
    color_list = ['#FB6542', '#375E97']
    
    
    for idx, (run_name, data) in enumerate(data_dict.items()):
        if metric_key not in data:
            continue
        
        color = color_list[idx % len(color_list)]
        steps = np.array(data[metric_key]['steps'])
        values = np.array(data[metric_key]['values'])
        
        # 平滑处理并增加均值±标准差阴影带
        if smooth_window > 1 and len(values) > smooth_window:
            smoothed = np.convolve(values, np.ones(smooth_window)/smooth_window, mode='valid')
            smoothed_steps = steps[smooth_window-1:]
            # 计算滚动标准差 / rolling std for shaded band
            rolling_std = []
            for i in range(len(values) - smooth_window + 1):
                window_vals = values[i:i+smooth_window]
                rolling_std.append(np.std(window_vals))
            rolling_std = np.array(rolling_std)
            ax.plot(smoothed_steps, smoothed, label=run_name, 
                    color=color, linewidth=2, alpha=0.8)
            ax.fill_between(smoothed_steps,
                            smoothed - rolling_std,
                            smoothed + rolling_std,
                            color=color, alpha=0.2)
        else:
            ax.plot(steps, values, label=run_name, 
                    color=color, linewidth=2, alpha=0.8)
            if len(values) > 1:
                std_val = np.std(values)
                ax.fill_between(steps,
                                values - std_val,
                                values + std_val,
                                color=color, alpha=0.2)
    
    ax.set_xlabel('Training Steps', fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.xaxis.get_offset_text().set_fontsize(tick_fontsize)
    ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)
    # ax.set_title(title, fontsize=14, fontweight='bold')
    if ax.lines:
        ax.legend(loc='best', fontsize=21)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_rewards(data_dict, save_path=None, show_plot=True, smooth_window=1, 
                 save_separate=False, save_dir=None, tick_fontsize=20):
    """
    绘制奖励曲线和损失曲线
    
    Args:
        data_dict: 数据字典，格式为 {run_name: {tag: {'steps': [...], 'values': [...]}}}
        save_path: 保存图片的路径，如果为 None 则不保存（用于保存组合图）
        show_plot: 是否显示图表
        smooth_window: 平滑窗口大小（移动平均）
        save_separate: 是否单独保存每个指标的图片
        save_dir: 单独保存时的目录路径，如果为 None 则使用 save_path 的目录
        tick_fontsize: 坐标轴刻度数字大小（同时用于科学计数法偏移文字，如 1e6）
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    
    # 如果启用单独保存，先保存各个指标的对比图
    if save_separate:
        if save_dir is None:
            if save_path:
                save_dir = Path(save_path).parent
            else:
                save_dir = Path('.')
        else:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存训练奖励对比图
        if any('average_episode_rewards' in data for data in data_dict.values()):
            train_reward_path = save_dir / 'training_rewards_comparison.png'
            plot_single_metric(data_dict, 'training_rewards', 'average_episode_rewards',
                             '训练奖励曲线对比 (Training Rewards Comparison)',
                             'Average Episode Reward',
                             save_path=str(train_reward_path), show_plot=False,
                             smooth_window=smooth_window, tick_fontsize=tick_fontsize)
        
        # 保存 Policy Loss 对比图
        if any('policy_loss' in data for data in data_dict.values()):
            policy_loss_path = save_dir / 'policy_loss_comparison.png'
            plot_single_metric(data_dict, 'policy_loss', 'policy_loss',
                             'Policy Loss 对比 (Policy Loss Comparison)',
                             'Policy Loss',
                             save_path=str(policy_loss_path), show_plot=False,
                             smooth_window=smooth_window, tick_fontsize=tick_fontsize)
        
        # 保存 Value Loss 对比图
        if any('value_loss' in data for data in data_dict.values()):
            value_loss_path = save_dir / 'value_loss_comparison.png'
            plot_single_metric(data_dict, 'value_loss', 'value_loss',
                             'Value Loss 对比 (Value Loss Comparison)',
                             'Value Loss',
                             save_path=str(value_loss_path), show_plot=False,
                             smooth_window=smooth_window, tick_fontsize=tick_fontsize)
    
    # 检查是否有 policy_loss 和 value_loss 数据
    has_policy_loss = any('policy_loss' in data for data in data_dict.values())
    has_value_loss = any('value_loss' in data for data in data_dict.values())
    
    # 计算子图数量
    num_subplots = 2  # 训练奖励和评估奖励
    if has_policy_loss:
        num_subplots += 1
    if has_value_loss:
        num_subplots += 1
    
    # 根据子图数量调整图表大小
    fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 4 + 4 * num_subplots))
    if num_subplots == 2:
        axes = [axes[0], axes[1]]
    
    # 训练奖励
    ax1 = axes[0]
    # 评估奖励
    ax2 = axes[1]
    # Policy Loss 和 Value Loss（如果存在）
    ax_idx = 2
    ax3 = axes[ax_idx] if has_policy_loss else None
    if has_policy_loss:
        ax_idx += 1
    ax4 = axes[ax_idx] if has_value_loss else None
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
    
    for idx, (run_name, data) in enumerate(data_dict.items()):
        color = colors[idx]
        
        # 绘制训练奖励
        if 'average_episode_rewards' in data:
            steps = np.array(data['average_episode_rewards']['steps'])
            values = np.array(data['average_episode_rewards']['values'])
            
            # 平滑处理
            if smooth_window > 1 and len(values) > smooth_window:
                smoothed = np.convolve(values, np.ones(smooth_window)/smooth_window, mode='valid')
                smoothed_steps = steps[smooth_window-1:]
                # 计算滚动标准差 / rolling std for shaded area
                rolling_std = []
                for i in range(len(values) - smooth_window + 1):
                    window_vals = values[i:i+smooth_window]
                    rolling_std.append(np.std(window_vals))
                rolling_std = np.array(rolling_std)
                ax1.plot(smoothed_steps, smoothed, label=f'{run_name} (训练)', 
                        color=color, linewidth=2, alpha=0.8)
                ax1.fill_between(smoothed_steps,
                                 smoothed - rolling_std,
                                 smoothed + rolling_std,
                                 color=color, alpha=0.2)
            else:
                ax1.plot(steps, values, label=f'{run_name} (训练)', 
                        color=color, linewidth=2, alpha=0.8)
                if len(values) > 1:
                    std_val = np.std(values)
                    ax1.fill_between(steps,
                                     values - std_val,
                                     values + std_val,
                                     color=color, alpha=0.2)
        
        # 绘制评估奖励
        if 'eval_average_episode_rewards' in data:
            steps = np.array(data['eval_average_episode_rewards']['steps'])
            values = np.array(data['eval_average_episode_rewards']['values'])
            
            if smooth_window > 1 and len(values) > smooth_window:
                smoothed = np.convolve(values, np.ones(smooth_window)/smooth_window, mode='valid')
                smoothed_steps = steps[smooth_window-1:]
                # 计算滚动标准差 / rolling std for shaded area
                rolling_std = []
                for i in range(len(values) - smooth_window + 1):
                    window_vals = values[i:i+smooth_window]
                    rolling_std.append(np.std(window_vals))
                rolling_std = np.array(rolling_std)
                ax2.plot(smoothed_steps, smoothed, label=f'{run_name} (评估)', 
                        color=color, linewidth=2, alpha=0.8, linestyle='--')
                ax2.fill_between(smoothed_steps,
                                 smoothed - rolling_std,
                                 smoothed + rolling_std,
                                 color=color, alpha=0.2)
            else:
                ax2.plot(steps, values, label=f'{run_name} (评估)', 
                        color=color, linewidth=2, alpha=0.8, linestyle='--')
                if len(values) > 1:
                    std_val = np.std(values)
                    ax2.fill_between(steps,
                                     values - std_val,
                                     values + std_val,
                                     color=color, alpha=0.2)
        
        # 绘制 Policy Loss
        if 'policy_loss' in data and ax3 is not None:
            steps = np.array(data['policy_loss']['steps'])
            values = np.array(data['policy_loss']['values'])
            
            if smooth_window > 1 and len(values) > smooth_window:
                smoothed = np.convolve(values, np.ones(smooth_window)/smooth_window, mode='valid')
                smoothed_steps = steps[smooth_window-1:]
                ax3.plot(smoothed_steps, smoothed, label=f'{run_name}', 
                        color=color, linewidth=2, alpha=0.8)
            else:
                ax3.plot(steps, values, label=f'{run_name}', 
                        color=color, linewidth=2, alpha=0.8)
        
        # 绘制 Value Loss
        if 'value_loss' in data and ax4 is not None:
            steps = np.array(data['value_loss']['steps'])
            values = np.array(data['value_loss']['values'])
            
            if smooth_window > 1 and len(values) > smooth_window:
                smoothed = np.convolve(values, np.ones(smooth_window)/smooth_window, mode='valid')
                smoothed_steps = steps[smooth_window-1:]
                ax4.plot(smoothed_steps, smoothed, label=f'{run_name}', 
                        color=color, linewidth=2, alpha=0.8)
            else:
                ax4.plot(steps, values, label=f'{run_name}', 
                        color=color, linewidth=2, alpha=0.8)
    
    # 设置训练奖励图
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('平均回合奖励 (Average Episode Reward)', fontsize=12)
    ax1.set_title('训练奖励曲线 (Training Rewards)', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=tick_fontsize)
    ax1.xaxis.get_offset_text().set_fontsize(tick_fontsize)
    ax1.yaxis.get_offset_text().set_fontsize(tick_fontsize)
    if ax1.lines:
        ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 设置评估奖励图
    ax2.set_xlabel('Training Steps', fontsize=40)
    ax2.set_ylabel('平均回合奖励 (Average Episode Reward)', fontsize=50)
    ax2.set_title('评估奖励曲线 (Evaluation Rewards)', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=tick_fontsize)
    ax2.xaxis.get_offset_text().set_fontsize(tick_fontsize)
    ax2.yaxis.get_offset_text().set_fontsize(tick_fontsize)
    if ax2.lines:
        ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 设置 Policy Loss 图
    if ax3 is not None:
        ax3.set_xlabel('Training Steps', fontsize=12)
        ax3.set_ylabel('Policy Loss', fontsize=12)
        ax3.set_title('Policy Loss 曲线', fontsize=14, fontweight='bold')
        ax3.tick_params(axis='both', labelsize=tick_fontsize)
        ax3.xaxis.get_offset_text().set_fontsize(tick_fontsize)
        ax3.yaxis.get_offset_text().set_fontsize(tick_fontsize)
        if ax3.lines:
            ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
    
    # 设置 Value Loss 图
    if ax4 is not None:
        ax4.set_xlabel('Training Steps', fontsize=12)
        ax4.set_ylabel('Value Loss', fontsize=12)
        ax4.set_title('Value Loss 曲线', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='both', labelsize=tick_fontsize)
        ax4.xaxis.get_offset_text().set_fontsize(tick_fontsize)
        ax4.yaxis.get_offset_text().set_fontsize(tick_fontsize)
        if ax4.lines:
            ax4.legend(loc='best', fontsize=10)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"组合图表已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='从 TensorBoard 日志绘制奖励曲线')
    parser.add_argument('--log_dir', type=str, 
                       default=None,
                       help='TensorBoard 日志目录路径（单个运行）')
    parser.add_argument('--log_dirs', type=str, nargs='+', default=None,
                       help='多个 TensorBoard 日志目录路径（对比多个运行）')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='结果根目录，会自动查找所有运行')
    parser.add_argument('--run_names', type=str, nargs='+', default=None,
                       help='要对比的运行名称（如 run1, run2），如果为 None 则查找所有')
    parser.add_argument('--save_path', type=str, default=None,
                       help='保存组合图片的路径（如 reward_curve.png），如果为 None 则不保存组合图')
    parser.add_argument('--save_dir', type=str, default="comparison_results",
                       help='单独保存对比图片的目录，如果为 None 则使用当前目录')
    parser.add_argument('--smooth', type=int, default=10,
                       help='平滑窗口大小（移动平均），默认 10')
    parser.add_argument('--tick_fontsize', type=int, default=23,
                       help='坐标轴刻度数字大小')
    parser.add_argument('--no_show', action='store_true',
                       help='不显示图表，仅保存')
    parser.add_argument('--separate', action='store_true', default=None,
                       help='单独保存每个指标的对比图片（训练奖励、Policy Loss、Value Loss）')
    parser.add_argument('--no_separate', action='store_true',
                       help='不单独保存，仅保存组合图')
    
    args = parser.parse_args()
    
    data_dict = {}
    
    # 情况1: 指定多个日志目录（优先级高于 log_dir）
    if args.log_dirs:
        for log_dir in args.log_dirs:
            log_path = Path(log_dir)
            if not log_path.exists():
                # 尝试从项目根目录查找
                script_dir = Path(__file__).parent
                project_root = script_dir.parent
                alt_path = project_root / log_dir
                if alt_path.exists():
                    log_path = alt_path
                else:
                    print(f"警告: 跳过不存在的目录: {log_dir}")
                    continue

            data = load_tensorboard_data(
                log_path,
                scalar_tags=[
                    'average_episode_rewards',
                    'eval_average_episode_rewards',
                    'policy_loss',
                    'value_loss'
                ])
            if data:
                # 先用路径名作为临时 key，后面再根据需要改成 mpc/pid
                if log_path.name == 'logs':
                    run_key = log_path.parent.name
                else:
                    run_key = log_path.name
                data_dict[run_key] = data
    
    # 情况2: 指定单个日志目录
    elif args.log_dir:
        log_dir = Path(args.log_dir)
        if not log_dir.exists():
            # 尝试从项目根目录查找
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            alt_path = project_root / args.log_dir
            if alt_path.exists():
                log_dir = alt_path
                print(f"使用项目根目录下的路径: {log_dir}")
            else:
                print(f"错误: 日志目录不存在: {args.log_dir}")
                print(f"  尝试的路径: {log_dir}")
                print(f"  尝试的路径: {alt_path}")
                return
        
        data = load_tensorboard_data(log_dir, 
                                     scalar_tags=['average_episode_rewards', 
                                                 'eval_average_episode_rewards',
                                                 'policy_loss',
                                                 'value_loss'])
        if data:
            data_dict[log_dir.name if log_dir.name == 'logs' else log_dir.parent.name] = data
    
    # 情况3: 从 results 目录自动查找
    else:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        results_dir = project_root / args.results_dir if not Path(args.results_dir).is_absolute() else Path(args.results_dir)
        
        if not results_dir.exists():
            print(f"错误: 结果目录不存在: {results_dir}")
            return
        
        # 查找所有运行目录
        if args.run_names:
            run_dirs = []
            for run_name in args.run_names:
                # 查找匹配的运行目录
                pattern = f"**/{run_name}/logs"
                matches = list(results_dir.glob(pattern))
                run_dirs.extend(matches)
        else:
            # 查找所有 logs 目录
            run_dirs = list(results_dir.glob("**/logs"))
        
        if not run_dirs:
            print(f"未找到任何日志目录，请检查路径: {results_dir}")
            return
        
        print(f"找到 {len(run_dirs)} 个运行:")
        for log_dir in run_dirs:
            print(f"  - {log_dir}")
            data = load_tensorboard_data(log_dir,
                                        scalar_tags=['average_episode_rewards',
                                                    'eval_average_episode_rewards',
                                                    'policy_loss',
                                                    'value_loss'])
            if data:
                # 使用运行路径作为名称（如 MyEnv/MyEnv/mappo/check/run1）
                run_name = str(log_dir.parent.relative_to(results_dir))
                data_dict[run_name] = data
    
    if not data_dict:
        print("错误: 未找到任何数据")
        return
    
    # 如果是通过 --log_dirs 精确指定了两个运行，则将 legend label 固定为 mpc、pid
    # 按照命令行中 log_dirs 的顺序，第一个为 mpc，第二个为 pid
    if args.log_dirs is not None and len(data_dict) == 2:
        original_items = list(data_dict.items())
        alias_labels = ['PID-MARL', 'MPC-MARL']
        aliased_dict = {}
        for alias, (_, run_data) in zip(alias_labels, original_items):
            aliased_dict[alias] = run_data
        data_dict = aliased_dict

    print(f"\n成功加载 {len(data_dict)} 个运行的数据")
    for run_name, data in data_dict.items():
        print(f"\n{run_name}:")
        for tag in data.keys():
            print(f"  - {tag}: {len(data[tag]['steps'])} 个数据点")
    
    # 确定是否单独保存：如果有多个运行且未明确禁用，则默认启用单独保存
    is_comparison = len(data_dict) > 1
    save_separate = False
    if args.separate:
        save_separate = True
    elif args.no_separate:
        save_separate = False
    elif is_comparison:
        # 多个运行时，默认启用单独保存
        save_separate = True
        print(f"\n检测到 {len(data_dict)} 个运行，将单独保存每个指标的对比图片")
    
    # 确定保存目录
    save_dir = args.save_dir
    if save_separate and save_dir is None:
        if args.save_path:
            save_dir = Path(args.save_path).parent
        else:
            save_dir = Path('.')
    
    # 绘制图表
    plot_rewards(data_dict, save_path=args.save_path, 
                show_plot=not args.no_show, 
                smooth_window=args.smooth,
                save_separate=save_separate,
                save_dir=save_dir,
                tick_fontsize=args.tick_fontsize)


if __name__ == '__main__':
    main()


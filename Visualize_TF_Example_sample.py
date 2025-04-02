# Data location. Please edit.

# A tfrecord containing tf.Example protos as downloaded from the Waymo dataset
# webpage.

# Replace this path with your own tfrecords.
FILENAME = '/home/heihuhu/PycharmProjects/waymo/data/uncompressed_tf_example_training_training_tfexample.tfrecord-00001-of-01000'

import math
import os
import uuid
import time

from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
from IPython.display import HTML
import itertools
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2

# 定义道路图样本数量（Waymo数据集固定值）
num_map_samples = 30000

# ------------------- 道路图特征定义 -------------------
roadgraph_features = {
    'roadgraph_samples/dir': tf.io.FixedLenFeature(
        [num_map_samples, 3], tf.float32, default_value=None
    ),
    # 道路样本的方向向量（x,y,z）
    'roadgraph_samples/id': tf.io.FixedLenFeature(
        [num_map_samples, 1], tf.int64, default_value=None
    ),
    # 道路元素的唯一标识符
    'roadgraph_samples/type': tf.io.FixedLenFeature(
        [num_map_samples, 1], tf.int64, default_value=None
    ),
    # 道路类型（如车道、人行道等）
    'roadgraph_samples/valid': tf.io.FixedLenFeature(
        [num_map_samples, 1], tf.int64, default_value=None
    ),
    # 标记该样本是否有效（1=有效，0=无效）
    'roadgraph_samples/xyz': tf.io.FixedLenFeature(
        [num_map_samples, 3], tf.float32, default_value=None
    )
    # 道路元素的3D坐标位置
}

# ------------------- 代理状态特征定义 -------------------
state_features = {
    'state/id': tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    # 代理的唯一标识符（最多128个代理）
    'state/type': tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    # 代理类型（如车辆、行人等）
    'state/is_sdc': tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    # 标记是否为自动驾驶车辆（SDC）
    'state/tracks_to_predict': tf.io.FixedLenFeature(
        [128], tf.int64, default_value=None
    ),
    # 需要预测的轨迹标记（挑战赛中用于选择目标）

    # 当前状态（时间步0）
    'state/current/bbox_yaw': tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    # 当前时刻的包围盒方向角
    'state/current/height': tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/length': tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/width': tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    # 当前尺寸参数（高度/长度/宽度）
    'state/current/x': tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/y': tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/z': tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    # 当前3D位置坐标
    'state/current/velocity_x': tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_y': tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/vel_yaw': tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    # 当前速度向量（x/y方向+方向角速度）
    'state/current/timestamp_micros': tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    # 时间戳（微秒）
    'state/current/valid': tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    # 当前状态有效性标记

    # 过去状态（过去10个时间步）
    'state/past/bbox_yaw': tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/height': tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/length': tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/width': tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/x': tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/y': tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/z': tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_x': tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_y': tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/vel_yaw': tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/timestamp_micros': tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/valid': tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    # 过去状态历史记录（10个时间步）

    # 未来状态（未来80个时间步）
    'state/future/bbox_yaw': tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/height': tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/length': tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/width': tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/x': tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/y': tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/z': tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_x': tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_y': tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/vel_yaw': tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/timestamp_micros': tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/valid': tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    # 未来状态预测目标（80个时间步，挑战赛中需要预测）
}

# ------------------- 交通灯状态特征定义 -------------------
traffic_light_features = {
    'traffic_light_state/current/state': tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    # 当前交通灯状态（16个灯，每个灯状态：红/黄/绿）
    'traffic_light_state/current/valid': tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    # 当前交通灯有效性标记
    'traffic_light_state/current/x': tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/y': tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/z': tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    # 当前交通灯位置坐标

    'traffic_light_state/past/state': tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/valid': tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/x': tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/y': tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/z': tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    # 过去10个时间步的交通灯状态历史
}

# 合并所有特征定义，用于解析Waymo数据集的TFRecord文件
features_description = {}
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)

dataset = tf.data.TFRecordDataset(
    FILENAME,          # 替换为实际的TFRecord文件路径
    compression_type='' # 数据未压缩时设为空，若为gzip则设为'GZIP'
)

#   - as_numpy_iterator() 返回可迭代对象，每个元素是原始字节数据
#   - next() 获取第一个样本的二进制数据（bytes类型）
data = next(dataset.as_numpy_iterator())

#   - 将原始字节数据解析为结构化张量
#   - features_description 定义了数据解析的特征模板（如前所述）
#   - 输出为字典，键为特征名，值为对应张量
parsed = tf.io.parse_single_example(
    data,               # 输入的原始字节数据
    features_description  # 特征解析模板（包含所有预定义的特征定义）
)


def create_figure_and_axes(size_pixels):
    """初始化唯一画布和坐标轴系统（适用于多代理轨迹可视化）

    参数:
      size_pixels -- 输出图像的像素尺寸（正方形）

    返回:
      fig -- matplotlib figure对象
      ax -- matplotlib axes对象
    """
    # 生成唯一标识符避免图像缓存冲突
    fig, ax = plt.subplots(1, 1, num=uuid.uuid4())

    # 设置图像分辨率参数
    dpi = 100  # 标准打印分辨率
    size_inches = size_pixels / dpi  # 计算物理尺寸
    fig.set_size_inches([size_inches, size_inches])  # 设置画布尺寸
    fig.set_dpi(dpi)  # 设置分辨率
    fig.set_facecolor('white')  # 白色背景
    ax.set_facecolor('white')  # 坐标区背景

    # 设置坐标轴标签颜色
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='y', colors='black')

    fig.set_tight_layout(True)  # 紧凑布局
    ax.grid(False)  # 关闭网格线
    return fig, ax


def fig_canvas_image(fig):
    """将matplotlib画布转换为RGB图像数组[1](@ref)

    参数:
      fig -- 已绘制的figure对象

    返回:
      [H, W, 3]形状的uint8数组，可直接用于图像处理
    """
    # 调整边距确保坐标刻度完整显示
    fig.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.98,
                        wspace=0.0, hspace=0.0)
    fig.canvas.draw()  # 强制渲染

    # 从画布缓冲区提取RGB数据
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def get_colormap(num_agents):
    """生成随机颜色映射表（用于多代理区分）[1](@ref)

    参数:
      num_agents -- 需要区分的代理数量

    返回:
      [num_agents,4]形状的RGBA颜色数组
    """
    colors = cm.get_cmap('jet', num_agents)  # 使用jet色谱
    colors = colors(range(num_agents))  # 生成离散颜色
    np.random.shuffle(colors)  # 随机排列避免相邻颜色相似
    return colors


def get_viewport(all_states, all_states_mask):
    """计算数据区域的中心点和显示范围（动态适配场景）[6,7](@ref)

    参数:
      all_states -- 所有代理状态 [num_agents, num_steps, 2]
      all_states_mask -- 状态有效性掩码 [num_agents, num_steps]

    返回:
      (center_y, center_x, width) -- 显示区域中心坐标和范围
    """
    valid_states = all_states[all_states_mask]  # 过滤无效数据
    all_y = valid_states[..., 1]  # 提取所有Y坐标
    all_x = valid_states[..., 0]  # 提取所有X坐标

    # 计算中心点坐标
    center_y = (np.max(all_y) + np.min(all_y)) / 2
    center_x = (np.max(all_x) + np.min(all_x)) / 2

    # 计算显示范围（取XY方向最大值）
    range_y = np.ptp(all_y)  # 计算Y方向极差（Peak-to-peak）
    range_x = np.ptp(all_x)  # 计算X方向极差
    width = max(range_y, range_x)  # 取最大值作为显示范围

    return center_y, center_x, width


def visualize_one_step(states, mask, roadgraph, title,
                       center_y, center_x, width, color_map,
                       size_pixels=1000, sdc_index=None):
    """单时间步可视化引擎（核心绘图逻辑）[6,7](@ref)

    新增参数:
      sdc_index -- 主车代理的索引（默认None）
    """
    # 初始化画布
    fig, ax = create_figure_and_axes(size_pixels=size_pixels)

    # 绘制道路网络（黑色点状图）
    rg_pts = roadgraph[:, :2].T  # 提取XY坐标并转置
    ax.plot(rg_pts[0, :], rg_pts[1, :], 'k.', alpha=1, ms=2)  # 黑色半透明点

    # 过滤有效代理状态
    masked_x = states[:, 0][mask]
    masked_y = states[:, 1][mask]
    colors = color_map[mask]

    # 绘制普通代理
    ax.scatter(masked_x, masked_y,
               marker='o',
               s=30,  # 普通代理大小
               linewidths=1.5,
               color=colors,
               alpha=0.7)

    # 如果存在主车索引，单独绘制
    if sdc_index is not None and mask[sdc_index]:
        sdc_x = states[sdc_index, 0]
        sdc_y = states[sdc_index, 1]
        ax.scatter([sdc_x], [sdc_y],
                   marker='*',  # 星形标记
                   s=200,  # 大尺寸
                   linewidths=3,
                   color='red',  # 红色
                   edgecolors='black',  # 黑色边框
                   zorder=3)  # 确保在顶层显示

    # 设置标题和坐标范围
    ax.set_title(title)
    size = max(10, width * 1.0)  # 最小显示范围10米
    ax.axis([-size / 2 + center_x, size / 2 + center_x,
             -size / 2 + center_y, size / 2 + center_y])
    ax.set_aspect('equal')  # 等比例坐标

    # 转换为图像数组
    image = fig_canvas_image(fig)
    plt.close(fig)  # 关闭画布释放内存
    return image


def visualize_all_agents_smooth(decoded_example, size_pixels=1000):
    """全时段多代理轨迹可视化流水线[6,7,8](@ref)"""
    # 解包各时段数据 ---------------------------------------------------
    # 过去轨迹 [num_agents, num_past_steps, 2]
    past_states = tf.stack([decoded_example['state/past/x'],
                            decoded_example['state/past/y']], -1).numpy()
    past_states_mask = decoded_example['state/past/valid'].numpy() > 0.0

    # 当前状态 [num_agents, 1, 2]
    current_states = tf.stack([decoded_example['state/current/x'],
                               decoded_example['state/current/y']], -1).numpy()
    current_states_mask = decoded_example['state/current/valid'].numpy() > 0.0

    # 预测轨迹 [num_agents, num_future_steps, 2]
    future_states = tf.stack([decoded_example['state/future/x'],
                              decoded_example['state/future/y']], -1).numpy()
    future_states_mask = decoded_example['state/future/valid'].numpy() > 0.0

    # 道路网络数据 [num_points, 3]
    roadgraph_xyz = decoded_example['roadgraph_samples/xyz'].numpy()

    # 新增：获取主车索引
    sdc_index = np.where(decoded_example['state/is_sdc'].numpy() == 1)[0]
    print(f"sdc_index{sdc_index}")
    if len(sdc_index) > 0:
        sdc_index = sdc_index[0]  # 取第一个主车（通常只有一个）
    else:
        sdc_index = None  # 无主车的情况

    # 数据预处理 ------------------------------------------------------
    num_agents = past_states.shape[0]
    color_map = get_colormap(num_agents)  # 生成颜色映射

    # 合并全时段数据 [过去+当前+未来]
    all_states = np.concatenate([past_states, current_states, future_states], 1)
    all_states_mask = np.concatenate([past_states_mask,
                                      current_states_mask,
                                      future_states_mask], 1)

    # 计算显示区域参数
    center_y, center_x, width = get_viewport(all_states, all_states_mask)

    # 生成时序图像序列 -------------------------------------------------
    images = []
    # 过去时段可视化（倒序显示）
    for i, (s, m) in enumerate(zip(np.split(past_states, past_states.shape[1], 1),
                                   np.split(past_states_mask, past_states.shape[1], 1))):
        im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz,
                                f'past: {past_states.shape[1] - i}',
                                center_y, center_x, width, color_map,
                                size_pixels, sdc_index)  # 传递主车索引
        images.append(im)

    # 当前时刻可视化
    im = visualize_one_step(current_states[:, 0], current_states_mask[:, 0],
                            roadgraph_xyz, 'current',
                            center_y, center_x, width, color_map,
                            size_pixels, sdc_index)
    images.append(im)

    # 未来预测可视化（正序显示）
    for i, (s, m) in enumerate(zip(np.split(future_states, future_states.shape[1], 1),
                                   np.split(future_states_mask, future_states.shape[1], 1))):
        im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz,
                                f'future: {i + 1}',
                                center_y, center_x, width, color_map,
                                size_pixels, sdc_index)
        images.append(im)

    return images


images=visualize_all_agents_smooth(parsed, size_pixels=1000)

def create_animation(images):
    """创建基于图像序列的Matplotlib动画对象

    参数:
      images -- 图像序列，应为形状为[H,W,3]的numpy数组列表

    返回:
      matplotlib.animation.FuncAnimation对象

    使用示例:
      anim = create_animation(images)
      anim.save('/tmp/animation.avi')  # 保存为视频文件[3,7](@ref)
      HTML(anim.to_html5_video())     # 在Jupyter中内嵌显示[10](@ref)
    """
    # 关闭交互模式避免重复绘图[7](@ref)
    plt.ioff()
    # 创建画布和坐标轴系统
    fig, ax = plt.subplots()

    # 设置图像分辨率参数（1000x1000像素）
    dpi = 100  # 标准打印分辨率[1](@ref)
    size_inches = 1000 / dpi  # 物理尺寸计算
    fig.set_size_inches([size_inches, size_inches])  # 设置画布尺寸

    # 启用交互模式（适用于脚本环境）[7](@ref)
    plt.ion()

    def animate_func(i):
        """帧更新函数，每帧调用一次[1,3](@ref)

        参数:
          i -- 当前帧索引，自动从frames参数范围获取[4](@ref)
        """
        ax.imshow(images[i])  # 绘制当前帧图像
        ax.set_xticks([])  # 隐藏X轴刻度
        ax.set_yticks([])  # 隐藏Y轴刻度
        ax.grid('off')  # 关闭网格线

    # 创建动画对象[1,3,4](@ref)
    anim = animation.FuncAnimation(
        fig=fig,  # 绑定画布
        func=animate_func,  # 帧更新函数
        frames=len(images) // 2,  # 总帧数（取半数图像减少内存消耗）
        interval=100  # 帧间隔时间(ms)，控制播放速度[6](@ref)
    )

    # 关闭画布释放内存（不影响动画对象）[7](@ref)
    plt.close(fig)
    return anim


# 生成动画实例（每5帧取1帧降低分辨率）
anim = create_animation(images[::5])

# 在Jupyter中内嵌显示HTML5视频[10](@ref)
HTML(anim.to_html5_video())
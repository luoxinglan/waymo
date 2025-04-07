import tensorflow as tf
import numpy as np

def extract_and_filter_data(parsed):
    # Extract relevant fields
    objects_of_interest = parsed['state/objects_of_interest'].numpy()
    valid_past = parsed['state/past/valid'].numpy()
    valid_current = parsed['state/current/valid'].numpy()
    valid_future = parsed['state/future/valid'].numpy()

    # Check if all past, current, and future steps are valid for each agent
    all_valid_past = np.all(valid_past == 1, axis=0)
    all_valid_current = valid_current == 1
    all_valid_future = np.all(valid_future == 1, axis=0)

    # Filter indices where objects_of_interest is 1 and all past, current, future are valid
    filter_indices = np.where(objects_of_interest == 1 & all_valid_past & all_valid_current & all_valid_future)[0]

    # Initialize a dictionary to store filtered data
    filtered_data = {}
    keys_to_extract = [
        'state/past/x', 'state/past/y', 'state/past/z',
        'state/past/bbox_yaw', 'state/past/speed', 'state/past/vel_yaw',
        'state/past/velocity_x', 'state/past/velocity_y', 'state/past/timestamp_micros',
        'state/past/valid',
        'state/current/x', 'state/current/y', 'state/current/z',
        'state/current/bbox_yaw', 'state/current/speed', 'state/current/vel_yaw',
        'state/current/velocity_x', 'state/current/velocity_y', 'state/current/timestamp_micros',
        'state/current/valid',
        'state/future/x', 'state/future/y', 'state/future/z',
        'state/future/bbox_yaw', 'state/future/speed', 'state/future/vel_yaw',
        'state/future/velocity_x', 'state/future/velocity_y', 'state/future/timestamp_micros',
        'state/future/valid'
    ]

    for key in keys_to_extract:
        if key.startswith('state/past'):
            filtered_data[key] = parsed[key].numpy()[filter_indices, :, :]
        elif key.startswith('state/current'):
            filtered_data[key] = parsed[key].numpy()[filter_indices, :]
        elif key.startswith('state/future'):
            filtered_data[key] = parsed[key].numpy()[filter_indices, :, :]

    return filtered_data


def create_tfrecord(filtered_data, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        num_records = len(filtered_data['state/current/x'])
        for i in range(num_records):
            feature = {}

            # Add past features
            for key in filtered_data.keys():
                if key.startswith('state/past'):
                    feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=filtered_data[key][i].flatten()))

            # Add current features
            for key in filtered_data.keys():
                if key.startswith('state/current'):
                    feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=filtered_data[key][i]))

            # Add future features
            for key in filtered_data.keys():
                if key.startswith('state/future'):
                    feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=filtered_data[key][i].flatten()))

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


# 假设 parsed 已经定义并包含必要的数据
# 提取并过滤数据
# filtered_data = extract_and_filter_data(parsed)
#
# # 保存为 TFRecord 文件
# output_file = 'filtered_data.tfrecord00001-of-01000'
# create_tfrecord(filtered_data, output_file)
# print(f"Filtered data saved to {output_file}")




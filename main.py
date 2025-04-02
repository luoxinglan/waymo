import io

import numpy as np
import tensorflow as tf
from PIL import Image
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils
import matplotlib.pyplot as plt


# 解析单个TFRecord文件
def parse_tfrecord(tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type='byte')
    for data in dataset:
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        return frame


# 显示相机图像
def display_camera_image(frame, camera_name='FRONT'):
    (range_images, camera_projections, range_image_top_pose) = (
        frame_utils.parse_range_image_and_camera_projection(frame))

    cameras = sorted(frame.context.camera_calibrations, key=lambda c: c.name)

    # 找到指定的相机
    camera_calibration = next(
        x for x in cameras if x.name == dataset_pb2.CameraName.Name.Name(camera_name))

    camera_image = None
    for image in frame.images:
        if image.name != camera_calibration.name:
            continue

        s1 = str(image.image).encode('utf8')
        encoded_jpeg_io = io.BytesIO(s1)
        image_np = np.array(Image.open(encoded_jpeg_io), dtype=np.uint8)
        camera_image = image_np

        break

    plt.figure(figsize=(20, 12))
    plt.imshow(camera_image)
    plt.title(f'Camera {camera_name} Image')
    plt.show()


# 示例使用
tfrecord_file = 'data/uncompressed_tf_example_training_training_tfexample.tfrecord-00000-of-01000'
frame = parse_tfrecord(tfrecord_file)
display_camera_image(frame, camera_name='FRONT')




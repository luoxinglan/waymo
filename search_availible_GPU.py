import tensorflow as tf

# 获取所有可用的物理设备
physical_devices = tf.config.list_physical_devices('GPU')

if physical_devices:
    print("Available GPUs:")
    for device in physical_devices:
        print(device)
else:
    print("No GPUs found.")

import numpy as np
import h5py
import os

# 设置文件夹路径
input_folder = './test/sparse_npy/8'
label_folder = './test/full_npy'

# 获取文件夹中的文件列表，并按字母顺序排序
input_files = sorted(os.listdir(input_folder))
label_files = sorted(os.listdir(label_folder))

# 初始化数据列表
input_data_list = []
label_data_list = []

# 遍历文件夹，加载npy文件到数据列表中
for input_file, label_file in zip(input_files, label_files):
    input_data_list.append(np.load(os.path.join(input_folder, input_file)))
    label_data_list.append(np.load(os.path.join(label_folder, label_file)))

# 将数据列表转换为numpy数组
input_data = np.stack(input_data_list)
label_data = np.stack(label_data_list)

# 获取数据维度
num_images, width, height = input_data.shape

# 创建一个新的h5文件
with h5py.File('output.h5', 'w') as f:
    # 创建一个四维数据集，用于存储输入数据
    input_dataset = f.create_dataset('input', (num_images, width, height, 1), dtype='float32')
    # 将输入数据的前三个维度复制到新的数据集中，并为最后一个维度添加1
    input_dataset[:, :, :, 0] = input_data

    # 创建一个四维数据集，用于存储标签数据
    label_dataset = f.create_dataset('label', (num_images, width, height, 1), dtype='float32')
    # 将标签数据的前三个维度复制到新的数据集中，并为最后一个维度添加1
    label_dataset[:, :, :, 0] = label_data

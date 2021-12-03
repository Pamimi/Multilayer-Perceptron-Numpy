import numpy as np
import scipy.io as sio

def mav(x, len_window = 0.1, overlap = 0.05):
    """
    特征提取
    平均绝对值变化
    :param x: 输入样本
    :param len_window: 滑窗长度
    :param overlap: 重叠
    :return: 平均绝对值变化
    """
    len_window = round(len_window * x.shape[0])
    overlap = round(overlap * x.shape[0])

    # 取绝对值
    x = np.maximum(x, -x)

    # 按滑窗以及overlap大小取平均值
    mav_x = x[0:len_window,:].sum(axis=0)/len_window
    for i in range(len_window-overlap, x.shape[0], len_window-overlap):
        mav_x = np.vstack((mav_x, x[i:i+len_window, :].sum(axis=0)/len_window))

    return mav_x


def dataload(data_file = '数据.mat', label_file = '标签.mat'):
    """
    读取数据并将数据做特征提取(mav)
    以元组列表的形式存储于all_data中,其中每个tuple为(x, y), x为数据, y
    :return:
    """

    data_dict = sio.loadmat(data_file)
    label_dict = sio.loadmat(label_file)
    raw_data = data_dict['preprocessed_dyn']
    raw_label = label_dict['label']

    # 提取每个数据的mean absolut value的同时展平并将数据存为列表
    data = [mav(raw_data[0][i]).reshape(-1,1) for i in range(0, 147)]

    # 将标签由整型转化成一个(10,1)的ndarray
    # 将数据沿列复制为9次
    _label = raw_label
    for i in range(0, 9):
        _label = np.concatenate((_label, raw_label), axis=0)
    y_l = [[1], [2], [3], [5], [6], [7], [8], [11], [14], [15]]
    _label = (_label == y_l) * 1.  # 利用广播机制将True,False转化为1, 0

    # 将标签存储到列表中
    label = [_label[0:10, i].reshape(10, 1) for i in range(0, 147)]

    # 将数据以元组列表的形式存放
    all_data = [(data[i], label[i]) for i in range(0, 147)]

    return all_data


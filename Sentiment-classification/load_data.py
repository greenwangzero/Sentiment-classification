import numpy as np


# 训练集

def dealData():
    res_data = []
    res_label = []
    with open("word2vector.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("[", "")
            line = line.replace("]", "")
            str = line.split(',', 1)
            # 提取标签
            res_label.append(int(str[0].strip()))
            str_data = str[1].split(', ')
            data = [float(x.strip()) for x in str_data]
            res_data.append(np.array(data))
    f.close()
    return np.array(res_data), np.array(res_label)


def load_data_training():
    # 修改数据输入：
    tr0, tr1 = dealData()
    training_inputs = [np.reshape(x, (640, 1)) for x in tr0]
    training_results = [vectorized_result(y) for y in tr1]
    training_data = list(zip(training_inputs, training_results))
    return training_data


def vectorized_result(j):
    e = np.zeros((2, 1))

    e[j] = 1.0

    return e



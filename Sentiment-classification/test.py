import network as network

import load_data as loader
import pickle

import numpy as np


# 测试数据集

def dealTestData():
    res_data = []

    res_label = []

    with open("wector.txt") as f:
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


def load_data_test():
    te0, te1 = dealTestData()

    test_inputs = [np.reshape(x, (640, 1)) for x in te0]

    test_data = list(zip(test_inputs, te1))

    return (test_data)


if __name__ == "__main__":
    test_data = load_data_test()
    training_data = loader.load_data_training()

    net = network.Network([640, 50, 2])

    net.SGD(training_data, 10, 10, 3.0)

   # fr1 = open("weights.pkl", 'rb')

   # net.weights = pickle.load(fr1)

   # fr2 = open("biases.pkl", 'rb')

   # net.biases = pickle.load(fr2)

    total = len(test_data)

    right = net.evaluate(test_data)

    print("rate:{}/{},{}%".format(right, total, (right * 100 / total)))

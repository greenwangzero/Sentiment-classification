import numpy as np

import pickle

import load_data as loader
import jieba

import gensim

import network as network


def numextract(lists):
    vector = []

    for i in range(len(lists)):

        for t in range(len(lists[i])):
            vector.append(lists[i].item(t))

    return vector


# 分词，生成词向量

def cin_test(vector_model, vocstr):
    voclist = jieba.cut(vocstr)

    x = [v.strip() for v in voclist]

    vectors = []

    for word in x:

        try:

            vectors.append(vector_model[word])

        except Exception as e:

            vectors.append(np.zeros(64))

    if len(vectors) < 10:
        vectors.append(np.zeros(64 * (10 - len(vectors))))

        vector = numextract(vectors)

        test_data = vector

        return test_data

    if len(vectors) > 10:
        vector = numextract(vectors)[0: 640]

        test_data = vector

        return test_data


# 装配数据格式

def dealCinTestData(vocstr):
    vector_model_path = 'news_12g_baidubaike_20g_novel_90g_embedding_64.bin'

    vector_model = gensim.models.KeyedVectors.load_word2vec_format(vector_model_path, binary=True, limit=100000)

    voc_list = cin_test(vector_model, vocstr)

    clist = []

    clist.append(np.array(voc_list))

    cin_data = np.array(clist)

    cin_label = np.array([0])

    test_inputs = [np.reshape(x, (640, 1)) for x in cin_data]

    test_data = list(zip(test_inputs, cin_label))

    return test_data


# 输入神经网络进行测试

def test(vocstr):
    #net = network.Network([640, 50, 2])

   # fr1 = open("weights.pkl", 'rb')

    #fr2 = open("biases.pkl", 'rb')

    #net.weights = pickle.load(fr1)

    #net.biases = pickle.load(fr2)

    test_data = dealCinTestData(vocstr)

    res = net.evaluate2(test_data)[0]

    return res


if __name__ == "__main__":

    training_data = loader.load_data_training()

    net = network.Network([640, 50, 2])

    net.SGD(training_data, 10, 10, 3.0)

    while True:

        str1 = input("请输入：")

        if test(str1):

            print("positive")

        else:

            print("negative")

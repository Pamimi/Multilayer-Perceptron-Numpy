import numpy as np
import random
import Muscledata_load

def actfunc(z):
    """
    激活函数,此处采用Sigmoid激活函数定义
    """
    b = 1.0 / (1.0 + np.exp(-z))
    return b


def d_actfunc(z):
    """
    激活函数的导数,即Sigmoid函数的导数
    """
    b = actfunc(z) * (1 - actfunc(z))
    return b


class MLP(object):
    """
    创建一个多层感知机的类
    """

    def __init__(self, sizes):
        """
        :param sizes: 是一个列表，其中包含了神经网络每一层的神经元的个数，列表的长度就是神经网络的层数。
        举个例子，假如列表为[784,30,10]，那么意味着它是一个3层的神经网络，第一层包含784个神经元，第二层30个，最后一层10个。
        注意，神经网络的权重和偏置是随机生成的，使用一个均值为0，方差为1的高斯分布。
        注意第一层被认为是输入层，它是没有偏置向量b和权重向量w的。因为偏置只是用来计算第二层之后的输出
        """
        self._num_layers = len(sizes)  # 记录神经网络的层数
        # 为隐藏层和输出层生成偏置向量b，还是以[784,30,10]为例，那么一共会生成2个偏置向量b，分别属于隐藏层和输出层，大小分别为30x1,10x1。
        self._biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # 为隐藏层和输出层生成权重向量W, 以[784,30,10]为例，这里会生成2个权重向量w，分别属于隐藏层和输出层，大小分别是30x784, 10x30。
        self._weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        前向传播
        :param a: 输入数据
        :return: 前向传播结果
        """
        for w, b in zip(self._weights, self._biases):
            a = actfunc(np.dot(w, a) + b)
        return a

    def backprop(self, x, y):
        """
        反向传播算法，计算loss对w和b的梯度
        :param x: 输入数据
        :param y: 数据标签
        :return: 权重以及偏置的梯度grad_weights, grad_biases
        """
        # 初始化权重w和偏置b的梯度
        grad_weights = [np.zeros(w.shape) for w in self._weights]
        grad_biases = [np.zeros(b.shape) for b in self._biases]
        # feedforward
        activation = x
        activations = [x]  # 存储每一层的激活值a
        zs = []  # 存储每一层的z向量
        for w, b in zip(self._weights, self._biases):

            z = np.dot(w, activation) + b
            zs.append(z)
            activation = actfunc(z)
            activations.append(activation)
        # backward pass
        delta = self.loss_derivative(activations[-1], y) * d_actfunc(zs[-1])  # loss对最后一层的求导
        grad_weights[-1] = np.dot(delta, activations[-2].transpose())
        grad_biases[-1] = delta
        for layer in range(2, self._num_layers):
            z = zs[-layer]
            d_act = d_actfunc(z)
            delta = np.dot(self._weights[-layer + 1].transpose(), delta) * d_act
            grad_weights[-layer] = np.dot(delta, activations[-layer - 1].transpose())
            grad_biases[-layer] = delta
        return (grad_weights, grad_biases)

    def loss_derivative(self, output_activations, y):
        """
        最小均方损失的导数
        """
        return output_activations - y

    def update_mini_batch(self, mini_batch, learning_rate):
        """
        通过小批量随机梯度下降以及反向传播来更新神经网络的权重和偏置向量
        :param mini_batch: 批量大小
        :param learning_rate: 学习率
        """
        nabla_w = [np.zeros(w.shape) for w in self._weights]
        nabla_b = [np.zeros(b.shape) for b in self._biases]
        for x, y in mini_batch:
            # 反向传播算法，运用链式法则求得对b和w的偏导
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            # 将一个小批量的的每一个偏导数相加，求得该批量的累计导数
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        # 使用梯度下降来更新权重w和偏置b
        self._weights = [w - (learning_rate / len(mini_batch)) * dw
                         for w, dw in zip(self._weights, nabla_w)]
        self._biases = [b - (learning_rate / len(mini_batch)) * db
                        for b, db in zip(self._biases, nabla_b)]

    def sgd(self,training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        """
        小批量随机梯度下降
        :param training_data: 训练集
        :param epochs: 训练周期
        :param mini_batch_size: batch大小
        :param learning_rate: 学习率
        :param test_data: 测试集
        """
        n = len(training_data)

        for j in range(epochs):
            # 每次迭代之前将随机数据打乱并分成小批量
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            # 对于每个batch输入网络进行训练
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            # 如果测试集test_data被指定, 则每一次迭代后都对训练集和测试集进行评估
            if test_data:
                train_accuracy, test_accuracy = self.evaluate(training_data, test_data)
                # print("Epoch %d: train accuracy rate: %.2f%% test accuracy rate: %.2f%%" % (j + 1, train_accuracy, test_accuracy))
                return train_accuracy, test_accuracy
            else:
                print("Epoch {0} complete".format(j+1))

    def evaluate(self, training_data, test_data):
        """
        返回神经网络对测试数据training_data, test_data的预测结果，并且计算其中识别正确的个数
        因为神经网络的输出是一个10x1的向量，我们需要知道哪一个神经元被激活的程度最大，
        因此使用了argmax函数以获取激活值最大的神经元的下标，那就是网络输出的最终结果。
        """
        y_t = [1, 2, 3, 5, 6, 7, 8, 11, 14, 15]

        train_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                         for (x, y) in training_data]

        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        train_trues = sum(int(y_hat == y) for (y_hat, y) in train_results)
        test_trues = sum(int(y_hat == y) for (y_hat, y) in test_results)
        return (train_trues / len(training_data)) * 100, (test_trues / len(test_data)) * 100

if __name__ == "__main__":

    all_data = Muscledata_load.dataload()

    def K_fold(all_data, net_shape, batch_size = 5, lr = 0.3, epoachs = 10):
        """
        实现留一法评估多层感知机性能, 默认epoachs为10
        相当于10次147(147为数据集大小)折交叉验证, 即相当于1470次留出法验证
        :param all_data: 未进行划分的数据集
        :param net_shape: 网络结构
        :param epoachs: 迭代周期
        """
        random.shuffle(all_data)  # 将数据打乱
        net = MLP(net_shape) # 创建多层感知机网络实例

        for epoach in range(epoachs):
            # 用于存储每折验证的精确度
            train_accuracy = np.zeros((len(all_data), 1))
            test_accuracy = np.zeros((len(all_data), 1))

            # 留一法数据集划分
            for i in range(0, len(all_data)):
                test_data = []
                test_data.append(all_data[i])
                pop = all_data.pop(i)
                train_data = all_data[:] # 注意此处需使用切片
                all_data.insert(i, pop)
                # 对于每一留一法数据集, 将验证结果存储于预先创建好的变量中
                train_accuracy[i], test_accuracy[i] = net.sgd(train_data, 1, batch_size, lr, test_data=test_data)
            # 计算并输出每次交叉验证的结果
            tr_ac = train_accuracy.sum(axis=0)/len(all_data)
            tst_ac = test_accuracy.sum(axis=0)/len(all_data)
            print("Epoch %d: train accuracy rate: %.2f%% test accuracy rate: %.2f%%" % (epoach + 1, tr_ac, tst_ac))

    K_fold(all_data, [1280, 40, 10], batch_size = 5, lr = 0.3)

    # for test, 将129注释,128取消注释
    # 随机划分数据集进行测试分类
    # random.shuffle(all_data)
    # test_data = all_data[123:147]
    # train_data = all_data[0:123]
    #
    # net = MLP([1280, 40, 10])
    # net.sgd(train_data, 1000, 5, 0.3, test_data=test_data)


"""
@author: Maziar Raissi
"""

import sys

sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time

np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X, u, layers, lb, ub):

        self.lb = lb
        self.ub = ub

        self.x = X[:, 0:1]
        self.t = X[:, 1:2]
        self.u = u

        self.layers = layers

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        # Initialize parameters
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([-6.0], dtype=tf.float32)

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.u_pred = self.net_u(self.x_tf, self.t_tf)
        self.f_pred = self.net_f(self.x_tf, self.t_tf)

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))

        # (me)加入记录容器方便记录参数
        self.loss_history = []
        self.l1_history = []
        self.l2_history = []
        self.it_history = []

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    # 构建神经网络框架
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    # 定义权重矩阵W
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    # 定义神经网络
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        # 每层网络中的具体运算定义
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    # 定义解函数
    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        return u

    # 定义f
    def net_f(self, x, t):
        lambda_1 = self.lambda_1
        lambda_2 = tf.exp(self.lambda_2)
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx

        return f

    # 实时获取参数状态
    def callback(self, loss, lambda_1, lambda_2):
        self.loss_history.append(loss)
        self.l1_history.append(lambda_1)
        self.l2_history.append(np.exp(lambda_2))
        self.it_history.append(len(self.it_history))
        print('Loss: %e, l1: %.5f, l2: %.5f' % (loss, lambda_1, np.exp(lambda_2)),
              flush=True)

    def train(self, nIter):
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u}
        start_time = time.time()

        # Adam
        print("=== START ADAM ===", flush=True)
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_1_value = self.sess.run(self.lambda_1)
                lambda_2_value = np.exp(self.sess.run(self.lambda_2))
                # (me)更新记录容器中参数数值
                self.loss_history.append(loss_value)
                self.l1_history.append(lambda_1_value)
                self.l2_history.append(lambda_2_value)
                self.it_history.append(it)
                print('It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f, Time: %.2f' %
                      (it, loss_value, lambda_1_value, lambda_2_value, elapsed),
                      flush=True)
                start_time = time.time()

        print("=== START L-BFGS ===", flush=True)
        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss, self.lambda_1, self.lambda_2],
                                loss_callback=self.callback)

    def predict(self, X_star):

        tf_dict = {self.x_tf: X_star[:, 0:1], self.t_tf: X_star[:, 1:2]}

        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)

        return u_star, f_star


if __name__ == "__main__":
    nu = 0.01 / np.pi

    N_u = 2000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    data = scipy.io.loadmat('../Data/burgers_shock.mat')

    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    """
    @author: Maziar Raissi
    """

    import sys

    sys.path.insert(0, '../../Utilities/')

    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.io
    from scipy.interpolate import griddata
    from plotting import newfig, savefig
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.gridspec as gridspec
    import time

    np.random.seed(1234)
    tf.set_random_seed(1234)


    class PhysicsInformedNN:
        # Initialize the class
        def __init__(self, X, u, layers, lb, ub):

            self.lb = lb
            self.ub = ub

            self.x = X[:, 0:1]
            self.t = X[:, 1:2]
            self.u = u

            self.layers = layers

            # Initialize NNs
            self.weights, self.biases = self.initialize_NN(layers)

            # tf placeholders and graph
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                         log_device_placement=True))

            # Initialize parameters
            self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
            self.lambda_2 = tf.Variable([-6.0], dtype=tf.float32)

            self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
            self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
            self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

            self.u_pred = self.net_u(self.x_tf, self.t_tf)
            self.f_pred = self.net_f(self.x_tf, self.t_tf)

            self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                        tf.reduce_mean(tf.square(self.f_pred))

            # (me)加入记录容器方便记录参数
            self.loss_history = []
            self.l1_history = []
            self.l2_history = []
            self.it_history = []

            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                    method='L-BFGS-B',
                                                                    options={'maxiter': 50000,
                                                                             'maxfun': 50000,
                                                                             'maxcor': 50,
                                                                             'maxls': 50,
                                                                             'ftol': 1.0 * np.finfo(float).eps})

            self.optimizer_Adam = tf.train.AdamOptimizer()
            self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

            init = tf.global_variables_initializer()
            self.sess.run(init)

        # 构建神经网络框架
        def initialize_NN(self, layers):
            weights = []
            biases = []
            num_layers = len(layers)
            for l in range(0, num_layers - 1):
                W = self.xavier_init(size=[layers[l], layers[l + 1]])
                b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
                weights.append(W)
                biases.append(b)
            return weights, biases

        # 定义权重矩阵W
        def xavier_init(self, size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
            return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

        # 定义神经网络
        def neural_net(self, X, weights, biases):
            num_layers = len(weights) + 1

            # 每层网络中的具体运算定义
            H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
            for l in range(0, num_layers - 2):
                W = weights[l]
                b = biases[l]
                H = tf.tanh(tf.add(tf.matmul(H, W), b))
            W = weights[-1]
            b = biases[-1]
            Y = tf.add(tf.matmul(H, W), b)
            return Y

        # 定义解函数
        def net_u(self, x, t):
            u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
            return u

        # 定义f
        def net_f(self, x, t):
            lambda_1 = self.lambda_1
            lambda_2 = tf.exp(self.lambda_2)
            u = self.net_u(x, t)
            u_t = tf.gradients(u, t)[0]
            u_x = tf.gradients(u, x)[0]
            u_xx = tf.gradients(u_x, x)[0]
            f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx

            return f

        # 实时获取参数状态
        def callback(self, loss, lambda_1, lambda_2):
            self.loss_history.append(loss)
            self.l1_history.append(lambda_1)
            self.l2_history.append(np.exp(lambda_2))
            self.it_history.append(len(self.it_history))
            print('Loss: %e, l1: %.5f, l2: %.5f' % (loss, lambda_1, np.exp(lambda_2)),
                  flush=True)

        def train(self, nIter):
            tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u}
            start_time = time.time()

            # Adam
            print("=== START ADAM ===", flush=True)
            for it in range(nIter):
                self.sess.run(self.train_op_Adam, tf_dict)

                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    lambda_1_value = self.sess.run(self.lambda_1)
                    lambda_2_value = np.exp(self.sess.run(self.lambda_2))
                    # (me)更新记录容器中参数数值
                    self.loss_history.append(loss_value)
                    self.l1_history.append(lambda_1_value)
                    self.l2_history.append(lambda_2_value)
                    self.it_history.append(it)
                    print('It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f, Time: %.2f' %
                          (it, loss_value, lambda_1_value, lambda_2_value, elapsed),
                          flush=True)
                    start_time = time.time()

            print("=== START L-BFGS ===", flush=True)
            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict,
                                    fetches=[self.loss, self.lambda_1, self.lambda_2],
                                    loss_callback=self.callback)

        def predict(self, X_star):

            tf_dict = {self.x_tf: X_star[:, 0:1], self.t_tf: X_star[:, 1:2]}

            u_star = self.sess.run(self.u_pred, tf_dict)
            f_star = self.sess.run(self.f_pred, tf_dict)

            return u_star, f_star


    if __name__ == "__main__":
        nu = 0.01 / np.pi

        N_u = 2000
        layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

        data = scipy.io.loadmat('../Data/burgers_shock.mat')

        t = data['t'].flatten()[:, None]
        x = data['x'].flatten()[:, None]
        Exact = np.real(data['usol']).T

        X, T = np.meshgrid(x, t)

        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_star = Exact.flatten()[:, None]

        # Doman bounds
        lb = X_star.min(0)
        ub = X_star.max(0)

        ######################################################################
        ########################### Noisy Data ###############################
        ######################################################################
        noise = 0.01
        u_train = u_train + noise * np.std(u_train) * np.random.randn(u_train.shape[0], u_train.shape[1])

        model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
        model.train(10000)

        # 画图
        plt.figure()

        plt.subplot(3, 1, 1)
        plt.plot(model.it_history, model.loss_history)
        plt.title("Loss")

        plt.subplot(3, 1, 2)
        plt.plot(model.l1_history)
        plt.title("lambda_1")

        plt.subplot(3, 1, 3)
        plt.plot(model.l2_history)
        plt.title("lambda_2")

        plt.show()

        u_pred, f_pred = model.predict(X_star)

        lambda_1_value_noisy = model.sess.run(model.lambda_1)
        lambda_2_value_noisy = model.sess.run(model.lambda_2)
        lambda_2_value_noisy = np.exp(lambda_2_value_noisy)

        error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0) * 100
        error_lambda_2_noisy = np.abs(lambda_2_value_noisy - nu) / nu * 100

        print('Error lambda_1: %f%%' % (error_lambda_1_noisy))
        print('Error lambda_2: %f%%' % (error_lambda_2_noisy))
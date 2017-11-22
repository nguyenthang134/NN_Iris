import numpy as np

class Neural:
    X = np.loadtxt("testIris.txt", delimiter=",", usecols=(0, 1, 2, 3))
    X1 = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    Y = np.loadtxt("testIris.txt", dtype="string", delimiter=",", usecols=4)
    m = X1.shape[0]

    def __init__(self, epsilon = 0.0001, numb_hidden_layer = 1, alpha = 0.1, iteration = 300, init_epsilon = 0.1, lambd = 1, hidden_layer_size = 4, class_size = 3):
        self.epsilon = epsilon
        self.numb_hidden_layer = numb_hidden_layer
        self.alpha = alpha
        self.iteration = iteration
        self.init_epsilon = init_epsilon
        self.lambd = lambd
        self.hidden_layer_size = hidden_layer_size
        self.class_size = class_size

    def sigmoid(self, z):
        return (1. / (1 + np.exp(-z)))

    def sigmoid_grad(self, z):
        gz = self.sigmoid(z)
        return gz * (1 - gz)

    def init_theta(self):
        theta1 = np.random.rand(self.hidden_layer_size, self.X.shape[1] + 1) * (2 * self.init_epsilon) - self.init_epsilon  # 4x5
        theta2 = np.random.rand(self.class_size, self.hidden_layer_size + 1) * (2 * self.init_epsilon) - self.init_epsilon  # 3x5
        return theta1, theta2
        # randomlize theta

    def forward(self, W1, W2):
        a1 = self.X1    # [150x5]
        z2 = np.dot(a1, np.transpose(W1))     #[150x4]
        a2 = self.sigmoid(z2)
        a2_bias = np.column_stack([np.ones((self.m, 1)), a2])    #[150x5]
        z3 = np.dot(a2_bias, np.transpose(W2))
        a3 = self.sigmoid(z3)
        return a2_bias, a3
        # a3 is hypothesis function which we need to find

    def backward(self):
        epoch = 0
        D2 = D1 = 0
        W1 = self.init_theta()[0]
        W2 = self.init_theta()[1]
        while epoch < 10:
            epoch += 1
            # self.forward()[1] is a3
            a3 = self.forward(W1, W2)[1]
            a2 = self.forward(W1, W2)[0]
            d3 = a3 - self.Y_binary_flowers()  # Y is vectorized of y [150x3]
            d2 = np.dot(d3, self.init_theta()[1]) * (a2 * (1 - a2))    #[150x5)

            D2 = D2 + np.dot(np.transpose(d3), a2)
            D1 = D1 + np.dot(np.transpose(d2[:, 1:]), self.X1)

            derivative2 = 1. / (len(self.X)) * (D2 + self.lambd * W2)
            derivative1 = 1. / (len(self.X)) * (D1 + self.lambd * W1)

            W2 -= self.alpha * derivative2
            W1 -= self.alpha * derivative1

            # print self.cost_function(a3)
        # print W2
        # print "================================================================="
        # print W1
        print a3

    def Y_binary(self, flower):
        y = []
        # trainning output as binary
        for i in range(0, len(self.Y)):
            if (self.Y[i] == flower):
                y.append(1)
            else:
                y.append(0)
        y = np.array(y)
        y.shape = (150, 1)

        return y

    def Y_binary_flowers(self):
        iris_setosa = self.Y_binary("Iris-setosa")
        iris_versicolor = self.Y_binary("Iris-versicolor")
        iris_virginica = self.Y_binary("Iris-virginica")
        y = np.append(np.append(iris_setosa, iris_versicolor, axis=1), iris_virginica, axis=1)
        return y

    def cost_function(self, predicted_output):
        y = self.Y_binary_flowers()
        J = (-1. / self.m) * np.sum((y * np.log(predicted_output) + (1 - y) * np.log(1 - predicted_output)))
        return J

    # def gradient_checking(self):
    #
    # def minimize(self):
    #     for i in range(0, self.iteration):


neural = Neural()
neural.backward()
# print Neural().init_theta()[1]
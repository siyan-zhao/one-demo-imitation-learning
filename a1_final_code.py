import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def loadData():
    with np.load('notMNIST.npz') as data:
        Data, Target = data['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx] / 255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target == posClass] = 1
        Target[Target == negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


def MSE(W, b, x, y, reg):
    # Your implementation here
    y_hat = np.matmul(x, W) + b
    error = y_hat - y
    n = len(y)
    mse = np.sum(error ^ 2) / n + reg / 2 * np.sum(W * W)
    return mse


def gradMSE(W, b, x, y, reg):
    # Your implementation here
    predictions = np.matmul(x, W) + b
    error = predictions - y
    # Accuracy:
    predict_label = []
    for i in predictions:
        if i >= 0.5:
            predict_label.append(1)
        else:
            predict_label.append(0)
    accuracy = sum(predict_label) / len(predict_label)
    delta_w = np.matmul(np.transpose(x), error) / (np.shape(y)[0]) + 2 * reg * W
    delta_b = (np.sum(error)) / (np.shape(y)[0])
    return delta_w, delta_b, accuracy, error


def grad_descent(W, b, epochs, trainData, trainTarget, validData, validTarget, testData,
                 testTarget, alpha, reg, err_tol, lossType='MSE'):
    # Your implementation here
    # Batch gradient: sum up all the examples in the dataset

    train_acc_list = []
    train_loss_list = []
    valid_acc_list = []
    valid_loss_list = []
    test_acc_list = []
    test_loss_list = []
    for epoch_idx in range(epochs):
        if lossType == 'MSE':
            dW, db, _, train_loss = gradMSE(W, b, trainData, trainTarget, reg)
        elif lossType == 'CE':
            dW, db = gradCE(W, b, trainData, trainTarget, reg)
        new_W = W - alpha * dW
        new_b = b - alpha * db

        prediction = np.matmul(trainData, new_W) + new_b
        train_loss_list.append(np.mean(np.square(prediction - trainTarget)))
        train_acc_list.append(np.sum((prediction >= 0.5) == trainTarget) / (trainTarget.shape[0]))

        prediction = np.matmul(validData, new_W) + new_b
        valid_acc_list.append(np.sum((prediction >= 0.5) == validTarget) / (validTarget.shape[0]))
        valid_loss_list.append(np.mean(np.square(prediction - validTarget)))

        prediction = np.matmul(testData, new_W) + new_b
        test_acc_list.append(np.sum((prediction >= 0.5) == testTarget) / (testTarget.shape[0]))
        test_loss_list.append(np.mean(np.square(prediction - testTarget)))

        error = np.linalg.norm(new_W - W)
        print('epoch idx:', epoch_idx, 'weights error:', error)
        if error < err_tol:
            return new_W, new_b
        else:
            W = new_W
            b = new_b

    return W, b, train_acc_list, train_loss_list, valid_acc_list, valid_loss_list, test_acc_list, test_loss_list


def norm_equation(x, y):
    x_b = np.ones((np.shape(x)[0], 1))
    xx = np.hstack((x, x_b))
    xx_t = np.transpose(xx)
    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(xx_t, xx)), xx_t), y)
    weights = W[:-1, :]
    b = W[-1][0]
    return weights, b

def crossEntropyLoss(W, b, x, y, reg):
    # Accuracy:
    predictions = np.matmul(x, W) + b
    predict_label = []
    for i in predictions:
        if i > 0.5:
            predict_label.append(1)
        else:
            predict_label.append(0)
    accuracy = len(predict_label == 1) / len(predict_label)
    y_hat = 1.0 / (1.0 + np.exp(-(predictions)))
    error = (np.sum(-(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))))
    reg_penalty = reg / 2 * np.sum(W * W)
    cross_entropy_loss = error / y.shape[0] + reg_penalty
    return cross_entropy_loss, accuracy, error


def gradCE(W, b, x, y, reg):
    # Your implementation here
    y_hat = 1.0 / (1.0 + np.exp(-(np.matmul(x, W) + b)))
    delta_w = np.matmul(np.transpose(x), (y_hat - y)) / y.shape[0] + 2 * reg * W
    delta_b = np.sum((y_hat - y)) / y.shape[0]
    return delta_w, delta_b

def compute_acc(prediction, y):
    acc = (np.sum((prediction >= 0.5) == y))
    acc = acc / len(y)
    return acc


def buildGraph(loss_function, batch_size, lr, ep, beta_mse, beta_ce, epochs):
    #Initialize weight and bias tensors
    graph = tf.Graph()
    batch_size = 500
    beta_mse = beta_mse
    beta_ce = beta_ce
    epochs = epochs
    lr = lr
    ep = ep

    with graph.as_default():
        W = tf.Variable(tf.truncated_normal(shape=(28*28, 1), dtype=tf.float32, mean=0.0, stddev=0.5))
        b = tf.Variable(tf.zeros(1))
        #train
        inputs = tf.placeholder(tf.float32, shape=(None, trainData.shape[1]))  # D A
        labels = tf.placeholder(tf.float32, shape=(None, 1))  # D 1, D = number of samples
        predictions = tf.matmul(inputs, W) + b
        tf.set_random_seed(421)
        if loss_function == "MSE":
            loss = tf.losses.mean_squared_error(labels, predictions) + 0.5 * reg * tf.square(tf.norm(W))

            update = tf.train.AdamOptimizer(learning_rate=lr, epsilon=ep).minimize(loss)

        elif loss_function == "CE":
            # The sigmoid function is often called the logistic function and hence a linear model with the cross-
            # entropy loss is named \logistic regression".
            loss = tf.losses.sigmoid_cross_entropy(labels, predictions) + 0.5 * reg * tf.square(tf.norm(W))

            update = tf.train.AdamOptimizer(learning_rate=lr, epsilon=ep).minimize(loss)

    with tf.Session(graph=graph) as session:
            total_num_batch = int(3500 / batch_size)
            tf.global_variables_initializer().run()

            train_acc_list = []
            train_loss_list = []
            valid_acc_list = []
            valid_loss_list = []
            test_acc_list = []
            test_loss_list = []
            train_data, valid_data, test_data, train_label, valid_label, test_label = loadData()
            train_data = train_data.reshape(len(train_label), 784)
            valid_data = valid_data.reshape(len(valid_label), 784)
            test_data = test_data.reshape(len(test_label), 784)

            for i in range(0, epochs):
                for j in range(0, total_num_batch):
                    x = train_data[j * batch_size : (j + 1) * batch_size, ]
                    y = train_label[j * batch_size : (j + 1) * batch_size, ]

                    _, new_W, new_b, train_loss, tr_pred = session.run([update, W, b, loss, predictions], {inputs: x, labels: y})
                    _, new_W, new_b, valid_loss, v_pred = session.run([update, W, b, loss, predictions], {inputs: valid_data, labels: valid_label})
                    _, new_W, new_b, test_loss, te_pred = session.run([update, W, b, loss, predictions],
                                                                      {inputs: testData, labels: test_label})
                print('idx:',j, 'train_loss',np.mean(train_loss))
                train_loss_list.append(train_loss)
                valid_loss_list.append(valid_loss)
                test_loss_list.append(test_loss)
                train_acc_list.append(compute_acc(tr_pred, y))
                valid_acc_list.append(compute_acc(v_pred, valid_label))
                test_acc_list.append(compute_acc(te_pred, test_label))

    return new_W, new_b, train_loss_list, valid_loss_list, test_loss_list, train_acc_list, valid_acc_list, test_acc_list


if __name__ == '__main__':
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = np.reshape(trainData, [trainData.shape[0], -1])
    validData = np.reshape(validData, [validData.shape[0], -1])
    testData = np.reshape(testData, [testData.shape[0], -1])
    mu, sigma = 0, 0.5
    W = np.random.normal(mu, sigma, (trainData.shape[1], 1))
    b = 0
    alpha = 0.005  # exp!
    epoch = 5000

    reg = 0.1
    err_tol = 1e-7
    lossType = 'CE'
    # it's a long line of code
    """
    _, _, train_acc_list, train_loss_list, valid_acc_list, valid_loss_list, \
    test_acc_list, test_loss_list = grad_descent(W, b, epoch, trainData, trainTarget, validData,
                                                 validTarget, testData, testTarget, alpha, reg, err_tol, lossType)

    plt.title('\nCE learning,\n5k epochs and regularization parameter = ' + str(reg) + '\n'
                                                                                       'Learning Rate =' + str(alpha))
    CE_train_loss = train_loss_list

    lossType = 'MSE'
    W, b, train_acc_list, train_loss_list, valid_acc_list, valid_loss_list, \
    test_acc_list, test_loss_list = grad_descent(W, b, epoch, trainData, trainTarget, validData,
                                                 validTarget, testData, testTarget, alpha, reg, err_tol, lossType)
    """
    # PART 3:
    batch_size = 300
    lr = 0.001
    ep = 1e-4
    beta1 = 0.9
    beta2 = 0.999
    epochs = 1500
    loss_function = "CE"
    new_W, new_b, train_loss_list, valid_loss_list, test_loss_list, train_acc_list, valid_acc_list, test_acc_list = buildGraph(
        lossType, batch_size, lr, ep, beta1, beta2, epochs)

    MSE_train_loss = train_loss_list
    # plt.plot(train_acc_list, 'g', valid_acc_list, 'b', test_acc_list, 'r')
    # plt.plot(train_loss_list, 'g', valid_loss_list, 'b', test_loss_list, 'r')
    plt.plot(CE_train_loss, 'g', MSE_train_loss, 'r')
    # plt.legend(['training acc', 'validation acc', 'test acc'])
    # plt.legend(['training loss', 'validation loss', 'test loss'])
    plt.legend(['MSE training loss', 'CE training loss'])
    plt.ylabel('Loss')
    plt.savefig('MSE_CE_loss_alpha=0.005.png')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.optimize
import time
import math
import itertools
from sklearn.decomposition import PCA

start = time.time()

# For this assignment, assume that every hidden layer has the same number of neurons.
NUM_INPUT = 784
NUM_OUTPUT = 10


# batch_size = 128

def relu(x):
    # return np.maximum(0,x)
    x_ = x.copy()
    x_[x_ < 0] = 0

    return x_


def standardize_data(X):
    return X / 256


def softmax(z):
    term_1 = np.exp(z)
    denom = np.sum(term_1, axis=0, keepdims=True)
    return term_1 / denom


def onehotencoding(y):
    b = np.zeros((y.size, y.max() + 1))
    b[np.arange(y.size), y] = 1

    return b


def relu_diff(z):
    relu_ = z.copy()
    relu_[relu_ <= 0] = 0
    relu_[relu_ > 0] = 1

    return relu_


def split_data(x, y):
    n = y.shape[1]
    batch_split = np.random.permutation(n)
    split_pt = (80 * n) // 100
    training_split = batch_split[0:split_pt]
    validation_split = batch_split[split_pt:]
    training_x = x[:, training_split]
    training_y = y[:, training_split]
    valid_x = x[:, validation_split]
    valid_y = y[:, validation_split]
    # print("here", training_x.shape)
    return training_x, training_y, valid_x, valid_y


def shuffle(x, y):
    n = y.shape[1]
    # print("shuffle", n)
    shuffle_idx = np.random.permutation(n)
    new_x = x[:, shuffle_idx]
    new_y = y[:, shuffle_idx]
    # print(new_x.shape)

    return new_x, new_y


def batch_gen(x, y, batch_size):
    x, y = shuffle(x, y)
    batch_list = []
    n = y.shape[1]
    n_batches = math.ceil(n / batch_size)
    batch_list.append(np.array_split(x, n_batches, axis=1))
    batch_list.append(np.array_split(y, n_batches, axis=1))

    return batch_list


# Unpack a list of weights and biases into their individual np.arrays.
def unpack(weightsAndBiases, n_hidden, n_layers):
    # Unpack arguments
    Ws = []

    # Weight matrices
    start = 0
    end = NUM_INPUT * n_hidden
    W = weightsAndBiases[start:end]
    Ws.append(W)

    for i in range(n_layers - 1):
        start = end
        end = end + n_hidden * n_hidden
        W = weightsAndBiases[start:end]
        Ws.append(W)

    start = end
    end = end + n_hidden * NUM_OUTPUT
    W = weightsAndBiases[start:end]
    Ws.append(W)

    Ws[0] = Ws[0].reshape(n_hidden, NUM_INPUT)
    for i in range(1, n_layers):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(n_hidden, n_hidden)
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, n_hidden)

    # Bias terms
    bs = []
    start = end
    end = end + n_hidden
    b = weightsAndBiases[start:end]
    bs.append(b)

    for i in range(n_layers - 1):
        start = end
        end = end + n_hidden
        b = weightsAndBiases[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weightsAndBiases[start:end]
    bs.append(b)

    return Ws, bs


def forward_prop(x, y, weightsAndBiases, n_hidden, n_layers):
    h = x
    ###print(len(weightsAndBiases))
    Ws, bs = unpack(weightsAndBiases, n_hidden, n_layers)
    ###print(Ws[0].shape)
    hs = []
    zs = []
    # z = np.matmul(Ws[0], x) +  bs[0].reshape(-1,1)
    # zs.append(z)
    # h = relu(z)
    # hs.append(h)
    # for w, b in zip(Ws, bs):
    for i in range(len(Ws) - 1):
        # print(Ws[i].shape, h[i].shape, bs[i].shape)
        ####print(count)
        # bs[i] = bs[i].reshape(-1,1)

        z = np.matmul(Ws[i], h) + bs[i].reshape(-1, 1)
        h = relu(z)
        hs.append(h)
        zs.append(z)
    z_ = np.matmul(Ws[-1], hs[-1]) + bs[-1].reshape(-1, 1)
    zs.append(z_)
    ####print(count)
    # print("zs", len(zs))
    # print("hs", len(hs))
    yhat = softmax(z_)
    # hs.append(yhat)
    # Return loss, pre-activations, post-activations, and predictions

    n = y.shape[1]
    # print("n", n)
    # y = onehotencoding(y)
    loss = np.sum(np.log(yhat) * y)
    loss = (-1 / n) * loss

    return loss, zs, hs, yhat


def back_prop(x, y, weightsAndBiases, n_hidden, n_layers):
    loss, zs, hs, yhat = forward_prop(x, y, weightsAndBiases, n_hidden, n_layers)

    Ws, bs = unpack(weightsAndBiases, n_hidden, n_layers)

    # print("bshape", Ws[0].shape)

    dJdWs = []  # Gradients w.r.t. weights
    dJdbs = []  # Gradients w.r.t. biases
    # y = onehotencoding(y)
    # print("y_shape", yhat.shape, y.shape)
    g = yhat - y

    # print("first_g", g.shape)

    n = y.shape[1]
    # print("n__________", n)

    # TODO
    for i in range(n_layers, -1, -1):
        # pass
        # TODO
        # temp_dw = np.dot(hs[i-1].T, djdzs)
        # dJdWs.append(temp_dw)
        # dJdbs.append(djdzs)
        # djdzs = np.dot(djdzs, Ws[i-1].T)
        # print("Ws", Ws[i].shape, bs[i].shape)
        if i == n_layers:
            dhdzs = g
        else:
            dhdzs = relu_diff(zs[i])
            # print("dhdzs", dhdzs.shape)
            # print("else_g", g.shape)
            g = dhdzs * g
            # print("else_g_", g.shape)

        value_app = np.sum(g, axis=1) / n
        # print("djdb", value_app.shape)
        dJdbs.append(value_app)
        if i != 0:
            # print("djdws",np.dot(g, hs[i-1].T).shape)
            dJdWs.append(np.dot(g, hs[i - 1].T) / n)
        else:
            # print("djdws",np.dot(g, x.T).shape)
            dJdWs.append(np.dot(g, x.T) / n)
        ####print(Ws[i].shape)
        g = np.dot(Ws[i].T, g)
    ###print("Loss:",loss)
    ####print(g.shape)
    dJdbs.reverse()
    dJdWs.reverse()
    # Concatenate gradients
    return np.hstack([dJdW.flatten() for dJdW in dJdWs] + [dJdb.flatten() for dJdb in dJdbs])


def accuracy(yhat, y):
    n = y.shape[1]
    # print("yshape", yhat.shape)
    sum = 0
    y_hat_temp = yhat.T
    y_temp = y.T
    for i in range(n):
        if np.argmax(y_hat_temp[i]) == np.argmax(y_temp[i]):
            sum += 1
    return sum * 100 / n


def train(trainX, trainY, weightsAndBiases, testX, testY, n_epochs, n_hidden, n_layers, lr, batch_size, batches,
          batch_list, reg):
    # NUM_EPOCHS = 200
    trajectory = []
    W_new = []
    # l = []
    B_new = []
    batch_num = 0
    n = trainY.shape[1]
    ###print(len(weightsAndBiases))
    # for epoch in range(n_epochs):
    ###print("Epoch:",epoch)
    # TODO: implement SGD.
    # TODO: save the current set of weights and biases into trajectory; this is
    # useful for visualizing the SGD trajectory.
    # batch_list = batch_gen(trainX, trainY, batch_size)
    # idx = 1
    # print("here", trainX.shape, trainY.shape)
    # batches = math.ceil(trainY.shape[1]/batch_size)
    # print("this",batches)
    # for i in range(batches):
    # print("batch_num=", batch_num)
    # batch_num+=1
    # print(len(batch_list))
    # print(batch_list[0][i].shape, batch_list[1][i].shape)
    # print(len(batch_list[0])
    k = back_prop(trainX, trainY, weightsAndBiases, n_hidden, n_layers)
    # l.append(loss)
    # k = back_prop(trainX, trainY, weightsAndBiases)
    # print("here", k.shape)
    dW, dB = unpack(k, n_hidden, n_layers)
    W, B = unpack(weightsAndBiases, n_hidden, n_layers)
    for i in range(len(W)):
        W[i] = W[i] - lr * dW[i] + reg * W[i] / n
        B[i] = B[i] - lr * dB[i]

        # n=batch_list[1][i].shape[0]
        # print(n)
        ####print(W[0].shape, dW[0].shape)
        # print("W before:",W[0])
        # dW = [i*0.0001/n for i in dW]
        # dB = [i*0.0001/n for i in dB]
        # print("after:",dW[0])
        # W = np.asarray([(j-k) for j,k in zip(W,dW)])
        # print("W after:",W[0])
        ###print("W[0]", W[0].shape)
        # B = np.asarray([j-k for j,k in zip(B,dB)])
        # weightsAndBiases = weightsAndBiases - 0.0001*k
        ###print("W_new",len(W_new))
        # B_new.append(B)
    weightsAndBiases = np.hstack([w.flatten() for w in W] + [b.flatten() for b in B])
    trajectory.append(weightsAndBiases)
    '''L,_,_,yhat = forward_prop(testX, testY, weightsAndBiases)
    acc = accuracy(yhat,testY)'''
    ###print("BackProp",len(weightsAndBiases))

    return weightsAndBiases, trajectory


def findbesthyperparameters(X_tr, ytr, X_te, yte):
    # trajectory_dict = {}
    ############ Best Hyper Parameters ##############
    n_layer_list = [3]
    n_hidden_list = [75]
    batch_size_list = [16]
    learning_rate_list = [0.1]
    epochs_list = [40]
    reg_list = [0]
    ############# Used Hyper Parameter List #############

    '''n_layer_list = [3,4,5]
    n_hidden_list = [45, 60, 75]
    batch_size_list = [16, 32]
    learning_rate_list = [0.1, 0.2, 0.05]
    epochs_list = [20, 30, 40]
    reg_list = [0, 0.001, 0.0001]'''

    traj = []
    ll = []
    list_hyper = itertools.product(batch_size_list, learning_rate_list, epochs_list, n_layer_list, n_hidden_list,
                                   reg_list)
    training_x, training_y, valid_x, valid_y = split_data(X_tr, ytr)
    tune_idx = 0
    comb_idx = 0
    for batch_size, lr, n_epochs, n_layers, n_hidden, reg in list_hyper:
        print("combination=", comb_idx, "Batch_size=", batch_size, "learning rate=", lr, "epochs=", n_epochs,
              "n_layers=", n_layers, "n_hidden=", n_hidden, "reg=", reg)
        weightsAndBiases = initWeightsAndBiases(n_hidden, n_layers)
        for epoch in range(n_epochs):
            print("epoch", epoch)
            batch_list = batch_gen(training_x, training_y, batch_size)
            idx = 1
            batch_num = 0
            batches = math.ceil(training_y.shape[1] / batch_size)
            # print("batches=", batches)
            for i in range(batches):
                # print("batch_id", batch_num)
                weightsAndBiases, trajectory = train(batch_list[0][i], batch_list[1][i], weightsAndBiases, X_te, yte,
                                                     n_epochs, n_hidden, n_layers, lr, batch_size, batches, batch_list,
                                                     reg)
                idx += 1
                batch_num += 1
                if i % 40 == 0:
                    traj.extend(trajectory)
                #  ll.append(loss)
            print("Trajec Len:", len(traj))
            L, _, _, yhat = forward_prop(X_te, yte, weightsAndBiases, n_hidden, n_layers)
            print("current_loss=", L)
            acc = accuracy(yhat, yte)
            # print("Accuracy after ", epoch, "epochs is ", acc)
        if tune_idx == 0:
            best_accuracy = acc
            best_comb = comb_idx
            best_b_size = batch_size
            best_lr = lr
            best_epoch = n_epochs
            # best_reg = reg
            best_layers = n_layers
            best_hidden = n_hidden
            print("done this")
            tune_idx += 1
        else:
            if acc > best_accuracy:
                best_accuracy = acc
                best_comb = comb_idx
                best_b_size = batch_size
                best_lr = lr
                best_epoch = n_epochs
                # best_reg = reg
                best_layers = n_layers
                best_hidden = n_hidden
                # print("done this")
        print("current_accuracy=", acc)
        print("current best=", "combination=", best_comb, "Batch_size=", best_b_size, "learning rate=", best_lr,
              "epochs=", best_epoch, "best accuracy=", best_accuracy, "n_layers=", best_layers, "n_hidden=",
              best_hidden)
        comb_idx += 1
        return traj, n_layers, n_hidden


# Performs a standard form of random initialization of weights and biases
def initWeightsAndBiases(n_hidden, n_layers):
    Ws = []
    bs = []

    np.random.seed(0)
    W = 2 * (np.random.random(size=(n_hidden, NUM_INPUT)) / NUM_INPUT ** 0.5) - 1. / NUM_INPUT ** 0.5
    Ws.append(W)
    b = 0.01 * np.ones(n_hidden)
    bs.append(b)

    for i in range(n_layers - 1):
        W = 2 * (np.random.random(size=(n_hidden, n_hidden)) / n_hidden ** 0.5) - 1. / n_hidden ** 0.5
        Ws.append(W)
        b = 0.01 * np.ones(n_hidden)
        bs.append(b)

    W = 2 * (np.random.random(size=(NUM_OUTPUT, n_hidden)) / n_hidden ** 0.5) - 1. / n_hidden ** 0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return np.hstack([W.flatten() for W in Ws] + [b.flatten() for b in bs])


def plotSGDPath(trainX, trainY, trajectory, n_l, n_h):
    # TODO: change this toy plot to show a 2-d projection of the weight space
    # along with the associated loss (cross-entropy), plus a superimposed
    # trajectory across the landscape that was traversed using SGD. Use
    # sklearn.decomposition.PCA's fit_transform and inverse_transform methods.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    pca = PCA(n_components=2)
    zs = pca.fit_transform(trajectory)

    # print(zs)

    def toyFunction(x, y):

        # wandb = pca.inverse_transform(z)
        # for i in range(len(zs)):
        z = [x, y]
        wandb = pca.inverse_transform(z)
        loss, _, _, _ = forward_prop(trainX, trainY, wandb, n_h, n_l)
        # print("ToyLoss:",loss)
        return loss

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # pca = PCA(n_components = 2)
    # zs = pca.fit_transform(trajectory)
    # print(np.amin(zs[:,0]))

    # Compute the CE loss on a grid of points (corresonding to different w).
    axis1 = np.arange(-30, 30, 2)  # Just an example
    axis2 = np.arange(-30, 30, 2)  # Just an example
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    # print(axis1.shape)
    # print(Yaxis)
    # print(axis1.shape)
    Zaxis = np.zeros((len(axis1), len(axis2)))
    for i in range(len(axis2)):
        for j in range(len(axis1)):
            # print(Zax)
            Zaxis[i, j] = toyFunction(Xaxis[i, j], Yaxis[i, j])
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.
    # Zaxis_ = np.zeros(len(Xaxis_))
    # Now superimpose a scatter plot showing the weights during SGD.
    Xaxis_ = zs[:, 0]  # Just an example
    Zaxis_ = np.zeros(len(Xaxis_))
    Yaxis_ = zs[:, 1]  # Just an example
    for i in range(len(Xaxis)):
        Zaxis_[i] = toyFunction(Xaxis_[i], Yaxis_[i])
    ax.scatter(Xaxis_, Yaxis_, Zaxis_, color='r')

    plt.show()


def fun_PCA(traj):
    pca = PCA(n_components=2)
    zs = pca.fit_transform(traj)
    weightandBias = pca.inverse_transform(zs)
    print(weightandBias)


if __name__ == "__main__":
    # TODO: Load data and split into train, validation, test sets
    trainX = np.load("fashion_mnist_train_images.npy") / 255
    trainX = trainX.T
    # print(trainX.shape)
    trainY = np.load("fashion_mnist_train_labels.npy")
    testX = np.load("fashion_mnist_test_images.npy") / 255
    testX = testX.T
    # print(testX.shape)
    testY = np.load("fashion_mnist_test_labels.npy")

    trainY = onehotencoding(trainY).T
    testY = onehotencoding(testY).T
    # print("y shape", trainY.shape, testY.shape)
    # print(trainX[:,0:2500].shape)
    # check grad for 75 hidden units and 4 hidden layers. Was tested for different conditions and it still worked
    weightsAndBiases = initWeightsAndBiases(75, 4)

    # Perform gradient check on random training examples
    print("________________________Check Grad_____________________")
    print(scipy.optimize.check_grad(
        lambda wab: forward_prop(np.atleast_2d(trainX[:, 0:5]), np.atleast_2d(trainY[:, 0:5]), wab, 75, 4)[0], \
        lambda wab: back_prop(np.atleast_2d(trainX[:, 0:5]), np.atleast_2d(trainY[:, 0:5]), wab, 75, 4), \
        weightsAndBiases))
    print("______________________________________________")
    # weightsAndBiases, trajectory = train(trainX, y_train, weightsAndBiases, testX, testY)
    trajectory, n_l, n_h = findbesthyperparameters(trainX, trainY, testX, testY)
    # fun_PCA(trajectory)
    # Plot the SGD trajectory
    shuffle_X, shuffle_Y = shuffle(trainX, trainY)
    plotSGDPath(trainX[:, 0:3500], trainY[:, 0:3500], trajectory, n_l, n_h)
end = time.time()
##print((end-start)/60)
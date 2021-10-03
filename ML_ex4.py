"""
Introduction to Machine Learning - Programming Assignment
Exercise 04
December 2020
Yotam Leibovitz
"""
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

#########################################################################
#                Load Dataset                                           #
#########################################################################
mnist = fetch_openml('mnist_784')
X = mnist['data'].astype('float64')
t = mnist['target']
random_state = check_random_state(1)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
t = t[permutation]
X = X.reshape((X.shape[0], -1))  # This line flattens the image into a vector of size 784
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.4)

# split again to obtain validation set (20% validation, 20% test)
X_validation, X_test, t_validation, t_test = train_test_split(X_test, t_test, test_size=0.5)

# The next lines standardize the images
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_validation = scaler.transform(X_validation)

# add 1 at the of every sample for the bias
X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
X_validation = np.hstack((X_validation, np.ones((X_validation.shape[0], 1))))

# convert from chars to int
t_train = t_train.astype(int)
t_test = t_test.astype(int)
t_validation = t_validation.astype(int)

# create initial random weights
W = np.random.random((785, 10))

#########################################################################
#                Auxiliary Functions                                    #
#########################################################################
def compute_Y(X, t, W):
    A = X @ W
    Y = np.zeros((X.shape[0], 10))
    for i in range(A.shape[0]):
        # prevent overflow by subtracting the max value from each entry in row i
        A[i, :] = A[i, :] - A[i, :].max()
        A[i, :] = np.exp(A[i, :])
        Y[i, :] = A[i, :] / A[i, :].sum()
    return Y


def compute_loss(Y, t):
    y_true = Y[np.arange(0, Y.shape[0], 1), t]
    # set minimum value so log() function won't overflow
    y_true[y_true < 1e-100] = 1e-100
    return np.sum(-1 * np.log(y_true))


def compute_gradient(Y, X, t):
    T = np.zeros((X.shape[0], 10))
    T[np.arange(0, X.shape[0], 1), t] = 1  # t_n_k = the (n,k) entry of matrix T
    gradients = [(X.T @ ((Y - T)[:, [j]])) for j in range(10)]
    grad_E = np.hstack(gradients)
    return grad_E


def compute_accuracy(X, t, W):
    T = np.zeros((X.shape[0], 10))
    T[np.arange(0, X.shape[0], 1), t] = 1  # t_n_k = the (n,k) entry of matrix T
    Y = compute_Y(X, t, W)
    for row in Y[:, :]:
        row[row < row.max()] = -1
        row[row == row.max()] = 1
    correct_classifications = (Y == T).sum()
    accuracy = correct_classifications / X.shape[0]
    return accuracy


def step(X, t, Y, W, step_size):
    grad_E = compute_gradient(Y, X, t)
    W_new = W - step_size * grad_E
    return W_new


def GD_optimizer(X_train, t_train, X_validation, t_validation, X_test, t_test, W, step_size, threshold):
    step_num = 0
    Y_train = compute_Y(X_train, t_train, W)
    loss_list = np.array(())
    val_accuracy_list = np.array(())
    recent_mean_accuracy_diff = 100  # set initial accuracy difference to 100
    # gradient descent
    while np.abs(recent_mean_accuracy_diff) >= threshold:
        step_num += 1
        # update weights
        W = step(X_train, t_train, Y_train, W, step_size)
        # compute new loss
        Y_train = compute_Y(X_train, t_train, W)
        loss_list = np.append(loss_list, compute_loss(Y_train, t_train))
        val_accuracy_list = np.append(val_accuracy_list, compute_accuracy(X_validation, t_validation, W))
        if step_num > 3:
            # compute mean of accuracy difference for the last 3 steps
            recent_mean_accuracy_diff = np.mean(val_accuracy_list[-3:] - val_accuracy_list[-4:-1])

    # compute final accuracy
    train_acc = compute_accuracy(X_train, t_train, W)
    test_acc = compute_accuracy(X_test, t_test, W)
    validation_acc = val_accuracy_list[-1]
    return train_acc, validation_acc, test_acc, loss_list, val_accuracy_list, step_num


#########################################################################
#                Main                                                   #
#########################################################################
if __name__ == "__main__":
    # initialize hyper-parameters
    step_size = 0.05
    threshold = 0.0005

    # run the gradient descent optimizer
    train_acc, validation_acc, test_acc, loss_list, val_accuracy_list, step_num = GD_optimizer(
        X_train,
        t_train,
        X_validation,
        t_validation,
        X_test, t_test,
        W, step_size,
        threshold)

    print(
        "**** MNIST Multiclass Logistic Regression Results ****\nHyper-parameters :\n       step size = {4}\n"
        "       threshold = {5}\nNumber of steps until convergence = {0} steps"
        "\nFinal training accuracy: {1} \nFinal validation accuracy: {2} \nFinal test accuracy: {3}\n\n\n".format(
            step_num, train_acc, validation_acc, test_acc, step_size, threshold))

    # plot the results
    x = np.arange(1, step_num + 1, 1)
    fig = plt.figure()

    ax = fig.add_subplot(211)
    ax.set(title='Loss per iteration', xlabel='Iteration', ylabel='E(W)')
    ax.plot(x, loss_list, linewidth=2, c='C3', marker='o')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    ax = fig.add_subplot(212)
    ax.plot(x, 100*val_accuracy_list, linewidth=2, c='C3', marker='o')
    ax.set(title='Validation accuracy per iteration', xlabel='Iteration', ylabel='Accuracy')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.tight_layout()
    plt.savefig('ML_ex4_plots.png')
    plt.show()


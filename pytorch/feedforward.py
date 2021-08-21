import numpy as np
from numpy.core.fromnumeric import reshape

np.random.seed(1)
X = np.random.rand(12288, 200)
Y = np.random.rand(1, 200)

n0, m = X.shape
n1 = 20
n2 = 7
n3 = 5
n4 = 1
layer_dims = [n0, n1, n2, n3, n4]
print(layer_dims)
L = len(layer_dims) - 1

def sigmoid(z) :
    return 1 / (1 + np.exp(-z))

def relu(z) :
    return np.maximum(0, z)

def neural_network(X, Y, learning_rate=0.01, num_iterations=2000,lambd=0) :
    m = X.shape[1]
    print(X.shape)
    param_w = [i for i in range(L + 1)]
    param_b = [i for i in range(L + 1)]
    print(param_w)
    print(param_b)
    np.random.seed(10)
    for l in range(1, L + 1) :
        if l < L :
            param_w[l] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        if l == L :
            param_w[l] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        param_b[l] = np.zeros((layer_dims[l], 1))
    print(param_w[1].shape)    

    activations = [X,] + [i for i in range(L)]
    print(activations)
    prev_activations = [i for i in range(L + 1)]
    print(prev_activations)

    dA = [i for i in range(L + 1)]
    dz = [i for i in range(L + 1)]
    dw = [i for i in range(L + 1)]
    db = [i for i in range(L + 1)]

    for i in range(num_iterations) :
        for l in range(1, L + 1) :
            prev_activations[l] = np.dot(param_w[l], activations[l - 1]) + param_b[l]
            if l < L :
                activations[l] = relu(prev_activations[1])
            else :
                activations[l] = sigmoid(prev_activations[1])
        cross_entropy_cost = -1/m * (np.dot(np.log(activations[L]), Y.T) \
                                     + np.dot(np.log(1-activations[L]), 1-Y.T))
        regularization_cost = 0
        for l in range(1, L+1):
            regularization_cost += np.sum(np.square(param_w[l])) * lambd/(2*m)
        cost = cross_entropy_cost + regularization_cost

        ### initialize backward propagation
        dA[L] =  np.divide(1-Y, 1-activations[L]) - np.divide(Y, activations[L])
        assert dA[L].shape == (1, m)

        ### backward propagation
        for l in reversed(range(1, L+1)):
            if l == L:
                dz[l] = dA[l] * activations[l] * (1-activations[l])
            else:
                dz[l] = dA[l].copy()
                dz[l][prev_activations[l] <= 0] = 0

            dw[l] = 1/m * np.dot(dz[l], activations[l-1].T) + param_w[l] * lambd/m
            db[l] = 1/m * np.sum(dz[l], axis=1, keepdims=True)
            dA[l-1] = np.dot(param_w[l].T, dz[l])

            assert dz[l].shape == prev_activations[l].shape
            assert dw[l].shape == param_w[l].shape
            assert db[l].shape == param_b[l].shape
            assert dA[l-1].shape == activations[l-1].shape

            param_w[l] = param_w[l] - learning_rate * dw[l]
            param_b[l] = param_b[l] - learning_rate * db[l]

        if i % 100 == 0:
            print("cost after iteration {}: {}".format(i, cost))

def predict(X_new, parameters, threshold=0.5):
    param_w = parameters["param_w"]
    param_b = parameters["param_b"]

    activations = [X_new, ] + [i for i in range(L)]
    prev_activations = [i for i in range(L + 1)]
    m = X_new.shape[1]

    for l in range(1, L + 1):
        prev_activations[l] = np.dot(param_w[l], activations[l - 1]) + param_b[l]
        if l < L:
            activations[l] = relu(prev_activations[l])
        else:
            activations[l] = sigmoid(prev_activations[l])
    prediction = (activations[L] > threshold).astype("int")
    return prediction
    
neural_network(X, Y)
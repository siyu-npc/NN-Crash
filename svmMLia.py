import random
import numpy as np
from numpy.core.defchararray import multiply
import matplotlib.pyplot as plt

def loadDataSet(filename) :
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines() :
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m) :
    j = i
    while (j == i) :
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, H, L) :
    if aj > H :
        aj = H
    if L > aj :
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter) :
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while (iter < maxIter) :
        alphaPairsChanged = 0
        for i in range(m) :
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
                ((labelMat[i] * Ei > toler) and (alphas[i] > 0)) :
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJoid = alphas[j].copy()
                if (labelMat[i] != labelMat[j]) :
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else :
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H :
                    print("L == H")
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                    dataMatrix[i, :] * dataMatrix[i, :].T - \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0 :
                    print("eta >= 0")
                    continue
                alphas[i] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJoid) < 0.00001) :
                    print("j not moving enough")
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJoid - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                    dataMatrix[i, :] * dataMatrix[i, :].T - \
                    labelMat[j] * (alphas[j] - alphaJoid) * \
                    dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                    dataMatrix[i, :] * dataMatrix[j, :].T - \
                    labelMat[j] * (alphas[j] - alphaJoid) * \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]) :
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]) :
                    b = b2
                else :
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: {} i : {}, pairs changed {}".format(iter, i, alphaPairsChanged))
            if (alphaPairsChanged == 0) : iter += 1
            else : iter = 0
            print("iteration number : {}".format(iter))
        return b, alphas
def plotBestFit(alphas) :
    weights = []; b = 0
    dataMat, labelMat = loadDataSet('Ch06_testSet.txt')
    dataArr = np.array(dataMat)
    w1 = 0; w2 = 0
    for i in range(len(alphas)) :
        if (alphas[i] >= 0) :
            w1 += (dataArr[i, 0] * labelMat[i] * alphas[i])
            w2 += (dataArr[i, 1] * labelMat[i] * alphas[i])
    weights = np.array([w1.getA()[0][0], w2.getA()[0][0]])
    print("weights = {}".format(weights))
    for i in range(len(alphas)) :
        if (alphas[i] > 0 ) :
            b = labelMat[i] - (w1 * dataArr[i, 0] + w2 * dataArr[i, 1]) 
            break
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n) :
        if int(labelMat[i]) == 1 :
            xcord1.append(dataArr[i, 0])
            ycord1.append(dataArr[i, 1])
        else :
            xcord2.append(dataArr[i, 0])
            ycord2.append(dataArr[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
    ax.scatter(xcord2, ycord2, s = 30, c = 'green')
    xs = np.arange(-3.0, 3.0, 0.1)
    ys = (-b - weights[0] * xs) / weights[1]
    print('ys = {}'.format(np.shape(ys.T)))
    
    ax.plot(xs, ys.T)
    plt.xlabel('X1')
    plt.ylabel('X2')  
    plt.show()    

if __name__ == '__main__' :
    dataArr, labelArr = loadDataSet('Ch06_testSet.txt')
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 100)
    print("b = {}".format(b))
    print("alphas = {}".format(alphas[alphas>0]))
    plotBestFit(alphas)
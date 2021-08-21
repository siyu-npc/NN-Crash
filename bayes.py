from numpy import *
import numpy as np

def loadDataSet() :
    postingList = [['my', 'dog', 'has', 'flea','problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how','to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1] # 1代表侮辱性文字，0代表正常言论
    return postingList, classVec

def createVocabList(dataSet) :
    vocabSet = set([])
    for document in dataSet :
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet) :
    returnVec = [0] * len(vocabList)
    for word in inputSet :
        if word in vocabList :
            returnVec[vocabList.index(word)] = 1
        else :
            print("The word: {} is not in my Vocabulary!".format(word))
    return returnVec

def trainNB0(trainMatrix, trainCategory) :
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs) :
        if trainCategory[i] == 1 :
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else :
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1) :
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p2 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p2 :
        return 1
    else :
        return 0

if __name__ == '__main__' :
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts :
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    print("p0V : \n {}".format(p0V))
    print("p1V : \n {}".format(p1V))
    print("pAb : \n {}".format(pAb))

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print("{} classified as: {}".format(testEntry, classifyNB(thisDoc, p0V, p1V, pAb)))

    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print("{} classified as: {}".format(testEntry, classifyNB(thisDoc, p0V, p1V, pAb)))
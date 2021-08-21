import numpy as np
from numpy.core.defchararray import array

'''
计算基金定投收益
'''
def aipProfit(perValue, periods, rate) :
    inputs = [perValue for i in range(periods)]
    print(inputs)
    period = [pow(1 + rate, i + 1) for i in range(periods)]
    print(period)
    print(np.convolve(inputs, period)[periods - 1])

aipProfit(2000, 36, 0.0315/12)
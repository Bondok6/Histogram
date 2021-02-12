from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt


flower = cv2.imread("flower.jpg")
flower = cv2.resize(flower, (800, 600))
s = flower.shape

flowerGray = cv2.cvtColor(flower, cv2.COLOR_BGR2GRAY)
flowerGray = cv2.convertScaleAbs(flowerGray, alpha=1.10, beta=40)
cv2.imshow('original', flowerGray)
cv2.waitKey(0)


def Hist(image):
    H = np.zeros(shape=(265, 1))
    s = image.shape
    for i in range(s[0]):
        for j in range(s[1]):
            k = image[i, j]
            H[k, 0] = H[k, 0] + 1

    return H


histg = Hist(flowerGray)
plt.plot(histg)
plt.show()


x = histg.reshape(1, 265)
y = np.zeros((1, 265))

for i in range(265):
    if x[0, i] == 0:
        y[0, i] = 0
    else:
        y[0, i] = i

min = np.min(y[np.nonzero(y)])
max = np.max(y[np.nonzero(y)])

strech = np.round(((255 - 0) / (max - min)) * (y - min))
strech[strech < 0] = 0
strech[strech > 255] = 255

for i in range(s[0]):
    for j in range(s[1]):
        k = flowerGray[i, j]
        flowerGray[i, j] = strech[0, k]

histg2 = Hist(flowerGray)
cv2.imshow('mysteching', flowerGray)
plt.plot(histg)
plt.plot(histg2)
plt.show()

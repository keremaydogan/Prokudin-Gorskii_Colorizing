import os

import numpy as np
import scipy
import skimage.io as skimgIO

def removeWhiteBorder(img:np.ndarray, fileName:str):
    # p = img[10,10] # holds value of background pixel value
    newBorders = [0,0,0,0] # array to hold background offset [u,d,l,r]
    xMid = int(len(img[0])/2)
    yMid = int(len(img)/2)

    # # checks upper offset of p colored background
    for i in range(0, len(img)): # up (0 to 1024)
        if(img[i,xMid] < 100):
            newBorders[0] = i
            break
    for i in range(len(img)-1, -1, -1): # down (1024 to 0)
        if(img[i,xMid] < 100):
            newBorders[1] = i
            break
    for i in range(0, len(img[0])): # left (0 to 398)
        if(img[yMid, i] < 100):
            newBorders[2] = i
            break
    for i in range(len(img[0])-1, -1, -1): # right (398 to 0)
        if(img[yMid, i] < 100):
            newBorders[3] = i
            break

    img = img[newBorders[0]:newBorders[1], newBorders[2]:newBorders[3]]
    skimgIO.imsave("./cropped/" + fileName, img)
    return img


def corr(A:np.ndarray, B:np.ndarray):
    maxCorr = scipy.signal.correlate2d(A,B)
    for i in range(0, len(maxCorr)):
        for j in range(0, len(maxCorr[0])):
            maxCorr[i][j]

    return maxCorr


def alignRGB(img:np.ndarray):
    height = int(len(img)/3)
    B = img[0:height]
    G = img[height:2*height]
    R = img[2*height:3*height]

    aligned = np.dstack([R,G,B])
    return aligned


fileNames = os.listdir("./data") # Get file names
#'./data/00056v.jpg'
# img = skimgIO.imread("./data/00056v.jpg")

for fileName in fileNames:
    img = skimgIO.imread("./data/"+fileName)
    img = removeWhiteBorder(img, fileName)
    img = alignRGB(img)
    skimgIO.push(img)
    skimgIO.show()
    break
    skimgIO.imsave("./output/" + fileName, img)


#for fileName in fileNames:
#'./data/00056v.jpg'
#align_rgb(fileNames[0])




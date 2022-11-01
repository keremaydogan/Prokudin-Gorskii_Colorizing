import numpy
import skimage.color


import os, subprocess
import skimage.io as skiIO

import numpy as np
import skimage.color as skiColor

from skimage.registration import phase_cross_correlation
import skimage.exposure

def removeWhiteBorder(img:np.ndarray, fileName:str):
    # p = img[10,10] # holds value of background pixel value
    newBorders = [0,0,0,0] # array to hold background offset [u,d,l,r]
    xMid = int(len(img[0])/2)
    yMid = int(len(img)/2)

    # # checks upper offset of p colored background
    for i in range(0, len(img)): # up (0 to 1024)
        if(img[i,xMid] < 220):
            newBorders[0] = i
            break
    for i in range(len(img)-1, -1, -1): # down (1024 to 0)
        if(img[i,xMid] < 220):
            newBorders[1] = i
            break
    for i in range(0, len(img[0])): # left (0 to 398)
        if(img[yMid, i] < 220):
            newBorders[2] = i
            break
    for i in range(len(img[0])-1, -1, -1): # right (398 to 0)
        if(img[yMid, i] < 220):
            newBorders[3] = i
            break

    img = img[newBorders[0]:newBorders[1], newBorders[2]:newBorders[3]] # white borders cropped version
    skiIO.imsave("./cropped/" + fileName, img)
    return img


def offset(A:np.ndarray,B:np.ndarray):
    off = phase_cross_correlation(A, B)[0]
    offInt = [int(off[0]), int(off[1])]

    return offInt


def shiftArr(arr:np.ndarray, off:np.ndarray):
    arr = np.roll(arr, off[0], axis=0)
    arr = np.roll(arr, off[1], axis=1)

    return arr


def alignRGB(img:np.ndarray):

    height = int(len(img)/3) # find height of each channel

    # get BGR channels
    B = img[0:height]
    G = img[height:2*height]
    R = img[2*height:3*height]

    # Shift chanels by offset ========
    gOffset = offset(B,G)
    rOffset = offset(B,R)

    G = shiftArr(G, gOffset)
    R = shiftArr(R, rOffset)
    # ================================

    return np.dstack([R,G,B])


# ================================================================
# ================================================================


def gammaCorrection(img:np.ndarray, gamma:float):
    return np.uint8(255.0 * (img / 255.0) ** (gamma))

def histogramEquializaiton(img:np.ndarray):
    hsvimg = skiColor.rgb2hsv(img)
    val = hsvimg[:,:,2]

    val = skimage.exposure.equalize_hist(val)

    hsvimg[:,:,2] = val
    img = skiColor.hsv2rgb(hsvimg)

    return img


def laplacianFilter8w(img:np.ndarray, coeff:float):

    hsvimg = skiColor.rgb2hsv(img)

    val = hsvimg[:,:,2]
    newVal = np.ndarray(val.shape, dtype=val.dtype)

    for i in range(1, len(val)-1):
        for j in range(1, len(val[0])-1):
            newVal[i][j] = (1.0+coeff*8) * val[i][j] - coeff*(val[i][j+1] + val[i][j-1] + val[i+1][j] + val[i-1][j]
                                                        + val[i+1][j+1] + val[i+1][j-1] + val[i-1][j+1] + val[i-1][j-1])

    newVal = newVal.clip(0.0,1.0)

    hsvimg[:, :, 2] = newVal
    img = skiColor.hsv2rgb(hsvimg)
    img = (img*255)
    img = numpy.ndarray.astype(img, "uint8")
    return img


def enhanceOpsInterface(img:np.ndarray, operation:str, func):

    skiIO.imshow(img)
    skiIO.show()

    result = img
    while True:

        ans = input("["+operation+"]\nEnter γ as float\n('d': done):\n")
        if(ans.__eq__("d")):
            return result
        try:
            value = float(ans)

            result = func(img,value)

            con = np.concatenate((img,result), axis=1)
            skiIO.imshow(con)
            skiIO.show()
        except ValueError:
            print("[INVALID INPUT]")



fileNames = os.listdir("./data") # Get file names

'./data/000911v.jpg'
fileName = "00911v.jpg"
img = skiIO.imread("./data/" + fileName)
img = removeWhiteBorder(img, fileName)
img = alignRGB(img)
skiIO.imshow(img)
skiIO.imsave("./aligned/" + fileName, img)

histogramEquializaiton(img)

for fileName in fileNames:

    img = skiIO.imread("./data/"+fileName)

    img = removeWhiteBorder(img, fileName)
    img = alignRGB(img)

    # ================================================================
    # ================================================================

    img = gammaCorrection(img, 1.6)
    img = laplacianFilter8w(img, 0.15)


    print(fileName + " done")
    skiIO.imsave("./enhanced/" + fileName, img)

subprocess.Popen(f'explorer {os.path.realpath("./enhanced")}')

# DECIDE FUNC COEFFS
# GAMMA AND LAPLACE COMBO IS ENOUGH
# CLEAR UNUSED CODES AND COMMENTS
# DONT WASTE TOO MUCH TIME ON ONE
# START TO WRITE REPORT




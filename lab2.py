#!/usr/bin/python

import time
from imageManipulation import *

def normalize(gradient, toSave):
    (h,w,s) = gradient.shape
    biggest = np.amax(gradient)
    smallest = np.amin(gradient)
    lenght = ( ( smallest+fabs(smallest) ) + (biggest+fabs(smallest) ) )
    for x in range(h):
        for y in range(w):
            gradient[x,y] = (gradient[x,y] - np.array([smallest, smallest, smallest]) ) / np.array([lenght,lenght,lenght]) \
                * np.array([255,255,255])
    im = Image.fromarray(np.uint8(pixels))
    im.save(toSave)
    return gradient

imagePath = 'circles.png'
img = Image.open(imagePath).convert('RGB')

iM = ImageManipulation()
pixels = np.array(img)

(iM.w,iM.h) = img.size

maskToUse = 'p' #'[rsp]'

mask1, mask2 = iM.getMaskForGradient(maskToUse)

timeStart = time.time()

gradientX = iM.convolution(pixels, mask1) # mask in X
timeGX = time.time()
gradientY = iM.convolution(pixels, mask2) # mask in y
timeGY = time.time()
pixels = iM.getMagnitude(gradientX, gradientY)
timeG = time.time()

timeEnd = time.time()

pixelsInBorder = iM.getPixelsInBorder(pixels)
iM.pixelsInBorder = pixelsInBorder
print 'Number of pixels in border:', len(pixelsInBorder)
print 'Total time:', (timeEnd - timeStart)
print 'Time to get gradient X:', (timeGX - timeStart)
print 'Time to get gradient Y:', (timeGY - timeGX)
print 'Time to get magnitude:', (timeG - timeGY)

normalize(gradientX, 'results/prewitt-x.png')
normalize(gradientY, 'results/prewitt-y.png')

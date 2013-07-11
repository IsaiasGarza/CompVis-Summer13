#!/usr/bin/python

from PIL import Image, ImageDraw
from math import floor
from filters import *
from imageManipulation import *
import numpy as np
import sys

# corner detection using median filter

WHITE = np.array([255,255,255])
BLACK = np.array([0,0,0])
GRAY28 = np.array([71,71,71])
RED = np.array([255,0,0])
GREEN = np.array([0,255,0])
BLUE = np.array([0,0,255])
ORANGE = np.array([255,165,0])
YELLOW = np.array([255,255,0])

def medianFilter(pixels):
    h,w,_ = pixels.shape
    newPixels = np.zeros(pixels.shape)

    for x in xrange(h):
        for y in xrange(w):
            neighbors = list()

            for dx in [-1,0,1]:
                for dy in [-1,0,1]:

                    try:
                        neighbors.append( pixels[x+dx,y+dy][0] )
                    except: continue
            neighbors.sort()
            neighbors.reverse()
            median = neighbors[int( len(neighbors)/2 )]
            newPixels[x,y] = np.array([median, median, median])
    return newPixels

def imgDiff(img1, img2):
    h,w,_ = img1.shape
    newImg = np.zeros(img1.shape)

    for x in xrange(h):
        for y in xrange(w):
            res = (img1[x,y] - img2[x,y]).astype(int)
            newImg[x,y] = np.absolute(res)
    return newImg



def main():

    try:
        pathToImage = sys.argv[1]
    except IndexError:
        pathToImage = 'testImage.png'
        print 'No image specified, using', pathToImage
    im = Image.open(pathToImage).convert('RGB')
    originalImage = im.copy()
    (w,h) = im.size
    print 'Image size (w,h):', w,h
    pixels = np.array(im)
    
    iM = ImageManipulation()
    fil = Filters()

    gray = fil.grayscale(pixels)
    pixels = medianFilter(gray)
    dif = iM.normalize(imgDiff(gray,pixels))


    oImg = np.array(originalImage)
    for x,y in iM.getPixelsInBorder(dif):
        oImg[x,y] = ORANGE


    #im = Image.fromarray(np.uint8(gray)) # original image in grayscale
    #im.show()

    #im = Image.fromarray(np.uint8(pixels)) # image after median filter
    #im.show()

    im = Image.fromarray(np.uint8(dif)) # image after image difference
    im.show()

    im = Image.fromarray(np.uint8(oImg)) # original image with corners in another color
    im.show()



if __name__ == '__main__':
    main()


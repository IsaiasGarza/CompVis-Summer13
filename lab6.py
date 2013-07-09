#!/usr/bin/python

from PIL import Image
from math import floor
from filters import *
from imageManipulation import *
import numpy as np
import sys

BLACK = np.array([0,0,0])

def shrink(pixels):
    h,w,_ = pixels.shape
    newImg = np.zeros( (h, w, 3) )
    for x in xrange(h):
        for y in xrange(w):
            sigma = 0
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    try:
                        if (pixels[x+dx,y+dy] == BLACK).all():
                            sigma += 1
                    except: pass
                    if sigma < 8:
                        newImg[x,y] = np.array([255,255,255])
                    else:
                        newImg[x,y] = pixels[x,y]
    return newImg

def expand(pixels):
    h,w,_ = pixels.shape
    newImg = np.zeros( (h, w, 3) )
    for x in xrange(h):
        for y in xrange(w):
            sigma = 0
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    try:
                        if (pixels[x+dx,y+dy] == BLACK).all():
                            sigma += 1
                    except: pass
                    if sigma > 0:
                        newImg[x,y] = np.array([0,0,0])
                    else:
                        newImg[x,y] = pixels[x,y]
    return newImg

if __name__ == '__main__':

    try:
        pathToImage = sys.argv[1]
    except IndexError:
        pathToImage = 'testImage.png'
        print 'No image specified, using', pathToImage
    im = Image.open(pathToImage).convert('RGB')
    originalImage = im.copy()
    (w,h) = im.size
    print 'Image size:', w,h
    pixels = np.array(im)
    
    iM = ImageManipulation()
    fil = Filters()
    
    pixels = fil.grayscale(pixels)
#    pixels = fil.threshold(pixels,128)
    pixels = fil.polarize(pixels)
#    im = Image.fromarray(np.uint8(pixels))
#    im.show()
    
    pixels = iM.detectBorder(pixels)
#    pixels = fil.polarize(pixels)    

    if raw_input() == 'expand':
        pixels = shrink(pixels)
    else:
        pixels = expand(pixels)

    im = Image.fromarray(np.uint8(pixels))
    im.show()


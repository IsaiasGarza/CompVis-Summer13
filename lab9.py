#!/usr/bin/python

from PIL import Image, ImageDraw
from math import floor
from filters import *
from imageManipulation import *
import numpy as np
import sys

# polygon detection

WHITE = np.array([255,255,255])
BLACK = np.array([0,0,0])
GRAY28 = np.array([71,71,71])
RED = np.array([255,0,0])
GREEN = np.array([0,255,0])
BLUE = np.array([0,0,255])
ORANGE = np.array([255,165,0])
YELLOW = np.array([255,255,0])

def detectLines(pixels, gradX, gradY, discr=10):
    h,w,_ = pixels.shape
    bias = 10
    freq = dict()
    lines = dict()
    for x in xrange(h):
        for y in xrange(w):
            gX = sum( gradX[x,y][list(1 * n for n in xrange(3))] )/3.0
            gY = sum( gradY[x,y][list(1 * n for n in xrange(3))] )/3.0

            if (gX < -1*bias or gX > bias) or (gY < -1*bias or gY > bias):
                theta = 0.0
                if gX > 0.0 and gY == 0.0:
                    theta = 0.0
                elif gX < 0.0 and gY == 0.0:
                    theta = 180.0
                if gX == 0.0 and gY > 0.0:
                    theta = 90.0
                elif gX == 0.0 and gY < 0.0:
                    theta = 270.0
                else:
                    theta = atan2(gY*1.0,gX*1.0) # radians range is from -pi to pi
                    theta = degrees(theta)  # convert angle to degree
                    theta = (int(ceil(theta))/discr)*discr # round to next degree, int
                    
                rho = (x * cos(theta)) + (y * sin(theta)) # this is the distance from the center

                lines[x,y] = (theta, rho)

                if not (theta, rho) in freq:
                    freq[(theta,rho)] = 1
                else:
                    freq[(theta,rho)] += 1
            else:
                lines[x,y] = None
    verticals = 0
    horizontals = 0

    for i in xrange(h):
        for j in xrange(w):
            if lines[i,j] in freq:
                if lines[i,j][0] == 0.0 or lines[i,j][0] == 180.0:
                    horizontals += 1
                    pixels[i,j] = RED
                elif lines[i,j][0] == 90.0 or lines[i,j][0] == 270.0:
                    verticals += 1
                    pixels[i,j] = BLUE
                #else:
                    #pixels[i,j] = ORANGE
    return lines, freq

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
    
    # convert to grayscale
    pixels = fil.grayscale(pixels)

    # detect borders
    pixels = iM.detectBorder(pixels)

    pixelsInBorder = iM.pixelsInBorder
    gX = iM.gX
    gY = iM.gY

    # group borders depending on angles of the pixels using bfs/dfs


    im = Image.fromarray(np.uint8(pixels)) # original image with corners in another color
    im.show()


if __name__ == '__main__':
    main()


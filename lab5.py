#!/usr/bin/python

from PIL import Image
from math import floor
import numpy as np
import sys

def resize(oldPixels, (nW, nH)):
    if nW <= 0 or nH <= 0:
        print 'Cant resize to smaller than zero'
        return False
    ratioW = (w * 1.0) / nW# ; print 'ratio w', ratioW
    ratioH = (h * 1.0) / nH# ; print 'ratio h', ratioH

    pixels = np.zeros( (nH, nW, 3) )

    for x in xrange(nH):
        for y in xrange(nW):
            xPos = floor( x * ratioH)
            yPos = floor( y * ratioW)
                
            try:
                #print 'Getting value for', x, y, 'giving', xPos, yPos
                pixels[x,y] = oldPixels[int(xPos), int(yPos)]
            except IndexError:
                print '\nProblem at X:', x, 'Y:', y
                print 'xPos:', xPos, 'yPos', yPos
                print pixels.size
                print pixels.shape
                print nW*ratioW
                print nH*ratioH
                return oldPixels
    #self.w = nW
    #self.h = nH
    return pixels

try:
    pathToImage = sys.argv[1]
except IndexError:
    pathToImage = 'testImage.png'
    print 'No image specified, using', pathToImage
im = Image.open(pathToImage).convert('RGB')
im.show()
originalImage = im.copy()
(w,h) = im.size
print 'Image size:', w,h
pixels = np.array(im)
#print pixels.shape

newSize = str(raw_input('New size (e.g. width height): '))
newSize = newSize.split(' ', 1)
try:
    newW = int(newSize[0])
    newH = int(newSize[1])
    pixels = resize(pixels, (newW,newH))
except ValueError:
    print 'Please try again with the suggested format'
except IndexError:
    print 'Please try again with the suggested format'

im = Image.fromarray(np.uint8(pixels))
im.show()



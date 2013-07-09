#!/usr/bin/python

from PIL import Image, ImageDraw
from math import floor
from filters import *
from imageManipulation import *
import numpy as np
import sys

BLACK = np.array([0,0,0])

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
    
    pixels = fil.grayscale(pixels)
    objs = iM.detectForms(pixels) #objs contains array of coordinates, each array represents an object

    im = Image.fromarray(np.uint8(pixels))
    draw = ImageDraw.Draw(im)

    for o in objs:
        Xmini = Xmaxi = int(w/2)
        Ymini = Ymaxi = int(h/2)
        for coord in o:
            y,x = coord
            if x > Xmaxi:
                Xmaxi = x
            elif x < Xmini:
                Xmini = x

            if y > Ymaxi:
                Ymaxi = y
            elif y < Ymini:
                Ymini = y
        print 'UpperLeft',(Xmini,Ymini), 'BottomRight',(Xmaxi,Ymaxi)
        draw.rectangle(((Xmini,Ymini),(Xmaxi,Ymaxi)), outline=(0,255,0))

    #im = Image.fromarray(np.uint8(pixels))
    im.show()

if __name__ == '__main__':
    main()


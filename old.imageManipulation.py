#!/usr/bin/python

import sys
import numpy as np
from PIL import Image
from random import random
from math import floor, fabs
import os

w,h = 0,0
originalImage = None

def grayscale(pixels):
    for x in xrange(h):
        for y in xrange(w):
            #gray = sum(pixels[x,y][list(1 * n for n in xrange(-4,-1))])/3
            #A = pixels[x,y][-1]
            (R,G,B,A) = pixels[x,y]
            gray = (int(R)+int(G)+int(B))/3
            pixels[x,y] = (gray, gray, gray, A)
    return pixels

def resize(oldPixels, (nW,nH)):
    global w,h
    if nW <= 0 or nH <= 0:
        print 'Cant resize to smaller than zero'
        return False

    ratioW = (w * 1.0) / nW# ; print 'ratio w', ratioW
    ratioH = (h * 1.0) / nH# ; print 'ratio h', ratioH


    pixels = np.zeros( (nH, nW, 4) )

    for x in xrange(nH):
        for y in xrange(nW):

            xPos = floor(x * ratioH)
            yPos = floor(y * ratioW)

            try:
                #print 'Getting value for', x, y, 'giving', xPos, yPos
                pixels[x,y] = oldPixels[int(xPos),  int(yPos)]
            except IndexError:
                print '\nProblem at X:', x, 'Y:', y
                print 'xPos:', xPos, 'yPos', yPos
                print pixels.size
                print pixels.shape

                print nW*ratioW
                print nH*ratioH
                return oldPixels
    w = nW
    h = nH
    return pixels

def addNoise(pixels, probab):
    # probab is a number between 0 and 1
    # indicates the probability of noise appearing in a pixel
    total = 0
    for x in xrange(h):
        for y in xrange(w):
            if random() < probab: # true add noise
                total += 1
                if random() < 0.5: # black noise
                    pixels[x,y] = (0,0,0,255)
                else: #white noise
                    pixels[x,y] = (255,255,255,255)
    print 'Modified', total, 'pixels with noise'
    return pixels

def removeNoise(pixels): # to be implemented
    for x in xrange(h):
        for y in xrange(w):
            1+1
            # promedio de vecinos
            # compara valor de pixel con promedio de vecinos
            # asigna promedio cuando necesario
    return pixels

def getMaskForGradient(m):
    if m == 'r': # Roberts Cross
        rx = np.array([[0,1],[-1,0]])
        ry = np.array([[1,0],[0,-1]])
        return rx,ry
    elif m == 's': # Sobel
        sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        sy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        return sx,sy
    elif m == 'p': # Prewitt
        px = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        py = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        return px,py
    else:
        return False

def getMaskForTemplateMatching(m):
    if m == 'prewitt':
        zero = [[-1,1,1],[-1,-2,-1],[-1,1,1]]
        fortyfive = [[1,1,1],[-1,-2,1],[-1,-1,1]]
        return zero,fortyfive
    elif m == 'kirsch':
        zero = [[-3,-3,5],[-3,0,5],[-3,-3,5]]
        fortyfive = [[-3,5,5],[-3,0,5],[-3,-3,-3]]
        return zero,fortyfive
    else:
        return False

def convolution(pixels, mask):
    maskH, maskW = mask.shape
    newPixels = np.zeros(pixels.shape)
    halfM = maskH >> 1 # division by 2
    for x in xrange(h):
        for y in xrange(w):
            mSum = np.array([0.0, 0.0, 0.0, 0.0])
            for i in xrange(maskH):
                centerI = i - halfM
                for j in xrange(maskW):
                    centerJ = j - halfM
                    try:
                        mSum += pixels[x+centerI,y+centerJ] * mask[i,j]
                    except IndexError: pass
                newPixels[x,y] = mSum
                print mSum
    im = Image.fromarray(np.uint8(newPixels))
    im.save('convolution-output.png')
    return newPixels

def getMagnitude(gradient1, gradient2):
    if gradient1.shape != gradient2.shape:
        exit('Unable to proceed, gradients with different shapes')
    gradientH, gradientW, _ = gradient1.shape
    newPixels = np.zeros(gradient1.shape)
    for x in xrange(gradientH):
        for y in xrange(gradientW):
            newPixels[x,y] = (np.fabs(gradient1[x,y]) + np.fabs(gradient2[x,y])).astype(int)
    #np.fabs(gradient[x,y][list(n for n in xrange(4))]).astype(int)
    print 'Looks like its done'
    im = Image.fromarray(np.uint8(newPixels))
    im.save('gradient-magnitude.png')
    return newPixels

def normalize(toNormal):
    #Val1 = (Val0 - minVal)/longitud
    print toNormal.shape
    print np.amax(toNormal)
    print np.amin(toNormal)

def main():
    global w,h, originalImage
    try:
        pathToImage = sys.argv[1]
    except IndexError:
        exit('Please specify a valid path to the file\n')
        
    im = Image.open(pathToImage).convert('RGBA')
    originalImage = im.copy()

    (w,h) = im.size
    print 'Image size:', w,h
    pixels = np.array(im)
    print np.array(im).shape

    action = ''
    options = '\n What do you want to do?\n' + \
        '    g : image to grayscale\n' + \
        '    r : resize image\n' + \
        '    n : add noise\n' + \
        '    m : apply a mask\n' + \
        '    s : save to file\n' + \
        '    q : quit the program\n'

    while action != 'q':
        action = raw_input(options)
        action = str(action)
        #os.system('cls' if os.name == 'nt' else 'clear')
        if action == 'g':
            pixels = grayscale(pixels)
        elif action == 'r':
            newSize = str(raw_input('New size (e.g. width height): '))
            newSize = newSize.split(' ', 1)
            try:
                newW = int(newSize[0])
                newH = int(newSize[1])
                pixels = resize(pixels, (newW,newH))
            except ValueError:
                print 'Please try again with the suggested format'
                action = 'noShow'
            except IndexError:
                print 'Please try again with the suggested format'
                action = 'noShow'
        elif action == 'n':
            noise = raw_input('From 0 to 1, how much noise?:')
            try:
                float(noise)
                pixels = addNoise(pixels, 0.15)
            except ValueError:
                print 'Please enter a valid value'
                action = 'noShow'
        elif action == 'm':
            print 'Available masks:\n' + \
                ' r : Roberts\n' + \
                ' s : Sobel\n' + \
                ' p : Prewitt\n'
            desiredMask = str(raw_input()).strip('\n')
            try:
                mask1,mask2 = getMaskForGradient(desiredMask)
                gradientX = convolution(pixels, mask1) # mask in X
                gradientY = convolution(pixels, mask2) # mask in y

                getMagnitude(gradientX,gradientY)
                
                action = 'noShow'
            except TypeError as e:
                print 'Please specify only an option from the list', e
                action = 'noShow'
        elif action == 's':
            im.save('output.png')
            action = 'noShow'
        elif action == 'q':
            os.system('cls' if os.name == 'nt' else 'clear')
            exit()
        im = Image.fromarray(np.uint8(pixels))
        if action != 'noShow':
            im.show()

if __name__ == '__main__':
    main()

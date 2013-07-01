#!/usr/bin/python

import sys
import numpy as np
from PIL import Image
from random import random
from math import floor, fabs
import os

WHITE = np.array([255,255,255])
BLACK = np.array([0,0,0])
RED = np.array([255,0,0])
GREEN = np.array([0,255,0])
BLUE = np.array([0,0,255])



class ImageManipulation:

    def __init__(self):
        self.w,self.h = 0,0
        self.originalImage = None

    def grayscale(self, pixels):
        for x in xrange(self.h):
            for y in xrange(self.w):
                #gray = sum(pixels[x,y][list(1 * n for n in xrange(-4,-1))])/3
                #A = pixels[x,y][-1]
                (R,G,B) = pixels[x,y]
                gray = (int(R)+int(G)+int(B))/3
                pixels[x,y] = (gray, gray, gray)
        return pixels

    def threshold(self, pixels, t):
        for x in xrange(self.h):
            for y in xrange(self.w):
                if pixels[x,y][0] > t:
                    pixels[x,y] = np.array([255,255,255])
                else:
                    pixels[x,y] = np.array([0,0,0])
        return pixels

    def resize(self, oldPixels, (nW, nH)):
        self.w, self.h
        if nW <= 0 or nH <= 0:
            print 'Cant resize to smaller than zero'
            return False
        ratioW = (self.w * 1.0) / nW# ; print 'ratio w', ratioW
        ratioH = (self.h * 1.0) / nH# ; print 'ratio h', ratioH

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
        self.w = nW
        self.h = nH
        return pixels

    def addNoise(self, pixels, probab):
        # probab is a number between 0 and 1
        # indicates the probability of noise appearing in a pixel
        total = 0
        for x in xrange(self.h):
            for y in xrange(self.w):
                if random() < probab: # true add noise
                    total += 1
                    if random() < 0.5: #black noise
                        pixels[x,y] = (0,0,0)
                    else: # white noise
                        pixels[x,y] = (255,255,255)
        print 'Modified', total, 'pixels with noise'
        return pixels

    def removeNoise(self, pixels): #to be implemented
        for x in xrange(self.h):
            for y in xrange(self.w):
                1+1
                # promedio de vecinos
                # compara valor de pixel con promedio de vecinos
                # asigna promedio cuando necesario
        return pixels

    def getMaskForGradient(self, m):
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

    def getMaskForTemplateMatching(self, m):
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

    def convolution(self, pixels, mask):
        maskH, maskW = mask.shape
        newPixels = np.zeros(pixels.shape)
        halfM = maskH >> 1 # division by 2
        for x in xrange(self.h):
            for y in xrange(self.w):
                mSum = np.array([0.0, 0.0, 0.0])
                for i in xrange(maskH):
                    centerI = i - halfM
                    for j in xrange(maskW):
                        centerJ = j - halfM
                        try:
                            mSum += pixels[x+centerI,y+centerJ] * mask[i,j]
                        except IndexError: pass
                    newPixels[x,y] = mSum
                    #print mSum
        im = Image.fromarray(np.uint8(newPixels))
        im.save('convolution-output.png')
        return newPixels

    def getMagnitude(self, gradient1, gradient2):
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

    def getPixelsInBorder(self, pixels):
        pixelsBorder = list()
        for x in xrange(self.h):
            for y in xrange(self.w):
                if (pixels[x,y] == WHITE).all():
                    pixelsBorder.append((x,y))
        return pixelsBorder
                    

    def normalize(self, toNormal):
        #Val1 = (Val0 - minVal)/longitud
        print toNormal.shape
        print np.amax(toNormal)
        print np.amin(toNormal)

    def DFS(self, pixels, origin, visited, color):
        h,w,_ = pixels.shape
        q = [origin]
        visited = list()
        mass = []
        xs = []
        ys = []
        original = pixels[origin]
        n = 0
        while len(q) > 0:
            (x,y) = q.pop(0)
            if (x,y) in visited:
                continue
            current = pixels[x,y]
            if (current == original).all() or (current == color).all():
                for dx in [-1,0,1]:
                    for dy in [-1,0,1]:
                        i,j = (x+dx, y+dy)
                        if (i,j) in visited:
                            continue
                        if i >= 0 and i < h and j >= 0 and j < w:
                            print 'Inside', x,y,'-',i,j
                            content = pixels[i,j]
                            visited.append((i,j))
                            if (content == original).all():
                                pixels[i,j] = color
                                mass.append( (i,j) )
                                n += 1
                                q.append((i,j))
        return n, mass, visited

    def detectForms(self, pixels, ignorePercent = 2.0):
        h,w,_ = pixels.shape
        total =  w*h
        visited = list()
        percent = list()
        cent = list()
        count = 0
        for x in xrange(self.h):
            for y in xrange(self.w):
                if (pixels[x,y] == BLACK).all():
                    ranColor = self.getRandomColor()
                    n,mass, visited = self.DFS(pixels, (x,y), visited, ranColor)
                    p = float(n)/float(total) * 100.0
                    if p > ignorePercent:
                        cent.append( (sum(mass[0])/len(mass[0]), sum(mass[1])/len(mass[1])) )
                        percent.append( [p, ranColor] )
                        count += 1
        print 'Done'

    def getRandomColor(self):
        return np.random.randint(0, high=255, size=(1,3))

if __name__ == '__main__':
    iM = ImageManipulation()
    try:
        pathToImage = sys.argv[1]
    except IndexError:
        exit('Please specify a valid path to the file\n')
        
    im = Image.open(pathToImage).convert('RGB')
    iM.originalImage = im.copy()
    (iM.w,iM.h) = im.size
    print 'Image size:', iM.w,iM.h
    pixels = np.array(im)
    print np.array(im).shape

    action = ''
    options = '\n What do you want to do?\n' + \
        '    g : image to grayscale\n' + \
        '    t : apply a threshold\n' + \
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
            pixels = iM.grayscale(pixels)
        elif action == 't':
            t = int( raw_input('Threshold value: ') )
            pixels = iM.threshold(pixels, t)
        elif action == 'r':
            newSize = str(raw_input('New size (e.g. width height): '))
            newSize = newSize.split(' ', 1)
            try:
                newW = int(newSize[0])
                newH = int(newSize[1])
                pixels = iM.resize(pixels, (newW,newH))
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
                pixels = iM.addNoise(pixels, 0.15)
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
                mask1,mask2 = iM.getMaskForGradient(desiredMask)
                gradientX = iM.convolution(pixels, mask1) # mask in X
                gradientY = iM.convolution(pixels, mask2) # mask in y

                pixels = iM.getMagnitude(gradientX,gradientY)

                pixelsInBorder = iM.getPixelsInBorder(pixels)
                print 'Number of pixels in border:', len(pixelsInBorder)
                
                action = 'noShow'
            except TypeError as e:
                print 'Please specify only an option from the list', e
                action = 'noShow'
        elif action == 'df':
            iM.detectForms(pixels, ignorePercent = 2.0)
        elif action == 's':
            im.save('output.png')
            action = 'noShow'
        elif action == 'q':
            os.system('cls' if os.name == 'nt' else 'clear')
            exit()
        im = Image.fromarray(np.uint8(pixels))
        if action != 'noShow':
            im.show()


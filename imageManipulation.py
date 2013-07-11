#!/usr/bin/python

import sys
import numpy as np
from PIL import Image, ImageDraw
from random import random
from math import floor, ceil, fabs, degrees, cos, sin, atan2, sqrt,radians
from filters import *
import os

WHITE = np.array([255,255,255])
BLACK = np.array([0,0,0])
GRAY28 = np.array([71,71,71])
RED = np.array([255,0,0])
GREEN = np.array([0,255,0])
BLUE = np.array([0,0,255])
ORANGE = np.array([255,165,0])
YELLOW = np.array([255,255,0])

class ImageManipulation:

    def __init__(self):
        self.w,self.h = 0,0
        self.originalImage = None

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
        h,w,_ = pixels.shape
        total = 0
        for x in xrange(h):
            for y in xrange(w):
                if random() < probab: # true add noise
                    total += 1
                    if random() < 0.5: #black noise
                        pixels[x,y] = (0,0,0)
                    else: # white noise
                        pixels[x,y] = (255,255,255)
        print 'Modified', total, 'pixels with noise'
        return pixels

    def removeNoise(self, pixels): #to be implemented
        h,w,_ = pixels.shape
        for x in xrange(h):
            for y in xrange(w):
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
        h,w,_ = pixels.shape
        newPixels = np.zeros(pixels.shape)
        halfM = maskH >> 1 # division by 2
        for x in xrange(h):
            for y in xrange(w):
                mSum = np.array([0.0, 0.0, 0.0])
                for i in xrange(maskH):
                    centerI = i - halfM
                    for j in xrange(maskW):
                        centerJ = j - halfM
                        try:
                            mSum += pixels[x+centerI,y+centerJ] * mask[i,j]
                        except IndexError: pass
                    newPixels[x,y] = mSum
                    #print pixels[x,y], mSum
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
                #newPixels[x,y] = (np.fabs(gradient1[x,y]) + np.fabs(gradient2[x,y])).astype(int)
        #np.fabs(gradient[x,y][list(n for n in xrange(4))]).astype(int)
        im = Image.fromarray(np.uint8(newPixels))
        im.save('gradient-magnitude.png')
        return newPixels

    def getPixelsInBorder(self, pixels):
        h,w,_ = pixels.shape
        pixelsBorder = list()
        for x in xrange(h):
            for y in xrange(w):
                if (pixels[x,y] == WHITE).all():
                    pixelsBorder.append((x,y))
        return pixelsBorder

    def normalize(self,gradient, toSave='normalization-result.png'):
        h,w,_ = gradient.shape
        biggest = np.amax(gradient)
        smallest = np.amin(gradient)
        lenght = ( (biggest+fabs(smallest) ) - ( smallest+fabs(smallest) ) )
        for x in range(h):
            for y in range(w):
                newVal = (gradient[x,y] - np.array([smallest, smallest, smallest]) ) / np.array([lenght,lenght,lenght]) * np.array([255,255,255])
                #print 'Pixel',x,',',y, 'had value', gradient[x,y], 'now has value', newVal
                gradient[x,y] = newVal
        imag = Image.fromarray(np.uint8(gradient))
        imag.save(toSave)
        return gradient

    def DFS(self, pixels, origin, visited, color):
        h,w,_ = pixels.shape
        queue = list()
        xCoord = list()
        yCoord = list()
        queue.append(origin)
        original = pixels[origin]
        n = 0
        while len(queue) > 0:
            (x,y) = queue.pop(0)
            actual = pixels[x,y]
            if (actual == original).all() or (actual == color).all():
                for dx in [-1,0,1]:
                    for dy in [-1,0,1]:
                        i,j = (x+dx,y+dy)
                        if (i,j) in visited:
                            continue
                        if i>=0 and i<h and j>=0 and j<w:
#                        if i>=0 and i<self.h and j>=0 and j<self.w:
                            content = pixels[i,j]
                            visited.add((i,j))
                            if (content == original).all():
                                pixels[i,j] = color
                                xCoord.append(i)
                                yCoord.append(j)
                                n+=1
                                queue.append((i,j))
        return n,(xCoord,yCoord)

    def detectBorder(self,pixels):

        h,w,_ = pixels.shape

        mask1,mask2 = self.getMaskForGradient('p')
        mask3,mask4 = self.getMaskForGradient('s')

        g1 = self.convolution(pixels, mask1)
        g2 = self.convolution(pixels, mask2)
        g3 = self.convolution(pixels, mask3)
        g4 = self.convolution(pixels, mask4)

        for x in xrange(h):
            for y in xrange(w):
                newVal = np.sqrt((g1[x,y]**2) + (g2[x,y]**2) + (g3[x,y]**2) + (g4[x,y]**2))
                pixels[x,y] = np.array([int(floor(sqrt(p))) for p in newVal])
        pixels = self.normalize(pixels)
        fil = Filters()
        fil.threshold(pixels, 128)
        pixelsInBorder = self.getPixelsInBorder(pixels)
        self.pixelsInBorder = pixelsInBorder
        print 'Number of pixels in border:', len(pixelsInBorder)

        return pixels

    def detectForms(self, pixels, ignorePercent = 2.0):
        h,w,_ = pixels.shape
        total =  w*h
        visited = set()
        percent = list()
        cent = list()

        objectList = list() # each element is a list of x,y coordinates

        count = 0
        for x in xrange(h):
            for y in xrange(w):
                if (pixels[x,y] == BLACK).all():
                    ranColor = self.getRandomColor()
                    n,(i,j) = self.DFS(pixels, (x,y), visited, ranColor)
                    p = float(n)/float(total) * 100.0
                    if p > ignorePercent:
                        cent.append( (sum(i)/len(i), sum(j)/len(j)) )
                        percent.append( [p, ranColor] )
                        coord = list() # list stores obj coordinates
                        for a,b in zip(i,j):
                            coord.append((a,b))
                        objectList.append(coord)
                        print 'Object', count+1, 'has color', ranColor
                        count += 1
        background = percent.index(max(percent))
        backgroundColor = percent[background][1]
        print 'An object identified as background with color', backgroundColor, 'is now', GRAY28

        for x in xrange(h):
            for y in xrange(w):
                if (pixels[x,y] == backgroundColor).all():
                    pixels[x,y] = GRAY28
        return objectList

    def getRandomColor(self):
        return np.random.randint(0, high=255, size=(1,3))

    def detectLines(self, pixels, gradX, gradY, discr=10):
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
                    #    pixels[i,j] = ORANGE
        return lines, freq

    def findMins(self, aList, compSens=15):
        minList = list()
        
        for i in xrange( len(aList) ):
            try:
                if aList[i-compSens] > aList[i] and aList[i] < aList[i+compSens]:
                    minList.append( i )
            except:
                continue
        return minList

    def saveHistograms(self,xHist,yHist):
        f = open('histogram-X.dat','w')
        for x in xrange(len(xHist)):
            f.write(str(x) + ',' + str(xHist[x]) + '\n')
        f.close()
        f = open('histogram-Y.dat','w')
        sumY = 0
        for y in xrange(len(yHist)):
            f.write( str(y) + ',' + str(yHist[y]) + '\n')
        f.close()

    def detectHoles(self, pixels, holeColor):
        h,w,_ = pixels.shape
        # get histograms
        rows = list()
        for x in xrange(h):
            columns = list()
            for y in xrange(w):
                columns.append( pixels[x,y][0] )
            rows.append( pixels[x,y][0] )


        rows = list()
        for x in xrange(h):
            acumulator = 0
            for y in xrange(w):
                acumulator += pixels[x,y][0]
            rows.append( acumulator )


        columns = list()
        for y in xrange(w):
            acumulator = 0
            for x in xrange(h):
                acumulator += pixels[x,y][0]
            columns.append( acumulator )


        print 'Len(rows)', len(rows)
        print 'Len(columns)', len(columns)

        self.saveHistograms(columns,rows)
        
        # find mins, mins are for black holes
        # maxs when white
        rowMin= iM.findMins(rows,5) #7
        columnMin= iM.findMins(columns,6) #6

        print 'Mins in rows', rowMin
        print 'Mins in columns', columnMin

        # draw lines
        im = Image.fromarray(np.uint8(pixels))
        draw = ImageDraw.Draw(im)

        for m in rowMin:
            draw.line( ((0,m),(w,m)), fill=(0,255,0))

        for m in columnMin:
            draw.line( ((m,0),(m,h)), fill=(0,0,255))

        im.save('hole-detection.png')
        return pixels


if __name__ == '__main__':
    iM = ImageManipulation()
    fil = Filters()

    try:
        pathToImage = sys.argv[1]
    except IndexError:
        exit('Please specify a valid path to the file\n')
        
    im = Image.open(pathToImage).convert('RGB')
    iM.originalImage = im.copy()
    (iM.w,iM.h) = im.size
    print 'Image size: (w h)', iM.w,iM.h
    pixels = np.array(im)

    action = ''
    options = '\n What do you want to do?\n' + \
        '    g  : image to grayscale\n' + \
        '    p  : polarize an image\n' + \
        '    br : apply bright\n' + \
        '    to : apply a tone\n' + \
        '    th : apply a threshold\n' + \
        '    r  : resize image\n' + \
        '    n  : add noise\n' + \
        '    m  : apply a mask\n' + \
        '    db : detect borders\n' + \
        '    df : detect forms\n' + \
        '    dl : detect lines\n' + \
        '    dc : detect circles\n' + \
        '    dh : detect holes\n' + \
        '    s  : save to file\n' + \
        '    q  : quit the program\n'

    while action != 'q':
        action = raw_input(options)
        action = str(action)
        #os.system('cls' if os.name == 'nt' else 'clear')
        if action == 'g':
            pixels = fil.grayscale(pixels)
        elif action == 'p':
            pixels = fil.polarize(pixels)
        elif action == 'br':
            pixels = fil.brighten(pixels, beta=50)
        elif action == 'to':
            pixels = fil.tone(pixels)
        elif action == 'th':
            t = int( raw_input('Threshold value: ') )
            pixels = fil.threshold(pixels, t)
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

                action = 'noShow'
            except TypeError as e:
                print 'Please specify only an option from the list\nError:', e
                action = 'noShow'
        elif action == 'db':
            pixels = iM.detectBorder(pixels)
        elif action == 'df':
            iM.detectForms(pixels, ignorePercent = 2.0)
        elif action == 'dl':
            mask1,mask2 = iM.getMaskForGradient('p')
            
            print 'Getting gradient X'; gradientX = iM.convolution(pixels, mask1) # mask in X
            print 'Getting gradient Y'; gradientY = iM.convolution(pixels, mask2) # mask in y

            lines,freq = iM.detectLines(pixels, gradientX, gradientY)
        elif action == 'dc':
            try:
                radio = int(raw_input('Radio: '))
            except ValueError:
                print 'Please enter a number'
                continue
            mask1,mask2 = iM.getMaskForGradient('p')
            print 'Getting gradient X'
            gradX = iM.convolution(pixels, mask1)
            print 'Getting gradient Y'
            gradY = iM.convolution(pixels, mask2)
            iM.detectCircles(pixels, gradX, gradY, radio)
        elif action == 'dh':
            pixels = iM.detectHoles(pixels, 0)
        elif action == 's':
            im.save('output.png')
            action = 'noShow'
        elif action == 'q':
            os.system('cls' if os.name == 'nt' else 'clear')
            exit()
        im = Image.fromarray(np.uint8(pixels))
        if action != 'noShow':
            im.show()


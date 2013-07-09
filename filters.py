#!/usr/bin/python

import numpy as np

class Filters:
    
    def __init__(self):
        self.w, self.h = 0,0

    def grayscale(self, pixels):
        h, w,_ = pixels.shape
        for x in xrange(h):
            for y in xrange(w):
                (R,G,B) = pixels[x,y]
                gray = (int(R)+int(G)+int(B))/3
                pixels[x,y] = np.array([gray, gray, gray])
        return pixels
    
    def polarize(self, pixels):
        h, w,_ = pixels.shape
        for x in xrange(h):
            for y in xrange(w):
                (R,G,B) = pixels[x,y]
                pixels[x,y] = np.array([ 255-R, 255-G, 255-B ])
        return pixels

    def brighten(self, pixels, beta=None):
        h, w,_ = pixels.shape
        if beta == None:
            beta = 30
        for x in xrange(h):
            for y in xrange(w):
                R,G,B = pixels[x,y]
                R+=beta
                G+=beta
                B+=beta
                if (R>255):R=255
                if (G>255):G=255
                if (B>255):B=255
                pixels[x,y] = np.array([R,G,B])
        print np.mean(pixels)
        return pixels

    def tone(self, pixels):
        h, w,_ = pixels.shape
        t = ' '
        while t not in 'RGB':
            t = str(raw_input('Which tone [R,G,B]: '))
        print ord(t)
        for x in xrange(h):
            for y in xrange(w):
                (R,G,B) = pixels[x,y]
                if t == 'R':
                    newTone = np.array([ R,0,0 ])
                elif t == 'G':
                    newTone = np.array([ 0,G,0 ])
                elif t == 'B':
                    newTone = np.array([ 0,0,B ])
                pixels[x,y] = newTone
        return pixels

    def threshold(self, pixels, t):
        h, w,_ = pixels.shape
        for x in xrange(h):
            for y in xrange(w):
                if pixels[x,y][0] > t:
                    pixels[x,y] = np.array([255,255,255])
                else:
                    pixels[x,y] = np.array([0,0,0])
        return pixels

    def resize(self, oldPixels, (nW, nH)):
        if nW <= 0 or nH <= 0:
            print 'Cant resize to smaller than zero'
            return False
        h, w,_ = pixels.shape
        ratioW = (w * 1.0) / nW# ; print 'ratio w', ratioW
        ratioH = (h * 1.0) / nH# ; print 'ratio h', ratioH
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

if __name__ == '__main__':
    fil = Filters()
    print 'Hi'

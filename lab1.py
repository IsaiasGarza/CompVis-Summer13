    def brighten(self, pixels, beta=None):
        if beta == None:
            beta = 30
        for x in xrange(self.h):
            for y in xrange(self.w):
                R,G,B = pixels[x,y]
                R+=beta
                G+=beta
                B+=beta
                if (R>255):R=255
                if (G>255):G=255
                if (B>255):B=255
                pixels[x,y] = np.array([R,G,B])
                return pixels

    def polarize(self, pixels):
        for x in xrange(self.h):
            for y in xrange(self.w):
                (R,G,B) = pixels[x,y]
                pixels[x,y] = np.array([ 255-R, 255-G, 255-B ])
        return pixels

    def tone(self, pixels):
        t = ' '
        while t not in 'RGB':
            t = str(raw_input('Which tone [R,G,B]: '))
        for x in xrange(self.h):
            for y in xrange(self.w):
                (R,G,B) = pixels[x,y]
                if t == 'R':
                    newTone = np.array([ R,0,0 ])
                elif t == 'G':
                    newTone = np.array([ 0,G,0 ])
                elif t == 'B':
                    newTone = np.array([ 0,0,B ])
                pixels[x,y] = newTone
        return pixels

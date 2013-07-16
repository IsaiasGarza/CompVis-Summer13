from Tkinter import *
from PIL import Image, ImageTk, ImageDraw
from math import floor, atan, fabs, pi, cos, sin, ceil, sqrt, degrees, atan2
from random import randint
import random
import numpy as np

def buildGUI():
    global frame
    root = Tk()
    root.title('Circles')
    frame = Frame()
    frame.pack(padx=5,pady=5)
    loadImage(getOriginal(originalPath))
    detCir = Button(text='Detect circles', command = detect).pack(in_=frame, side=TOP)
    root.mainloop()

def loadImage(image):
    global label
    photo = ImageTk.PhotoImage(image)
    label = Label(image=photo)
    label.imagen = photo
    label.pack()

def grayscale(path_original):
    img = Image.open(originalPath).convert("RGB")
    pixels = img.load()
    x, y = img.size
    newImage = Image.new("RGB", (x, y))
    colors = []
    for a in xrange(x):
        for b in xrange(y):
            pixel_color = pixels[a, b]
            gray = sum(pixel_color)/3
            newColor = (gray, gray, gray)
            colors.append(newColor)
            newImage.putpixel((a, b), newColor)
    return newImage

def convolution(f, h):
    pixels = f.load()
    x, y = f.size
    newP = np.zeros(x*y).reshape((x, y))
    F = Image.new("RGB", (x, y))
    i = len(h[0])
    j = len(h[0])
    for a in xrange(x):
        for b in xrange(y):
            total = 0
            for c in xrange(i):
                c1 = c - i/2
                for d in xrange(j):
                    d1 = d - j/2
                    try:
                        total = total + (pixels[a+c1, b+d1][0])*(h[c][d])
                    except:
                        pass
            newP[a, b] = (total)
            total = int(floor(total))
            prom = (total, total, total)
            F.putpixel((a,b),prom)
    return F, newP

def blur(img):
    pixels = img.load()
    x, y = img.size
    newImg = Image.new("RGB", (x, y))
    colors = []
    for a in xrange(x):
        for b in xrange(y):
            sumR = 0
            sumG = 0
            sumB = 0
            count = 0
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    try:
                        (R,G,B) = pixels[a+dx,b+dy]
                        sumR += R
                        sumG += G
                        sumB += B
                        count += 1
                    except: pass
            newColor = (sumR/count, sumG/count, sumB/count)
            colors.append(newColor)
            newImg.putpixel((a, b), newColor)    
    return newImg

def normalize(img):
    pixels = img.load()
    x, y = img.size
    minim = 0
    maxim = 0
    newImg = Image.new("RGB", (x, y))
    
    for a in xrange(x):
        for b in xrange(y):
            currentColor = pixels[a, b]
            prevMaxim = maxim
            newMaxim = currentColor[0]
            prevMinim = minim
            newMinim = currentColor[0]
            if (newMaxim >= prevMaxim):
                maxim = newMaxim
            elif (newMinim <= prevMinim):
                minim = newMinim
            elif (minim == 0 and maxim == 255):
                break

    for a in xrange(x):
        for b in xrange(y):
            currentColor = pixels[a, b]
            newPix = ( float(currentColor[0]) - float(minim) )*( float(255) / (float(maxim) - float(minim)) )
            newPix = int(floor(newPix))
            newVal = (newPix, newPix, newPix)
            newImg.putpixel((a, b), newVal)
    return newImg

def threshold(img, threshLevel):
    pixels = img.load()
    x, y = img.size
    newImg = Image.new("RGB", (x, y))
    for a in xrange(x):
        for b in xrange(y):
            currentColor = pixels[a, b]
            valueColor = float(currentColor[0])
            color_nor = valueColor/255.0
            if(color_nor>=threshLevel):
                repl = 255
            else:
                repl = 0
            newCol = (repl, repl, repl)
            newImg.putpixel((a, b), newCol)
    return newImg

def getOriginal(originalPath):
    return Image.open(originalPath)

def BFS(img, origin, color):
    pixels = img.load()
    height, width = img.size
    
    row, column = origin
    original = pixels[row, column]
    
    queue = []
    queue.append((row, column))
    mass = []
    c = []
    
    while len(queue) > 0:
        (row, column) = queue.pop(0)
        actual = pixels[row, column]
        if actual == original or actual == color:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    coord = (row + dy, column + dx)
                    if coord[0] >= 0 and coord[0] < height and coord[1] >= 0 and coord[1] < width:
                        contenido = pixels[coord[0], coord[1]]
                        if contenido == original:
                            pixels[coord[0], coord[1]] = color
                            img.putpixel((coord[0], coord[1]), color)
                            queue.append(coord)
                            mass.append((coord[0], coord[1]))
                            c.append((coord[0], coord[1]))
    return img, mass, c

def runBFS(img_BFS):
    global frame
    pixels = img_BFS.load()
    x, y = img_BFS.size
    colors = []
    circls = []
    for a in xrange(x):
        for b in xrange(y):
            if pixels[a, b] == (255, 255, 255):
                color = (random.randint(0,255), random.randint(0,255), random.randint(0, 255))
                img_BFS, mass, c = BFS(img_BFS.convert("RGB"), (a, b), color)
                circls.append(c)
                x_total = 0
                y_total = 0
                for i in xrange(len(mass)):
                    x_total = x_total + mass[i][0]
                    y_total = y_total + mass[i][1]
                x_centr = x_total/len(mass)
                y_centr = y_total/len(mass)
                colors.append([color, 0, (x_centr, y_centr)])
                pixels = img_BFS.load()
                mass = []
    #suma la cantidad de colores diferentes en la img
    pixels = img_BFS.load()
    for a in xrange(x):
        for b in xrange(y):
            for n in xrange(len(colors)):
                if colors[n][0] == pixels[a,b]:
                    colors[n][1] = colors[n][1] + 1
    #print colors
    total = 0
    for i in xrange(len(colors)):
        total = total + colors[i][1]
    
    y = frame.winfo_height()
    
    #Obtenemos porcentajes
    prom = []
    for i in xrange(len(colors)):
        avg = float(colors[i][1])/float(total)*100.0
        if avg > 3.0:
            #print "Porcentajes: "
            #print "Figura " + str(i+1) + ": " + str(avg)
            prom.append((i, avg, colors[i][0]))

    maximim = 0.0
    for i in xrange(len(prom)):
        if maximim < prom[i][1]:
            maximim = prom[i][1]
            fig = prom[i][0]
            color_maxim = prom[i][2]

    img_BFS = repaintBG(img_BFS, color_maxim)
    return img_BFS, colors, circls #returns image, [(color),numberOfPixelsColor, (centre)]

def repaintBG(img_BFS, color_maxim):
    pixels = img_BFS.load()
    x, y = img_BFS.size
    for a in xrange(x):
        for b in xrange(y):
            if pixels[a, b] == color_maxim:
                color = (100,100,100)
                img_BFS, mass, c = BFS(img_BFS.convert("RGB"), (a, b), color)
                return img_BFS

def detect():
    label.destroy()
    
    maxim = 0
    total = 0.0
        
    img = grayscale(originalPath)
    img.save("gris.png")
    
    h_lap = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    img_prom, points = convolution(img, np.multiply(1.0/1.0,h_lap))
    img_prom = threshold(img_prom, 0.5)
    
    img_nor = normalize(img_prom)
    
    threshLevel = 0.1
    img_bin = threshold(img_nor.convert("RGB"), threshLevel)
    img_prom = blur(img_bin.convert("RGB"))
    threshLevel = 0.08
    img_bin2 = threshold(img_prom.convert("RGB"), threshLevel)
    img_prom = blur(img_bin2.convert("RGB"))
    img_prom = blur(img_prom.convert("RGB"))
    
    threshLevel = 0.5
    img_BFS = threshold(img_prom.convert("RGB"), threshLevel)

    img_BFS, colors, circls = runBFS(img_BFS)
    #returns image, [(color),numberOfPixelsColor, (centre)], listOfPixelsInCircles

    x, y = img_BFS.size

    pixels = img_BFS.load()
    contour = []

    h_Y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    h_X = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    horizontal, points_GX = convolution(img, np.multiply(1.0/1.0,h_X))
    pixels_GX = horizontal.load()
    vertical, points_GY = convolution(img, np.multiply(1.0/1.0,h_Y))
    pixels_GY = vertical.load()
    
    drw = ImageDraw.Draw(img_BFS)
    x, y = img_BFS.size
    points = np.zeros(x*y).reshape((x, y))
    centr = [] #centr son los centros advinados
    for cir in circls:
        x, y = img_BFS.size
        points = np.zeros(x*y).reshape((x, y))
        for i in xrange(850): #bigger number more precise
            tam = len(cir)
            pnt1 = cir[randint(0,tam-1)]
            pnt2 = cir[randint(0,tam-1)]
            
            x_1 = pnt1[0]
            y_1 = pnt1[1]
        
            x_2 = pnt2[0]
            y_2 = pnt2[1]
        
            gx_1 = points_GX[x_1, y_1]
            gy_1 = points_GY[x_1, y_1]
        
            gx_2 = points_GX[x_2, y_2]
            gy_2 = points_GY[x_2, y_2]
        
            gx_1 = - float(gx_1)
            gx_2 = - float(gx_2)
        
            x_22 = x_2
            y_22 = y_2
            x_11 = x_1
            y_11 = y_1
        
            if abs(gx_1) + abs(gy_1) <= 0:
                theta = None
            else:
                theta = atan2(gy_1, gx_1)

            l = 50
                
            if theta is not None:
                theta = theta-(pi/2)
                x_1 = x_11 - l * cos(theta)
                y_1 = y_11 - l * sin(theta)
                x_2 = x_11 + l * cos(theta)
                y_2 = y_11 + l * sin(theta)
            
        
            if abs(gx_2) + abs(gy_2) <= 0:
                theta = None
            else:
                theta = atan2(gy_2, gx_2)
        
    
            if theta is not None:
                theta = theta-(pi/2)
                x_3 = x_22 - l * cos(theta)
                y_3 = y_22 - l * sin(theta)
                x_4 = x_22 + l * cos(theta)
                y_4 = y_22 + l * sin(theta)
                
            y_medio = (y_11+y_22) / 2
            x_medio = (x_11+x_22) / 2
                
            pixels = img_BFS.load()
            
            try:
                Px = ((x_1*y_2-y_1*x_2)*(x_3-x_4)-(x_1-x_2)*(x_3*y_4-y_3*x_4))/((x_1-x_2)*(y_3-y_4)-(y_1-y_2)*(x_3-x_4))
                Py = ((x_1*y_2-y_1*x_2)*(y_3-y_4)-(y_1-y_2)*(x_3*y_4-y_3*x_4))/((x_1-x_2)*(y_3-y_4)-(y_1-y_2)*(x_3-x_4))

                Dx = Px - x_medio
                Dy = Py - y_medio
                m = Dy/Dx 
                x0 = x_medio
                y0 = y_medio
                while True:
                    x = x0+1
                    y = m*(x-x0)+y0
                    if pixels[x,y] == (0, 0, 0):
                        points[x, y] = points[x, y] + 1
                        x0 = x
                        y0 = y
                    else:
                        break
            except:
                pass
        maxim = np.max(points)
        index = np.where(points==maxim)
        try:
            mayor_x = sum(index[0])/len(index[0])
            mayor_y = sum(index[1])/len(index[0])
            #drw.ellipse((mayor_x-2, mayor_y-2, mayor_x+2, mayor_y+2), fill="blue")
            centr.append((mayor_x, mayor_y))
        except:
            mayor_x = index[0]
            mayor_y = index[1]
            #drw.ellipse((mayor_x-2, mayor_y-2, mayor_x+2, mayor_y+2), fill="blue")
            centr.append((mayor_x, mayor_y))

    r_x = []
    r_y = []
    pixels_img = img_BFS.load()
    for i in xrange(len(centr)):
        x0 = int(centr[i][0])
        y0 = int(centr[i][1])
        while True:
            y0 = y0 + 1
            if pixels_img[x0, y0] != (0, 0, 0):
                r_y.append((x0, y0))
                break

    for i in xrange(len(centr)):
        x0 = int(centr[i][0])
        y0 = int(centr[i][1])
        while True:
            x0 = x0 + 1
            if pixels_img[x0, y0] != (0, 0, 0):
                r_x.append((x0, y0))
                break
    Radios = []
    noDraw = [] # stores indexes that are not to be drawn
    x, y = img_BFS.size
    for i in xrange(len(centr)):
        x0 = int(centr[i][0])
        y0 = int(centr[i][1])
        x1 = int(r_x[i][0])
        y1 = int(r_x[i][1])
        x2 = int(r_y[i][0])
        y2 = int(r_y[i][1])
        Rx = sqrt((x1-x0)**2+(y1-y0)**2)
        Ry = sqrt((x2-x0)**2+(y2-y0)**2)
        porcentaje = float(Rx * 100)/float(x)

        print "\nID: %s  Posible radio: %s " % (i+1, Rx)
        if Rx != Ry and (Rx-Ry > 10):
            print 'No Circulo'
            noDraw.append(i)
        else:
            print 'Circulo'


        color = (255, 255, 0)
        a = 0
        while a < 2*pi:
            x4, y4 = x0 + Rx * sin(a), y0 + Ry * cos(a)
            a = a + 0.01
            drw.ellipse((x4-2, y4-2, x4+2, y4+2), fill=color)

    pixels = img_BFS.load()
    width, height = img_BFS.size
    yellows = []
    for x in xrange(width): #adds yellows
        for y in xrange(height):
            pixel_color = pixels[x, y]
            if pixel_color == color:
                yellows.append((x,y))
    print '\n', len(yellows)

    total = [0]*len(circls)
    for y in yellows:
        for i in xrange(len(circls)):
            if y in circls[i]:
                total[i] += 1
    print total

    nPixelsCircls = [0]*len(circls)
    for i in xrange(len(circls)):
        nPixelsCircls[i] = colors[i][1]
    print nPixelsCircls

    print 'Comparando circulos calculados contra circulos en imagen'
    for i in xrange(len(circls)):
        print 'Medida de similitud para circulo', i+1, ':',(total[i] * 1.0) / nPixelsCircls[i]


    for i in xrange(len(centr)): #draws centers
        color = (random.randint(0,255), random.randint(0,255), random.randint(0, 255))
        drw.ellipse((centr[i][0]-2, centr[i][1]-2, centr[i][0]+2, centr[i][1]+2), fill="blue")

    loadImage(img_BFS)

    global frame
    y = frame.winfo_height()
    for i in xrange(len(centr)):
        if i in noDraw:
            continue
        label_fig = Label(text = str(i+1))
        label_fig.place(x = centr[i][0]+10,  y = centr[i][1] + y+10)
    img_BFS.save("fin.png")
    return

if __name__ == '__main__':
    originalPath = "circles.png"
    buildGUI()

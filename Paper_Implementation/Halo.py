# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 12:43:27 2018
@author: K.S.LOHITH
"""
import cv2
#import sqlite3
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cam=cv2.VideoCapture('video1.mp4');
green = []
red = []
blue = []

#insertOrUpdate(id,name)
sampleNum=0;
while(cam.isOpened()):
 ret,img=cam.read();
 gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 kernel = np.ones((5,5),np.float32)/25
 dst = cv2.filter2D(img,-1,kernel)
 faces=faceDetect.detectMultiScale(dst,1.3,5);
 for(x,y,w,h) in faces:
     roi_color = img[y:y+h, x:x+w]
     cv2.imshow("face",roi_color)
     eyes = eyes_cascade.detectMultiScale(roi_color, 1.1, 3)
     for (ex, ey, ew, eh) in eyes:
      roi_eye = roi_color[ey:ey+eh, ex-ew:ex+ew]
      cv2.imshow("eye",roi_eye)
      #cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
      hx=ex
      hy=ey-30
      hw=ex+ew
      hh=ey+eh-30
        #cv2.rectangle(roi_color,(ex,ey-30),(ex+ew,ey+eh-30),(0,0,255),2)
     try:
        #cv2.rectangle(roi_color,(hx-15,hy-20),((2*hw)+45,hh-30),(0,0,255),2)
        roi = roi_color[hy:hh-50,ex-ew:ex+ew]
        b,g,r = cv2.split(roi)
        bl = np.mean(b)
        re = np.mean(r)
        gr = np.mean(g)
        green.append(gr)
        red.append(re)
        blue.append(bl)
        print(b + " " + g + " " + r)
        cv2.imshow("roi",roi)
     except UnboundLocalError:
        pass
     except:
        pass
     sampleNum=sampleNum+1;
     cv2.imwrite("dataSet/User."+str('1')+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
     cv2.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),2)
 cv2.imshow("Face",dst);
 cv2.waitKey(25);
 if(sampleNum>100):
     break
cam.release()
cv2.destroyAllWindows()
for i in range(len(blue)-1):
    blue[i] = abs(blue[i+1]-blue[i])
blue = blue[:99]
for i in range(len(green)-1):
    green[i] = abs(green[i+1]-green[i])
green = green[:99]    
for i in range(len(red)-1):
    red[i] = abs(red[i+1]-red[i])
red = red[:99]    

from scipy import signal
from scipy.fftpack import fft
import numpy as np
from scipy.fftpack import fftfreq

y1 = fft(red)
y2 = fft(green)
y3 = fft(blue)

dred= signal.detrend(red)
dgreen = signal.detrend(green)
dblue = signal.detrend(blue)

yd1 = fft(dred)
yd2 = fft(dgreen)
yd3 = fft(dblue)

N = 99
# sample spacing
T = 1.0 / 100.0
x = np.linspace(0.0, N*T, N)

y1 = np.array(red)
y2 = np.array(green)
y3 = np.array(blue)

yf1 = fft(y1)
yf2 = fft(y2)
yf3 = fft(y3)

realyf1 = yf1.real
realyf2 = yf2.real
realyf3 = yf3.real

yg1 = np.abs(yd1[:N//2])
yg2 = np.abs(yd2[:N//2])
yg3 = np.abs(yd3[:N//2])
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
#xf = xf[2:]
xint = list(xf)
#for i in xint:
#    i = math.ceil(i)
xf1 = xf*10
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
ax.plot(xf*10, yg3)
ax.plot(xf*10, yg2)
ax.plot(xf*10, yg1)
plt.show()

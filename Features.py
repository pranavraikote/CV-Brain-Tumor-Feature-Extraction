# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:37:51 2020

@author: Pranav
"""
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def image_process(path):
    
    print(path)
        
    frame1 = cv2.imread(path)
    
    frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(frame, (15, 15), 0)
    
    ret, thresh = cv2.threshold(blur, 175, 230, 0)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    new_frame = cv2.drawContours(frame, contours, -1, (0,255,0), 3)

    plt.imshow(new_frame)
    
    data_export = []
    
       
    if len(contours)>0:
        
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        
        
        #print('Contours - {}'.format(len(contours)))
            
        x, y, w, h = cv2.boundingRect(cnt)
        r = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #plt.imshow(r)
        
        area = cv2.contourArea(cnt)
        data_export.append(area)
        print('Area - {}'.format(area))
        
        (x1, y1), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x1), int(y1))
        radius = int(radius)
        c = cv2.circle(frame, center, radius, (255, 0, 0), 2)
        #plt.imshow(c)
        
        perimeter = cv2.arcLength(cnt, True)
        data_export.append(perimeter)
        print('Perimeter - {}'.format(perimeter))
        
        #aspect_ratio = float(w)/h
        #print('Aspect Ratio - {}'.format(aspect_ratio))
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        data_export.append(hull_area)
        print('Convex Area - {}'.format(hull_area))
        
        solidity = float(area)/hull_area
        data_export.append(solidity)
        print('Solidity - {}'.format(solidity))
        
        equi_diameter = np.sqrt(4*area/np.pi)
        data_export.append(equi_diameter)
        print('Equivalent Diameter - {}'.format(equi_diameter))
        
        try:
            (x2,y2), (MA,ma), angle = cv2.fitEllipse(cnt)
        except:
            MA = 0
            ma = 0
        data_export.append(MA)
        data_export.append(ma)
        print('Major Axis - {}'.format(MA))
        print('Minor Axis - {}'.format(ma))
        
        #eccentricity = np.sqrt(1-(ma/MA)**2)
        eccentricity = (1-ma**2/MA**2)**(0.5)
        data_export.append(eccentricity)
        print('Eccentricity - {}'.format(eccentricity))
                           
        return data_export
        
    else:
                
        print('No Tumor Found')
        for i in range(8):
            data_export.append(0)
        return data_export




import glob
l = glob.glob('sample_dataset/*.jpg')
neg_list = l[0:92]
pos_list = l[93:]

data = pd.DataFrame(columns=['Area','Perimeter','Convex Area','Solidity','Equivalent Diameter','Major Axis','Minor Axis','Eccentricity','Class'])

for i in pos_list:
    d = image_process(i)
    d.append(1)
    new_data = pd.DataFrame([d], columns=['Area','Perimeter','Convex Area','Solidity','Equivalent Diameter','Major Axis','Minor Axis','Eccentricity','Class'])
    data = data.append(new_data, ignore_index=True)
    
for i in neg_list:
    d = image_process(i)
    d.append(0)
    new_data = pd.DataFrame([d], columns=['Area','Perimeter','Convex Area','Solidity','Equivalent Diameter','Major Axis','Minor Axis','Eccentricity','Class'])
    data = data.append(new_data, ignore_index=True)
    
data.to_csv('Dataset.csv')













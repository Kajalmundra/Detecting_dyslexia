import cv2
import numpy as np
import matplotlib.pyplot as plt

def pressure(image):
    if len(image.shape)==3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray=image.copy()
            
    h, w = gray.shape[:]
    
    #cv2.imshow('Grayscale', image)

    median = cv2.medianBlur(gray, 3)
    #cv2.imshow('Median Filter', median)

    total_intensity = 0
    pixel_count = 0
    for x in range(h):
        for y in range(w):
            if(median[x][y] < 150):
                total_intensity += median[x][y]
                pixel_count += 1

    average_intensity = round((total_intensity / pixel_count),2)
    percentage = round(((average_intensity * 100) / 255),2)
    return average_intensity, percentage

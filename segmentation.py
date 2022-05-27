import cv2
import numpy as np

def Segmentation(Gray_img):
       Txt_angle1=[]
       # Resize Image
       Resized_Image = cv2.resize(Gray_img,(200,400))

       #cv2.imshow('Input Image',Resized_Image)
       #cv2.waitKey(0)

       # Performing OTSU threshold 
       ret, thresh1 = cv2.threshold(Resized_Image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
       #cv2.imshow('',thresh1)
       #cv2.waitKey(0)
       nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh1, connectivity=8)
       #print(stats)
       kernel = np.ones((1, 2), np.uint8)
       kerne2 = np.ones((2,6), np.uint8)
       kerne3 = np.ones((2, 4), np.uint8)

       thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
       closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kerne2)
       opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kerne3)
       
       #cv2.imshow('',opening)
       #cv2.waitKey(0)
       # Finding Connected component
       nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=8)
       sizes = stats[1:, -1]
       nb_components = nb_components - 1
       min_size = 20
       max_size=12000
           # Removing Small Pixel value
       img2 = np.zeros((Resized_Image.shape))
       for i in range(0, nb_components):
           if sizes[i] >= min_size and sizes[i] < max_size:
               img2[output == i + 1] = 255

       img2 = cv2.dilate(img2, (np.ones((2, 2), np.uint8)), iterations=4)

       #cv2.imshow('',img2)
       #cv2.waitKey(0)
       
      ## Hog transform
       rho = 1
       theta = np.pi/180
       threshold = 60
       min_line_length = 10
       max_line_gap = 250
       #cv2.imshow('',Gray_img)
       
       line_image = np.copy(Gray_img)

       lines = cv2.HoughLinesP(np.uint8(img2), rho, theta, threshold, np.array([]),
                               min_line_length, max_line_gap)
       dist1=[]
       line_dim=[]
       clob_angle1=[]
       if len(lines) >0:
           # Draw lines on the image
           for line in lines:
               for x1,y1,x2,y2 in line:
                   cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),2)
                   dist = np.linalg.norm((np.array(x1,y1),np.array(x2,y2)))
                   line_dim.append([x1,y1,x2,y2])
                   dist1.append(dist)
                   Txt_angle1.append(round((np.arctan(((y2-y1)/(x2-x1+0.01))))*100,2))
           # Show result
           idx = dist1.index(max(dist1))
           txt_angle=Txt_angle1[idx]

       return img2,txt_angle

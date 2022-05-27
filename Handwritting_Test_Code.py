import cv2
import numpy as np
from scipy import ndimage as nd
from scipy import ndimage
from matplotlib import pyplot as plt
import joblib
import pressure
import zones
import segmentation
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import gabor
from skimage.filters import gabor_kernel
from tkinter import filedialog


from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = nd.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats

def GLCM_Feature(cropped):
    # GLCM Feature extraction
    glcm = greycomatrix(cropped, [1, 2], [0, np.pi/2], levels=256, normed=True, symmetric=True)
    dissim = (greycoprops(glcm, 'dissimilarity'))
    dissim=np.reshape(dissim, dissim.size)
    correl = (greycoprops(glcm, 'correlation'))
    correl=np.reshape(correl,correl.size)
    energy = (greycoprops(glcm, 'energy'))
    energy=np.reshape(energy,energy.size)
    contrast= (greycoprops(glcm, 'contrast'))
    contrast= np.reshape(contrast,contrast.size)
    homogen= (greycoprops(glcm, 'homogeneity'))
    homogen = np.reshape(homogen,homogen.size)
    asm =(greycoprops(glcm, 'ASM'))
    asm = np.reshape(asm,asm.size)
    glcm = glcm.flatten()
    Mn=sum(glcm)
    Glcm_feature = np.concatenate((dissim,correl,energy,contrast,homogen,asm,Mn),axis=None)
    return Glcm_feature

list1= ['strong personality', 'moderate personality' , 'weak personality']

    #Read Image
S_filename = filedialog.askopenfilename(title='Select Signature Image')
S_img = cv2.imread(S_filename)
        
if len(S_img.shape) == 3:
    G_img = cv2.cvtColor(S_img, cv2.COLOR_RGB2GRAY)
else:
    G_img=S_img.copy()

cv2.imshow('Input Image',G_img)
cv2.waitKey(0)           
    #Gaussian Filter and thresholding image
blur_radius = 2
blurred_image = ndimage.gaussian_filter(G_img, blur_radius)
threshold, binarized_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('Segmented Image',binarized_image)
cv2.waitKey(0)
    # Find the center of mass
r, c = np.where(binarized_image == 0)
r_center = int(r.mean() - r.min())
c_center = int(c.mean() - c.min())

    # Crop the image with a tight box
cropped = G_img[r.min(): r.max(), c.min(): c.max()]

    ## Signature Feature extraction
Average,Percentage = pressure.pressure(cropped)
top, middle, bottom = zones.findZone(cropped)

Glcm_feature_signature =GLCM_Feature(cropped)
Glcm_feature_signature=Glcm_feature_signature.flatten()

bw_img,angle1= segmentation.Segmentation(G_img)

feature_matrix1 = np.concatenate((Average,Percentage,angle1,top, middle, bottom,Glcm_feature_signature),axis=None)

Model_lod1 = joblib.load("Trained_H_Model.pkl")

#ypred = Model_lod.predict(cv2.transpose(Feature_matrix))
pred=Model_lod1.predict(cv2.transpose(feature_matrix1))

print(pred)
#import serial
#import time

#ser = serial.Serial('COM3',9600)
#time.sleep(1) #give the connection a second to settle

#if pred==1:
 #   time.sleep(1)
 #   ser.write(str.encode('A'))
    



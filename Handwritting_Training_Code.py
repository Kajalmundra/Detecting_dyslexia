import cv2
import os
import numpy as np
from scipy import ndimage as nd
from scipy import ndimage
from matplotlib import pyplot as plt
import joblib
import pressure
import zones
import feature_extraction
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import gabor
from skimage.filters import gabor_kernel
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


ANN_DATA1=[]
S_Data=[]
S_label=[]
cnt=0
cw_directory = os.getcwd()
H_dataset = cw_directory+'\Dataset2'
for filename in os.listdir(H_dataset):
    sub_dir=(H_dataset+'/' +filename)
    for img_name in os.listdir(sub_dir):
        img_dir=str(sub_dir+ '/' +img_name)
        print(img_dir)
        feature_matrix1 = feature_extraction.Feature_extraction(img_dir)
        #print(len(feature_matrix1))
        S_Data.append(feature_matrix1)
        S_label.append(int(filename))
    cnt+=1
    print(cnt)

## MLP Training      
model1 = MLPClassifier(activation='relu', verbose=True,
                                       hidden_layer_sizes=(100,), batch_size=30)
model1=model1.fit(np.array(S_Data), np.array(S_label))
ypred_MLP = model1.predict(np.array(S_Data))

plot_confusion_matrix(model1, np.array(S_Data), np.array(S_label))
plt.show()
S_ACC=accuracy_score(S_label,ypred_MLP)

print("Training ANN accuracy is",accuracy_score(S_label,ypred_MLP))
joblib.dump(model1, "Trained_H_Model.pkl")


## Train SVM
from sklearn.svm import SVC
def train_SVM(featuremat,label):
    clf = SVC(kernel = 'rbf', random_state = 0)
    clf.fit(np.array(S_Data), np.array(S_label))
    y_pred = clf.predict(np.array(featuremat))
    plot_confusion_matrix(clf, np.array(featuremat), np.array(label))
    plt.show()
    print("SVM Accuracy",accuracy_score(label,y_pred))
    return clf

svc_model1 = train_SVM(S_Data,S_label)
Y_SCM_S_pred= svc_model1.predict(S_Data)
SVM_S_ACC=accuracy_score(Y_SCM_S_pred,S_label)


plt.figure()
plt.bar(['ANN'],[S_ACC], label="ANN Accuracy", color='r')
plt.bar(['SVM'],[SVM_S_ACC], label="SVM Accuracy", color='g')
plt.legend()
plt.ylabel('Accuracy')
plt.show()

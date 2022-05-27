from tkinter import *
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
##
import glob
import json
import h5py
import imutils
from sklearn.datasets import load_files
from keras.models import load_model
import time
##
import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from tqdm import tqdm       
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
##
from scipy import ndimage as nd
from scipy import ndimage
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
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
##
from imutils.video import VideoStream
#
import segmentation
from tkinter import filedialog

def show_history_graph(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        # changing the title of our master widget      
        self.master.title("GUI")
        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)
        w = tk.Label(root, 
		 text=" Dyslexia Prediction using Handwritting and Eye Movement ",
		 fg = "light blue",
		 bg = "white",
		 font = "Helvetica 20 bold italic")
        w.pack()
        w.place(x=50, y=10)
        # creating a button instance
        quitButton = Button(self,command=self.Train_writting,text="Train Handwritting Image",fg="blue",activebackground="dark red",width=20)
        # placing the button on my window
        quitButton.place(x=50, y=60)
        quitButton = Button(self,command=self.Train_Eye_Movement,text="Train Eye movement Image",fg="blue",activebackground="dark red",width=20)
        quitButton.place(x=250, y=60)
        quitButton = Button(self,command=self.EYE_Tracking,text="Start Eye Tracking",fg="blue",activebackground="dark red",width=20)
        quitButton.place(x=450, y=60)
        quitButton = Button(self,command=self.Handwriting_Prediction,text="Select Handwritting Image",fg="blue",activebackground="dark red",width=20)
        quitButton.place(x=650, y=60)
        
        load = Image.open("logo.png")
        logo_img = ImageTk.PhotoImage(load)     
        image1=Label(self, image=logo_img,borderwidth=2, highlightthickness=5, height=300, width=400, bg='white')
        image1.image = logo_img
        image1.place(x=50, y=120)

        contents ="  Waiting for Results..."
        global T
        T = Text(self, height=20, width=25)
        T.pack()
        T.place(x=650, y=100)
        T.insert(END,contents)
        print(contents)

    def Train_writting(self, event=None):
        global T
        contents="Handwritting Feature extraction and Training"
        T = Text(self, height=20, width=25)
        T.pack()
        T.place(x=650, y=100)
        T.insert(END,contents)
        print(contents)  
        CNN_DATA1=[]
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
        contents="Handwritting Feature extraction and Training completed"
        T = Text(self, height=20, width=25)
        T.pack()
        T.place(x=650, y=100)
        T.insert(END,contents)
        print(contents)
        
    def Train_Eye_Movement(self, event=None):
        global T
        contents="Training EYE movement"
        T = Text(self, height=20, width=25)
        T.pack()
        T.place(x=650, y=100)
        T.insert(END,contents)
        print(contents)
        
        data2=[]
        data=[]
        featurematrix=[]
        label=[]
        label2=[]
        cw_directory = os.getcwd()
        #cw_directory='D:/Hand gesture/final_code'
        folder=cw_directory+'/eye dataset'
        for filename in os.listdir(folder):
            
            sub_dir=(folder+'/' +filename)
            for img_name in os.listdir(sub_dir):
                img_dir=str(sub_dir+ '/' +img_name)
                print(int(filename),img_dir)
                img = cv2.imread(img_dir)
                # Resize image
                img = cv2.resize(img,(128,128))
                if len(img.shape)==3:
                    img2 = cv2.resize(img,(32,32))
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    img2=img2.flatten()
                    data2.append(img2/255.0)
                    label2.append(int(filename))
                    
                data11=np.array(img)
                data.append(data11/255.0)
                label.append(int(filename))
         

        #target1=train_targets[label]
        ##

        def train_CNN(data,label):
            ##
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128,128, 3)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))

            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(36))

            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
           
            X_train, Y_test, X_label, Y_label = train_test_split(data,label, test_size=0.20)

            history = model.fit(np.array(X_train), np.array(X_label), epochs=20, 
                                validation_data=(np.array(Y_test), (Y_label)))
            
            show_history_graph(history)
            test_loss, test_acc = model.evaluate(np.array(Y_test), np.array(Y_label), verbose=2)
            print("Testing Accuracy is ", test_acc)
            print("Testing loss is ", test_loss)

            #hist=model.fit(np.array(data), (train_targets) ,validation_split=0.1, epochs=10, batch_size=64)
            model.save('eye_movement_trained.h5')
            return model

        # CNN Training
        model_CNN = train_CNN(data,label)
        Y_CNN=model_CNN.predict(np.array(data))
        contents="Training EYE movement completed"
        T = Text(self, height=20, width=25)
        T.pack()
        T.place(x=650, y=100)
        T.insert(END,contents)
        print(contents)

    def EYE_Tracking(self, event=None):
        global T
        contents="Starting Eye Movement based Dyslexia Prediction"
        T = Text(self, height=20, width=25)
        T.pack()
        T.place(x=650, y=100)
        T.insert(END,contents)
        print(contents)
        list1= ['looking at center','looking at left','looking at right','looking at up','looking at down']
        eye_cnn = tf.keras.models.load_model('trained_model_CNN1.h5')
        # histogram based equalization
        def histogram_equalization(img):
            r,g,b = cv2.split(img)
            f_img1 = cv2.equalizeHist(r)
            f_img2 = cv2.equalizeHist(g)
            f_img3 = cv2.equalizeHist(b)
            img = cv2.merge((f_img1,f_img2,f_img3))
            return img

        def get_index_positions_2(list_of_elems, element):
            ''' Returns the indexes of all occurrences of give element in
            the list- listOfElements '''
            index_pos_list = []
            for i in range(len(list_of_elems)):
                if list_of_elems[i] == element:
                    index_pos_list.append(i)
            return index_pos_list

        # Define paths
        base_dir = os.path.join( os.path.dirname( __file__ ), './' )
        prototxt_path = os.path.join(base_dir + 'model_data/deploy.prototxt')
        caffemodel_path = os.path.join(base_dir + 'model_data/weights.caffemodel')

        ##
        eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_eye.xml')

        # Read the model
        model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

        # Start video capture
        vs = cv2.VideoCapture(0)
        cnt=1
        # Display each video frame
        W1=1
        W2=1
        T1=[' ']
        seyeimg=1
        eyemovement=[]
        Dyslexia_result=[]
        n1=0
        n2=10


        while True:
            ret, frame = vs.read()
            frame = imutils.resize(frame, width = 750, height = 512)
            #frame = histogram_equalization(frame)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            
            model.setInput(blob)
            detections = model.forward()
            
            for i in range(0, detections.shape[2]):
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                confidence = detections[0, 0, i, 2]
                
                # If confidence > 0.4, show box around face
                if (confidence > 0.40):
                    f_img=frame[startY:endY,startX:endX]
                    f_img = histogram_equalization(f_img)
                    
                    roi_gray = cv2.cvtColor(f_img, cv2.COLOR_BGR2GRAY)
                    eyes = eye_cascade.detectMultiScale(roi_gray)

                    ##
                    cn=0
                    pred=None
                    for (ex,ey,ew,eh) in eyes:
                        #write Image
                        filename='eye image/'+str(seyeimg) +'.jpg'
                        seyeimg+=1
                        #cv2.imwrite(filename,f_img[ey:ey+eh,ex:ex+ew])
                        if cn==1:
                            one_eye=np.expand_dims(cv2.resize(f_img[ey:ey+eh,ex:ex+ew],(128,128)), axis=0)
                            pred=np.argmax(eye_cnn.predict(one_eye))
                            eyemovement.append(pred)
                            #print(pred)
                        #bounding eye Image
                        cv2.rectangle(f_img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                        frame[startY:endY,startX:endX]=f_img
                        cn+=1
                        if cn==2:
                            break
                    if pred is None:
                        text= 'Eyes Not Detected'
                    else:
                        text = list1[pred]
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 200, 200), 2)
                    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 150), 2)
                cv2.imshow("Frame", frame)
            if (len(eyemovement)>=10) and len(eyemovement)>=n2:
                eye_array = eyemovement[n1:n2]
                if len(np.unique(eye_array)) >2:
                    Dyslexia=1
                else:
                    Dyslexia=0
                    
                n1+=20
                n2+=20
                Dyslexia_result.append(Dyslexia)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vs.release()
        cv2.destroyAllWindows()

        number_of_positive = get_index_positions_2(Dyslexia_result, 1)
        number_of_negative = get_index_positions_2(Dyslexia_result, 0)

        if len(number_of_positive)>=10 or len(number_of_positive)>len(number_of_negative):
            print("Syntoms of Dyslexia detected")
            contents="Syntoms of Dyslexia detected"
            Dyslexia_eye=1

        else:
            print("Syntoms of Dyslexia NOT detected")
            contents="Syntoms of Dyslexia NOT detected"
            Dyslexia_eye=1

        
        T = Text(self, height=20, width=25)
        T.pack()
        T.place(x=650, y=100)
        T.insert(END,contents)
        print(contents)
        self.Dyslexia_eye=Dyslexia_eye
        
    def Handwriting_Prediction(self, event=None):
        global T
        contents="Starting Handwriting based Dyslexia Prediction"
        T = Text(self, height=20, width=25)
        T.pack()
        T.place(x=650, y=100)
        T.insert(END,contents)
        print(contents)
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

        list1= ['Dyslexia Handwriting', 'Normal Handwriting']

            #Read Image
        S_filename = filedialog.askopenfilename(title='Select Signature Image')
        S_img = cv2.imread(S_filename)
        Sh_img=cv2.resize(S_img,(300,50))
        cv2.imwrite('H_Image.png',Sh_img)        
        if len(S_img.shape) == 3:
            G_img = cv2.cvtColor(S_img, cv2.COLOR_RGB2GRAY)
        else:
            G_img=S_img.copy()

        load = Image.open("H_Image.png")
        logo_img = ImageTk.PhotoImage(load)     
        image1=Label(self, image=logo_img,borderwidth=2, highlightthickness=5, height=300, width=400, bg='white')
        image1.image = logo_img
        image1.place(x=50, y=120)
        
        cv2.imshow('Input Image',cv2.resize(G_img,(300,50)))
        cv2.waitKey(0)           
            #Gaussian Filter and thresholding image
        blur_radius = 2
        blurred_image = ndimage.gaussian_filter(G_img, blur_radius)
        threshold, binarized_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('Segmented Image',cv2.resize(binarized_image,(300,50)))
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
        Dyslexia_writing=pred[0]
        print(pred)
        contents=list1[pred[0]]
        T = Text(self, height=20, width=25)
        T.pack()
        T.place(x=650, y=100)
        T.insert(END,contents)
        print(contents)
        self.Dyslexia_writing=Dyslexia_writing
        
root = Tk()
#size of the window
root.geometry("900x450")
app = Window(root)
root.mainloop()  

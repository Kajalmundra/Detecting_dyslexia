import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from tqdm import tqdm       
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models


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
   
    X_train, Y_test, X_label, Y_label = train_test_split(X_train, X_label, test_size=0.20)

    history = model.fit(np.array(data), np.array(label), epochs=20, 
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

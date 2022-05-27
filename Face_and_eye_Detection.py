from imutils.video import VideoStream
import os
import numpy as np
import imutils
import cv2
import tensorflow as tf

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
    print("Sympntoms of Dyslexia detected")

else:
    print("Sympntoms of Dyslexia NOT detected")

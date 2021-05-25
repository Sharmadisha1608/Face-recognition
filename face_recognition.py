import numpy as np
import cv2
file_name=input("Enter File name : ")
cap=cv2.VideoCapture(0)
skip=0
face_data=[]
dataset_path="C:\\Users\\hp\\Documents\\faces\\"

face_cascade=cv2.CascadeClassifier("C:\\Users\\hp\\anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml")
while True:
    ret,frame=cap.read()
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if ret==False:
        continue
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    faces=sorted(faces,key=lambda f:f[2]*f[3],reverse=True)
    for face in faces:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        skip+=1
        if skip%10==0:
            face_data.append(face_section)
            print(skip)
    cv2.imshow("Video",frame)
    cv2.imshow("Section",face_section)
    key_pressed=cv2.waitKey(1)&0xFF
    if key_pressed==ord('q'):
        break
face_data=np.array(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
np.save(dataset_path+file_name+'.npy',face_data)
cap.release()
cv2.destroyAllWindows()

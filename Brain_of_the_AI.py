import numpy as np 
import cv2
import os

#First we need to load the method of recognition
#On the color part i used gray because its the easiest to pick and pixels will be easy to recognize
def Detector(frame):
             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             classifier_Option=cv2.CascadeClassifier('classifier_Option\\haarcascade_frontalface_default.xml')
             objects_detected=classifier_Option.detectMultiScale(gray,scaleFactor=1.32,minNeighbors=6)
             return gray , objects_detected

def training(directory):
	faces=[]
	faceID=[]

	for path,subdirnames,filenames in os.walk(directory):
		for filename in filenames:
			if filename.startswith("."):
				print("I could not read the System File")
				continue
	
			id=os.path.basename(path)
			img_path=os.path.join(path,filename)
			print("img_path: ",img_path)
			print("id: ",id)
			input_img=cv2.imread(img_path)
			if input_img is None:
				print("Image not loaded properply Spencer")
				continue
			face_rect,gray_img=Detector(input_img)	
			if len(face_rect)!=1:
				continue #Since the programm is being loaded single person images
			(x,y,w,h)=face_rect[0]
			roi_gray=gray_img[y:y+w,x:x+h]
			faces.append(roi_gray)
			faceID.append(int(id))
	return faces,faceID

def train_classifier(face,faceID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(face,np.array(faceID))
    return face_recognizer

def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(250, 0, 28),thickness=2)

def put_text(test_img,text,x,y):
	cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,2,(255, 0, 0),1)


    

import cv2
import os
import numpy as np
import Brain_of_the_AI as computer
cap = cv2.VideoCapture(0)
import requests
while True:
    def choiceCommand():
       while True:
            #souce = input('Input the source of the image :')
            frame = cap.read()

            input_img = 'me.jpg'
            #test_img,gray_img=cv2.imread(souce)
            objects_detected=computer.Detector(input_img)
            face_recognizer=cv2.face.LBPHFaceRecognizer_create()
            #data_name = input('Give Data name:')
            face_recognizer.read('newtrainingDatabase.yml')
            
            name={0:'Spencer',1:'Anyone'}
            
            
            for face in objects_detected:
                (x,y,w,h)=face
                
                roi_gray=gray_img[y:y+h,x:x+h]
                
                label,confidence=face_recognizer.predict(roi_gray)
                print("confidence: "+ confidence)
                #print("label: "+ label)
                
                computer.draw_rect(test_img,face)
                
                predicted_name=name[label]
                
                computer.put_text(test_img,predicted_name,x,y)
            
                resized_img=cv2.resize(test_img,(450,600))
            cv2.imshow("face detected", resized_img)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                 break

   
    def Live_stream_recognitio():
        pass

    
    def Training_the_Ai():
         data_name = input('Give the name of the your :')
         #training_img = input('Give training image path :')
         faces,faceID=computer.training('faceRecognitionAI\\trained_data')
         face_recognizer=computer.train_classifier(faces,faceID)
         face_recognizer.save('trained_data\\',data_name,'.yml')
         print('Training Process Complete......')
        
    def help():
        print('Use \n train   -for training your Ai \n run  -for running your Ai \n exit  -for exiting the programm')     
    
    
    if __name__ == "__main__":
        print('Type -h  for help')
        choice = input('what do you want to do :')
        if choice =='-h':
            help()
        elif choice =='run':
            choiceCommand()
        elif choice == 'train':
            Training_the_Ai()
        elif choice =='exit':
            cv2.destroyAllWindows()
            cap.release()
            quit()                
    


            
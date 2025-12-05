import cv2
import numpy as np
import os
import time
from datetime import datetime

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get the trained LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained_model.yml')

# Get user details from 'users.txt'
user_data = {}
if os.path.exists("users.txt"):
    with open("users.txt", "r") as file:
        for line in file:
            data = line.strip().split(",")
            if len(data) >= 5:
                user_id, name, age, gender, history = data[0], data[1], data[2], data[3], ','.join(data[4:])
                user_data[int(user_id)] = {
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "history": history
                }

def recognize_patient():
    recognized_user = None  
    error_message = None  
    try:
        cap = None
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY, cv2.CAP_MSMF]:
            try:
                cap = cv2.VideoCapture(0, backend)
                if cap.isOpened():
                    break  
            except:
                continue
        
        # camera not accessible 
        if cap is None or not cap.isOpened():
            return None, "Failed to open camera. Please check permissions."
        
        time.sleep(0.5)
        
        max_attempts = 3
        frame = None
        for _ in range(max_attempts):
            ret, frame = cap.read()
            if ret and frame is not None:
                break  
            time.sleep(0.2)  #
        
        cap.release()
        
        if frame is None:
            return None, "Failed to capture image. Please try again."
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        #no faces is detected
        if len(faces) == 0:
            return None, "No face detected. Please try again."
        
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]  # Crop the face
            
            try:
                face_img = cv2.resize(face_img, (100, 100))  #
                face_img = cv2.equalizeHist(face_img)  
                
                #LBPH face recognizer
                label, confidence = recognizer.predict(face_img)
                
                
                if confidence < 80:  
                    recognized_user = user_data.get(label) 
                    if recognized_user:
                        recognized_user['recognized_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        recognized_user['confidence'] = f"{100 - confidence:.2f}%"  
                    break  

            except Exception as e:
                return None, f"Recognition error: {str(e)}"
        
    except Exception as e:
        return None, f"System error: {str(e)}"
    
    # no user was recognized
    if recognized_user is None and error_message is None:
        error_message = "Face not recognized. Please try again."
        
    return recognized_user, error_message

'''
@app.route('/recognize-face', methods=['GET'])
def recognize_face():
    user, error = recognize_patient()
    return render_template('recognize.html', user=user, error=error)
'''

if __name__ == "__main__":
    print("Starting face recognition test...")

    # Call the function to recognize the face
    user, error = recognize_patient()

    if error:
        print(f"Error: {error}")  
    elif user:
        print(f"Recognized User:")  
        print(f"   ID       : {user['id']}")
        print(f"   Name     : {user['name']}")
        print(f"   Age      : {user['age']}")
        print(f"   Gender   : {user['gender']}")
        print(f"   History  : {user['history']}")
    else:
        print("Unknown error occurred.")

import cv2
import os
import time  

haar_cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

if face_cascade.empty():
    print(f" Error loading Haar cascade file. Check path: {haar_cascade_path}")
    exit()

# Create 'dataset' directory 
dataset_path = os.path.join(os.path.dirname(__file__), "dataset")
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Create 'users.txt' 
users_file = os.path.join(os.path.dirname(__file__), "users.txt")
if not os.path.exists(users_file):
    open(users_file, "w").close()

# Load existing user IDs from 'users.txt'
with open(users_file, "r") as f:
    existing_ids = [line.split(',')[0] for line in f.readlines()]

# Input user detail
while True:
    user_id = input("Enter a unique User ID: ")
    if user_id in existing_ids:
        print("This User ID already exists. Please enter a unique one.")
    else:
        break

user_name = input("Enter Full Name: ")
user_age = input("Enter Age: ")
user_gender = input("Enter Gender (M/F/O): ")
user_history = input("Enter Medical History (comma-separated): ")

# Save user info to 'users.txt'
with open(users_file, "a") as f:
    f.write(f"{user_id},{user_name},{user_age},{user_gender},{user_history}\n")

print("\n Capturing 30 face samples. Please look at the camera...")
print("Hold still, an image will be captured every 0.5 seconds.\n")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  
count = 0

while count < 30:
    ret, frame = cap.read()
    if not ret:
        print("Webcam access error.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        file_path = os.path.join(dataset_path, f"user_{user_id}_{count}.jpg")
        cv2.imwrite(file_path, face_img)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(f" Saved image {count}/30")

    cv2.imshow("Face Capture - Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Capture interrupted by user.")
        break

    time.sleep(0.5)  

cap.release()
cv2.destroyAllWindows()

print(f"\n{count} face images saved successfully for User ID: {user_id}")

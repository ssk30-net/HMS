import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import speech_recognition as sr
import pyttsx3 as px

# Speech Recognition Part
speech=sr.Recognizer()

# Sectioning
try:
    engine=px.init() # Engine for text to speech
except ImportError:
    print('Requested Driver not found')
except RuntimeError:
    print('Driver Fails to Initialize')

# Section property of engine
voices=engine.getProperty('voices')

engine.setProperty('voice','HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0') # setting the tts to zira the female voice
# for setting speech rate
rate=engine.getProperty('rate')
engine.setProperty('rate',rate)
# Function for speaking the text given
def speakfromtext_cmd(cmd):
    engine.say(cmd)
    engine.runAndWait()

# Function for reading voice
def read_voice_cmd():
    voice_text=''
    print("Listening...")

    try:
        with sr.Microphone() as source:
            audio = speech.listen(source=source, timeout=10, phrase_time_limit=5)
        voice_text=speech.recognize_google(audio)

    except sr.RequestError:
        print ('Network Error!!')
    return voice_text

# from PIL import ImageGrab
# Importing the images present in ImageAttendance folder
path = 'ImagesAttendance' # path for image or folder in which images are present
images = [] # list of images
classNames = [] # names of images
myList = os.listdir(path) # Printing the contents of folder
#print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
#print(classNames)

# Function for finding encodings of images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Function for mark attendance
def markAttendance(name,doc_name,age,gender):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring},{doc_name},{age},{gender}')


        """else:
            now=datetime.now()
            f.writelines(f'\n{name}')"""

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

# Updates new face when the face is unknown
def add_new_face(ImagesAttendance, name):
    # Open webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the webcam
        ret, frame = video_capture.read()

        # Display the frame
        cv2.imshow('Add New Face', frame)

        # Wait for 'q' key to be pressed to capture the face
        if cv2.waitKey(1):
            # Save the captured face image
            cv2.imwrite('ImagesAttendance/'+name+'.jpg', frame)
            cv2.waitKey(1)
            cv2.destroyWindow('Add New Face')
            break
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    # Finding the faces and comparing the faces which qwe have
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            speakfromtext_cmd("Which doctor would you like to see:")
            doc_name = read_voice_cmd()
            print('cmd:{}'.format(doc_name))
            speakfromtext_cmd("Please say your age:")
            age = read_voice_cmd()
            print('cmd:{}'.format(age))
            speakfromtext_cmd("Please say your gender (Say M/F/B/Nr):")
            gender = read_voice_cmd()
            print("cmd:{}".format(gender))
            markAttendance(name, doc_name, age, gender)
            break

        else:
            print("Unidentified")
            speakfromtext_cmd('Please say your name:')
            name = read_voice_cmd()
            print('cmd:{}'.format(name))
            add_new_face(images, name)
            speakfromtext_cmd("Please say your age:")
            age = read_voice_cmd()
            print('cmd:{}'.format(age))
            speakfromtext_cmd("Please say your gender (Say M/F/B/Nr):")
            gender = read_voice_cmd()
            print("cmd:{}".format(gender))
            speakfromtext_cmd("Which doctor would you like to see:")
            doc_name = read_voice_cmd()
            print('cmd:{}'.format(doc_name))
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name,doc_name,age,gender)
            break
    cv2.imshow('Face Recognition', img)
    cv2.waitKey(1)
cv2.destroyAllWindows()
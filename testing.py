# import cv2
import numpy as np
import time
import os
import mediapipe as mp
from matplotlib import pyplot as plt
import cv2
from keras.models import load_model

mp_holistic = mp.solutions.holistic #holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utils
model = load_model("Model/actions_model.h5")
model.load_weights("Model/actions_weights.h5")
actions = ['hello','thanks','forgot']


def extract_keypoints(result):
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in result.pose_landmarks.landmark] if result.pose_landmarks else np.zeros(33*4)).flatten()
    lh = np.array([[res.x,res.y,res.z] for res in result.left_hand_landmarks.landmark] if result.left_hand_landmarks else np.zeros(21*3)).flatten()
    rh = np.array([[res.x,res.y,res.z] for res in result.right_hand_landmarks.landmark] if result.right_hand_landmarks else np.zeros(21*3)).flatten()
    return np.concatenate([pose,lh,rh])

def mp_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    res = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,res

def draw_landmarks(image,results):
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(142,110,60)),mp_drawing.DrawingSpec(color=(142,110,60)))
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(142,110,60)),mp_drawing.DrawingSpec(color=(142,110,60)))
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(142,110,60)),mp_drawing.DrawingSpec(color=(142,110,60)))    
    return image



sequence = []
sentence = []
predictions = []
threshold = 0.5

# collections
cap = cv2.VideoCapture(0)
x=0
with mp_holistic.Holistic(min_detection_confidence=0.6,min_tracking_confidence=0.8) as holistic:
    while cap.isOpened():
        #read frame
        success,frame = cap.read()

        #detection         
        image,data = mp_detection(frame,holistic)
        #draw landmark
        image = draw_landmarks(image,data)
        kp = extract_keypoints(data)

        sequence.append(kp)
        sequence = sequence[-30:]
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            if np.unique(predictions[-10:])[0]==np.argmax(res):
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])   
            sequence=[] 
        
        # cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        # cv2.putText(image, ' '.join(sentence), (3,30), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # sentence=[]
        cv2.imshow("Feed",image)

        #breaking/exit feed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
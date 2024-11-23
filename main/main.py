from flask import Flask,request,jsonify,render_template,Response,redirect,url_for
from pymongo import MongoClient
import time
import keyboard

app = Flask(__name__,template_folder='templates', static_folder='Static')

# MongoDB connection
client = MongoClient('mongodb://localhost:27017')
db = client.aslmjp
collection = db.creds

value=""

# import cv2
import numpy as np
import os
import mediapipe as mp
from matplotlib import pyplot as plt
import cv2
from keras.models import load_model

mp_holistic = mp.solutions.holistic #holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utils
model2 = load_model("Model\model2_alphabet.h5")
model1 = load_model("Model\model_gestures.h5")
# model.load_weights("Model/actions_weights.h5")
PATH = os.path.join('KeyPoint_Data')
actions2 = np.array(os.listdir(PATH))
PATH2 = os.path.join('alphabets')
actions = np.array(os.listdir(PATH2))
turn = True
mode2="Model:Gesture model"
mode="Model:Aplha model"


def extract_keypoints(result):
    global turn
    rh = np.array([[res.x,res.y,res.z] for res in result.right_hand_landmarks.landmark] if result.right_hand_landmarks else np.zeros(21*3)).flatten()
    if turn:
        lh = np.array([[res.x,res.y,res.z] for res in result.left_hand_landmarks.landmark] if result.left_hand_landmarks else np.zeros(21*3)).flatten()
        pose = np.array([[res.x,res.y,res.z,res.visibility] for res in result.pose_landmarks.landmark] if result.pose_landmarks else np.zeros(33*4)).flatten()
        return np.concatenate([pose,lh,rh])
    return rh

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


def generate():
    global value,mode,mode2,turn,actions,actions2,model1,model2
    sequence = []
    threshold = 0.7
    cap = cv2.VideoCapture(0)
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
            if not turn:
                sequence = sequence[-15:]
                if len(sequence) == 15:
                    res = model2.predict(np.expand_dims(sequence, axis=0))[0]
                    # if res[np.argmax(res)] > threshold: 
                    value+=actions[np.argmax(res)] 
                    sequence=[]
            else:
                sequence = sequence[-30:]
                if len(sequence) == 30:
                    res = model1.predict(np.expand_dims(sequence, axis=0))[0]
                    # if res[np.argmax(res)] > threshold: 
                    value+=actions2[np.argmax(res)] 
                    sequence=[]

            frame = cv2.imencode('.jpg', image)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpg\r\n\r\n'+frame+b'\r\n\r\n')
            time.sleep(0.04)
            #breaking/exit feed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
            
            # shift key
            if keyboard.is_pressed("shift") : 
                turn=not turn
                mode,mode2=mode2,mode
                sequence=[]
                continue
            
            #space key
            if keyboard.is_pressed("space"):
                value+=" "
                sequence=[]
                continue
        
            # clear sentence
            if keyboard.is_pressed("tab"):
                value=""
                sequence=[]
                continue


        cap.release()
        cv2.destroyAllWindows()

@app.route("/mode_shift")
def mode_shift():
    global mode
    return jsonify(mode)

@app.route("/result")
def result():
    global value
    return jsonify(value)

@app.route("/video_feed")
def video_feed():
    return Response(generate(),mimetype='multipart/x-mixed-replace;boundary=frame')
    

@app.route('/dashb/<user>')
def index(user):
    return render_template('dash.html')


@app.route('/sign-up', methods=['POST'])
def sign_up():
    try:
        # Get the JSON payload from the request
        data = request.get_json()

        # Extract data from the payload
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')

        # Check if the username already exists
        existing_user = collection.find_one({'username': username})
        if existing_user:
            return jsonify({'status': 'error', 'message': 'Signup failed. Username already exists.', 'code': 117}), 200

        # Save the data to MongoDB
        user_data = {
            'username': username,
            'email': email,
            'password': password
        }

        result = collection.insert_one(user_data)

        # Check if the data was successfully inserted
        if result.inserted_id:
            return jsonify({'status': 'success', 'message': 'Signup successful', 'code': 113}), 200
        else:
            return jsonify({'status': 'error', 'message': 'Signup failed', 'code': 500}), 500

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e), 'code': 500}), 500

@app.route('/lgn', methods=['POST'])
def login():
    try:
        # Get the JSON payload from the request
        data = request.get_json()

        # Extract data from the payload
        username = data.get('username')
        password = data.get('password')

        # Check if the username exists
        user_data = collection.find_one({'username': username})

        if user_data:
            # Check if the provided password matches the stored password
            if user_data['password'] == password:
                return jsonify({'status': 'success', 'message': 'Login successful', 'code': 200}), 200
            else:
                return jsonify({'status': 'error', 'message': 'Login failed. Incorrect password.', 'code': 401}), 401
        else:
            return jsonify({'status': 'error', 'message': 'Login failed. User not found.', 'code': 404}), 404

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e), 'code': 500}), 500


@app.route('/',methods=['GET','POST'])
def login_page():
    return render_template("index.html")

@app.route('/signup',methods=['GET','POST'])
def signup():
    return render_template('signup.html')

@app.route('/logout')
def logout():
    return redirect(url_for('login_page'))

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000',debug=True)

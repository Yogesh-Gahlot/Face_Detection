from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2 as cv
import mediapipe as mp
import numpy as np
import base64
import io
from PIL import Image 

app = Flask(__name__)
socketio = SocketIO(app)

mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('frame')
def handle_frame(data):
    img_data = base64.b64decode(data.split(',')[1])
    img = Image.open(io.BytesIO(img_data))
    frame = np.array(img)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    
    with mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        results = face_mesh.process(frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )
    
    _, buffer = cv.imencode('.jpg', frame)
    encoded_frame = base64.b64encode(buffer).decode('utf-8')
    emit('processed_frame', f'data:image/jpeg;base64,{encoded_frame}')

if __name__ == '__main__':
    socketio.run(app, debug=True)
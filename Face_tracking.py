import cv2 as cv 
import mediapipe as mp 


mp_drawing =mp.solutions.drawing_utils
mp_face= mp.solutions.face_mesh

drawing_spec= mp_drawing.DrawingSpec(thickness=1,circle_radius= 1)

video =cv.VideoCapture(0)

with mp_face.FaceMesh(min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
    
    while True:
        ret, image= video.read()
        image =cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable= False
        results= face_mesh.process(image)
        image.flags.writeable=True
        
        image =cv.cvtColor(image, cv.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(image= image , landmark_list= face_landmarks,
                                            connections= mp_face.FACEMESH_TESSELATION ,
                                            landmark_drawing_spec= drawing_spec,
                                            connection_drawing_spec= drawing_spec)
    
        cv.imshow("Face Mesh",image)
        k= cv.waitKey(1)
        if k != -1:
            break

video.release()
cv.destroyAllWindows()
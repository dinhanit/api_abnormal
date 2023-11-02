import numpy as np
from utils import *
from MediapipeFaceMesh import get_landmark_from_image
import cv2

session = load_session("model.onnx")

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def Inference(img):
    cov_img = get_landmark_from_image(img)
    if np.all(cov_img==0):
        return 0
    input_data = np.array([[cov_img]])
    results = infer(session, input_data)
    results = softmax(results)
    if results[0][1]>=0.5:
        return 1
    else:
        return 0

position = (50, 50) 
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2

cap = cv2.VideoCapture(0) 
label =["Abnormal","Normal"]
color =[(0,0,255),(0,255,0)]
while True:
    ret, frame = cap.read()
    if not ret:
        break  
    idx = Inference(frame)
    cv2.putText(frame, label[idx], position, font, font_scale, color[idx], font_thickness)
    cv2.imshow("Camera Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

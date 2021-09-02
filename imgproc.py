import cv2
import numpy as np



def empty():
    pass

def getcontour(img, imgcontour):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1200 and area > 450:
            print(area)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
            if len(approx) >=8:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(imgcontour,(x,y),(x+w,y+h),(0,255,0),2)




cap = cv2.VideoCapture("red light/night.mov")

while True:
    ret, frame = cap.read()
    imgcontour = frame.copy()
    if ret:
        img = cv2.GaussianBlur(frame, (5,5), 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold1 = 255
        threshold2 = 255
        canny = cv2.Canny(img, threshold1, threshold2)
        dilate = cv2.dilate(canny, np.ones((11,11)), iterations=1)
        getcontour(dilate, imgcontour)
        cv2.imshow("frame", imgcontour)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
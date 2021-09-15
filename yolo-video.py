import cv2
import numpy as np

net = cv2.dnn.readNet('yolov4-tiny.weights' , 'yolov4-tiny.cfg')
    

vid = cv2.VideoCapture('video/night.mov')


def detect_color(imageFrame):
    label = None

    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0,100,100])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([160,100,100])
    upper_red2 = np.array([180,255,255])
    mask1 = cv2.inRange(hsvFrame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsvFrame, lower_red2, upper_red2)
    red_mask =  cv2.add(mask1, mask2)

    green_lower = np.array([40,50,50])
    green_upper = np.array([90,255,255])
    
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    yellow_lower = np.array([15,150,150])
    yellow_upper = np.array([35,255,255])

    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        label = "Red"

    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        label = "Green"

    contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        label = "Yellow"

    return label



codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps =  int(vid.get(cv2.CAP_PROP_FPS))
vid_width , vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) , int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
out= cv2.VideoWriter('night-out.avi' , codec , vid_fps , (vid_width,vid_height))


while True:
    _ , image = vid.read()

    height , width , _ = image.shape

    # converting the image to specfic form by that we can pass this as input to the model
    blob = cv2.dnn.blobFromImage(image , 1/255 , (416,416) ,(0,0,0) , swapRB = True , crop = False)

    net.setInput(blob)
    output_layer_names = net.getUnconnectedOutLayersNames()
    layeroutputs = net.forward(output_layer_names)

    #visualize
    #Extract the bounding boxes and confidences and predicted classes
    boxes=[]
    confidences=[]
    class_id=[]

    for output in layeroutputs:
        for detection in output:
            score = detection[5:]
            ids = np.argmax(score) 
            confidence = score[ids]
            if confidence > 0.8: 
                center_x = int(detection[0]*width)# to denoramalize multiplying with original h and w
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))


    # To avoid many boxes we are picking up the boxes with high confidence
    indexes = cv2.dnn.NMSBoxes(boxes,confidences ,0.5,0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            img = image[y:y+h, x:x+w]
            label = detect_color(img)
            cv2.rectangle(image , (x,y) , (x+w,y+h) , (0,0,0) ,2)
            cv2.putText(image , label  , (x,y+20) ,cv2.FONT_HERSHEY_PLAIN ,2 ,(0,0,255),2)

    cv2.imshow("Frame" , image)
    
    out.write(image)
    key = cv2.waitKey(1)
    if key == 27:
        break
vid.release()
out.release()
cv2.destroyAllWindows()

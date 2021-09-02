import cv2
import numpy as np
import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

net = cv2.dnn.readNet('yolov4.weights' , 'yolov4.cfg')
model = tf.keras.models.load_model('model')
print(model.summary())

classes = ['green', 'red', 'yellow']

    

vid = cv2.VideoCapture('video/night.mov')


codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps =  int(vid.get(cv2.CAP_PROP_FPS))
vid_width , vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) , int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
out= cv2.VideoWriter('output1.avi' , codec , vid_fps , (vid_width,vid_height))


s = 0
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
            if ids == 9:       
                confidence = score[ids]
                if confidence > 0.5: 
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
            img = np.expand_dims(img, axis=0)
            pred = model.predict(img)
            label = classes[np.argmax(pred)]
            # cv2.imwrite('data/'+str(s)+'.jpg', img)
            s = s + 1
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

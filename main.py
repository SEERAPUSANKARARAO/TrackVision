import cv2
import pandas as pd
from ultralytics import YOLO
from src.tracker import*
import cvzone


model=YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('D:/1_Client_project_resources/1_yolov8peoplecounter/vidp/vidp.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0
persondown={}
tracker=Tracker()
counter1=[]

personup={}
counter2=[]
cy1=194 #y cordinate for first line
cy2=220 #y cordinate for second line line
offset=6
while True:    
    ret,frame = cap.read()
    if not ret:
        break
#    frame = stream.read()

    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
   
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        
        c=class_list[d]
        if 'person' in c:

            list.append([x1,y1,x2,y2])
       
        
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
        if cy1<(cy+offset) and cy1>(cy-offset):
            cv2.rectangle(frame, (x3, y3),(x4,y4) ,(0,0,225), 2)
            cvzone.putTextRect(frame,f'{id}',(x3,y3),scale=2,thickness=3)
            persondown[id]=(cx,cy)
            
        if id in persondown:
            if cy2<(cy+offset) and cy2>(cy-offset):
                cv2.rectangle(frame, (x3, y3),(x4,y4) ,(0,225,225), 2)
                cvzone.putTextRect(frame,f'{id}',(x3,y3),scale=2,thickness=3)
                if counter1.count(id)==0:
                    counter1.append(id)
        #for counting people going upside
        if cy2<(cy+offset) and cy2>(cy-offset):
            cv2.rectangle(frame, (x3, y3),(x4,y4) ,(0,0,225), 2)
            cvzone.putTextRect(frame,f'{id}',(x3,y3),scale=2,thickness=3)
            personup[id]=(cx,cy)
            
        if id in personup:
            if cy1<(cy+offset) and cy1>(cy-offset):
                cv2.rectangle(frame, (x3, y3),(x4,y4) ,(0,225,225), 2)
                cvzone.putTextRect(frame,f'{id}',(x3,y3),scale=2,thickness=3)
                if counter2.count(id)==0:
                    counter2.append(id)
                    
    cv2.line(frame,(3,cy1),(1018,cy1),(0,255,0),2)
    cv2.line(frame,(5,cy2),(1019,cy2),(0,255,255),2)
   
    down=len(persondown)
    up=len(personup)
    
    cvzone.putTextRect(frame,f'down:{down}',(50,60),scale=2,thickness=2)
    cvzone.putTextRect(frame,f'up:{up}',(60,90),scale=2,thickness=2)
    
    
    print("------------COUNT OF PEOPLE GOING DOWN-------------->>\n",len(counter1))
    print("------------COUNT OF PEOPLE GOING UP-------------->>\n",len(counter2))
    
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()


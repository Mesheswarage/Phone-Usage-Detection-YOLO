from ultralytics import YOLO
import cv2
import math
import numpy as np


def get_angle(index,kpn1,kpn2,kpn3):
  kp1 = results[0].keypoints.data[index][kpn1]
  kp2 = results[0].keypoints.data[index][kpn2]
  kp3 = results[0].keypoints.data[index][kpn3]
  VBA = (int(kp2[0]) - int(kp1[0]), int(kp2[1]) - int(kp1[1]))
  VBC = (int(kp2[0]-int(kp3[0])),int(kp2[1]-int(kp3[1])))
  magnitudeAB = math.sqrt(VBA[0]**2 + VBA[1]**2)
  magnitudeBC = math.sqrt(VBC[0]**2 + VBC[1]**2)
  dot_product = VBA[0] * VBC[0] + VBA[1] * VBC[1]
  angle_rad = math.acos(dot_product / (magnitudeAB * magnitudeBC))
  angle_deg = math.degrees(angle_rad)
  return angle_deg


def head_angle(index):
  x1,y1,_ = results[0].keypoints.data[index][6]
  x2,y2,_ = results[0].keypoints.data[index][5]
  x3,y3,_ = results[0].keypoints.data[index][11]
  x4,y4,_ = results[0].keypoints.data[index][12]
  nosex,nosey,_ = results[0].keypoints.data[index][0]
  nx=(x1+x2)/2
  ny=(y1+y2)/2
  hx=(x3+x4)/2
  hy=(y3+y4)/2
  VBA = (int(nx) - int(nosex), int(ny) - int(nosey))
  VBC = (int(nx)-int(hx) ,int(ny)-int(hy))
  magnitudeAB = math.sqrt(VBA[0]**2 + VBA[1]**2)
  magnitudeBC = math.sqrt(VBC[0]**2 + VBC[1]**2)
  dot_product = VBA[0] * VBC[0] + VBA[1] * VBC[1]
  angle_rad = math.acos(dot_product / (magnitudeAB * magnitudeBC))
  angle_deg = math.degrees(angle_rad)
  return angle_deg

def head_angle2(index):
  _,y1,_ = results[0].keypoints.data[index][1]
  _,y2,_ = results[0].keypoints.data[index][2]
  _,y3,_ = results[0].keypoints.data[index][3]
  _,y4,_ = results[0].keypoints.data[index][4]
  eyesy=(y1+y2)/2
  earsy=(y3+y4)/2
  if eyesy > earsy:
     return True
  if eyesy > earsy:
     return False

def hands_down(index):
   y1 = results[0].keypoints.data[index][6][1]
   y2 = results[0].keypoints.data[index][10][1]
   if int(y1) < int(y2):
     return True
   else: return False
   
def hand_distance(index):
   wx1,wy1,_ = results[0].keypoints.data[index][9]
   wx2,wy2,_ = results[0].keypoints.data[index][10]
   distance = np.abs(wx2- wx1)
   return distance

def check(i,lha,rha,ha,hwd):
   threshold_hands = 108
   threshold_head = 131
   threshold_distance = 8
   rha = get_angle(i,6,8,10)
   lha = get_angle(i,5,7,9)
   ha = head_angle(i)
   hwd = hand_distance(i)
   if rha < threshold_hands:
      right_hand =True
   else:
      right_hand =False

   if lha < threshold_hands:
      left_hand = True
   else:
      left_hand =False

   if ha < threshold_head:
      head_down = True
   else:
      head_down =False

   if hwd < threshold_distance:
      hands = True
   else:
      hands =False

   if hands_down(i):
      handsdown = True
   else:
      handsdown = False
   
   if (right_hand or left_hand) and (head_down or hands or head_angle2(i)) and handsdown:
       return True
   else:
      return False

def immain():
   global im
   im = cv2.imread(source)
   for i in range(0,len(results[0].boxes.data)):
    #im= results[0].plot(bounding_box=False)
       x1,y1,x2,y2,score,class_id= results[0].boxes.data[i]
       x1 = int(x1)
       y1 = int(y1)
       x2 = int(x2)
       y2 = int(y2)
       cv2.rectangle(im,(x1,y1),(x2,y1),(255,0,0),2)
       rha = get_angle(i,6,8,10)
       lha = get_angle(i,5,7,9)
       ha = head_angle(i)
       hwd = hand_distance(i)
    #print(f"person {i} right hand angle : {rha}")
    #print(f"person {i} left hand angle : {lha}")
    #print(f"person {i} head angle : {ha}")
    #print(f"person {i} distance between hands : {hwd}")
       phone = check(i,lha,rha,ha,hwd)
       if phone:
          cv2.putText(im, str(f"{i}- Using a phone"), (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
       else:
          cv2.putText(im, str(f"{i}"), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


model = YOLO("yolov8n-pose.pt")
source = 'D:/Lec2023/Campus/4thYear/EES/ComputerVision/YOLO pose -20230815/data/data/12.jpg'
results = model.predict(source=source,save=True,conf=0.5)
r=results[0].keypoints.data.tolist()
immain()
while(1):
    cv2.imshow("Annotated Image",im)
    if(cv2.waitKey(1)& 0xFF == ord('q')):
       break
#p1=results[0].boxes.data[0] 
#cv2.rectangle(im,(int(p1[0]),int(p1[1])),(int(p1[2]),int(p1[3])),(0,255,0),5)
cv2.destroyAllWindows()


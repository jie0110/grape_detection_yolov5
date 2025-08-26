# coding=utf-8
import time
import rospy
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import socket
import time
import os
import cv2
import torch
import numpy as np
from utils.general import non_max_suppression, scale_coords, xyxy2xywh

model = torch.load('/home/caizhai/python3_ws/src/grape_detection_yolov5/yolov5s_grape.pt', map_location='cpu')['model'].float().fuse().eval()

#手眼转换矩阵
calibrateArray=[[1,    0,    0,   -0.05],   #左右 
                                [0,     1,    0,   -0.08],  #上下   上偏增大  下偏减小 
                                [0,     0,     1,    0.12],
                                [0,     0,      0,          1]]             

#右臂拍照点矩阵   
end2baseR=[[ 2.67027788e-06, -1.50758971e-02, -9.99886352e-01,-0.035752],
                           [ 2.01295611e-08,  9.99886352e-01, -1.50758971e-02, -0.300000],
                           [ 1.00000000e+00, 2.01295611e-08,  2.67027788e-06 , 0.059841],
                           [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
  #位置信息初始化
coordinateR=[1,2,3] 
positionR=[1,2,3]  
bridge2 = CvBridge()
fx = 609.134765 
fy = 608.647949
ppx = 312.763214
ppy = 240.882049
low_depthr = np.array(200)
high_depthr = np.array(1000) 


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def predict(model, img):
    # img = cv2.imread(os.path.join(image_path, img_name))
    img_org = img
    h, w, s = img_org.shape
    img = letterbox(img, new_shape=640)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)
    image = img.astype(np.float32)
    image /= 255.0
    image = np.expand_dims(image, 0)

    image = torch.from_numpy(image)

    # print("shape tensor image:", image.shape)

    pred = model(image)[0]
    # print("pred shape:", pred.shape)
    temp_img = None
    pred = non_max_suppression(pred, 0.5, 0.5, None)
    # print(pred[0])
    num_boxes = 0
    box = []
    ht=[]
    for i, det in enumerate(pred):
        im0 = img_org
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(image.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
            # Write results
            for *xyxy, conf, cls in reversed(det):
                check = True
                bbox = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                bbox_new = bbox.clone() if isinstance(bbox, torch.Tensor) else np.copy(bbox)
                # line = (cls, *xywh, conf)  # label format
                bbox_new[0] = bbox[0] - bbox[2] / 2  # top left x
                bbox_new[1] = bbox[1] - bbox[3] / 2  # top left y
                bbox_new[2] = bbox[0] + bbox[2] / 2  # bottom right x
                bbox_new[3] = bbox[1] + bbox[3] / 2  # bottom right y

                bbox_new[0] = bbox_new[0] * w
                bbox_new[2] = bbox_new[2] * w
                bbox_new[1] = bbox_new[1] * h
                bbox_new[3] = bbox_new[3] * h
                # print("class: ", labels[int(cls)])
                # print("conf: ", float(conf))
                cv2.rectangle(img_org, (int(bbox_new[0]), int(bbox_new[1])), (int(bbox_new[2]), int(bbox_new[3])),
                              (0, 255, 0), 3)
                box.append(((int(bbox_new[0] + (bbox_new[2] - bbox_new[0]) / 2)),
                            (int(bbox_new[1] + (bbox_new[3] - bbox_new[1]) / 2))))
                ht.append((int(bbox_new[3] - bbox_new[1]) / 2))
                num_boxes = num_boxes + 1
    return img_org, num_boxes, box,ht

def socket_service_dataR(tableR):
    s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s2.connect(("192.168.1.107",6699))
    while True:
        buf_r1= s2.recv(2048) 
        buf_r1 = buf_r1.decode()  
        if str(buf_r1) =="R":
            s2.send(tableR.encode())
            rospy.loginfo("send ok")
        elif str(buf_r1) =="p":
            rospy.loginfo("pick ok")
            break
        else:
             rospy.loginfo("Connection  error!!!")

def callbackr(data1, data2):
    real_zr_list=[]
    positionR_list=[]
    table_all=''
    num=0
    color_imager = bridge2.imgmsg_to_cv2(data2, 'bgr8')
    depth_imager = bridge2.imgmsg_to_cv2(data1, '16UC1')
    maskr =cv2.inRange(depth_imager,low_depthr,high_depthr)
    imager = cv2.bitwise_and(color_imager, color_imager, mask=maskr)
    imager[:,0:60]=0
    # start_time=time.time()
    frame, num_boxes, box,ht = predict(model, imager)
    # end_time=time.time()
    # print(end_time-start_time)
    # cv2.imshow("frame",frame)
    # cv2.waitKey(1)
    # cv2.imwrite(str(time.time())+".png",frame)
    rospy.loginfo("num grapes:%d", num_boxes)
    if num_boxes==0:
         rospy.set_param("zr",1)
         rospy.loginfo("foward_r:1m")
    else:
        # time3=time.time()
        for i in range(num_boxes):
            # time1=time.time()
            real_zr = depth_imager[box[i][1], box[i][0]] * 0.001
            real_xr = (box[i][0] - ppx) / fx * real_zr
            real_yr = (box[i][1]-ht[i] - ppy) / fy * real_zr
            # time2=time.time()
            # print(time2-time1)
            rospy.loginfo("camera position:x1=%f,y1=%f,z1=%f", real_xr, real_yr, real_zr)
        # time4=time.time()
        # print(time4-time3)
            if real_zr == 0:
                continue
            elif real_zr<=0.65:
                positionR_list.append((real_xr,real_yr,real_zr))
            else:
                real_zr_list.append(real_zr)
        if  positionR_list==[]:
            rospy.loginfo("pickunable")
        else:
            positionR_list.sort(key=lambda positionR_list: positionR_list[2])  # 基于Z坐标由近到远排序
            rospy.loginfo("pickable grapes:%d",len(positionR_list))
            for j in range(len(positionR_list)):
                num+=1
                for i in range(3):
                    coordinateR[i] = positionR_list[j][0] * calibrateArray[i][0] + positionR_list[j][1]  * calibrateArray[i][1] + positionR_list[j][2]  * \
                                    calibrateArray[i][2] + calibrateArray[i][3] * 1
                rospy.loginfo("end position:xl=%f,yl=%f,zl=%f", coordinateR[0], coordinateR[1], coordinateR[2])
                for i in range(3):
                    positionR[i] = coordinateR[0] * end2baseR[i][0] + coordinateR[1] * end2baseR[i][1] + coordinateR[2] * \
                                end2baseR[i][2] + end2baseR[i][3] * 1
                rospy.loginfo("base position:xl=%f,yl=%f,zl=%f", positionR[0], positionR[1], positionR[2])
                tabelR = str(positionR[0]) + ',' + str(positionR[1]) + ',' + str(positionR[2])
                table_all=table_all+tabelR+','
            table_all=str(num)+','+table_all  
            socket_service_dataR(table_all) 
        if real_zr_list ==[] :
            return
        else:
            foward_r=min(real_zr_list)
            rospy.set_param('zr', float(foward_r))
            rospy.loginfo("foward_r:%fm",foward_r)
    # cv2.imwrite(str(time.time())+".png",frame)
    while True:
        if rospy.get_param("zr")==0:
            break
        


if __name__ == '__main__':
    rospy.init_node('get_imager', anonymous=True)
    # rospy.set_param("z",0)
    # colorr = message_filters.Subscriber("/camera2/color/image_raw", Image,queue_size=1)
    # depthr = message_filters.Subscriber("/camera2/aligned_depth_to_color/image_raw", Image,queue_size=1)
    colorr = message_filters.Subscriber("/camera2/color/image_raw", Image,queue_size=1,buff_size=52428800)
    depthr = message_filters.Subscriber("/camera2/aligned_depth_to_color/image_raw", Image,queue_size=1,buff_size=52428800)
    # color_depth1 = message_filters.TimeSynchronizer([color1, depth1], 10)
    color_depthr = message_filters.TimeSynchronizer([depthr,colorr], queue_size=1)
    color_depthr.registerCallback(callbackr)  
    rospy.spin()


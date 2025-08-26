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
from utils.datasets import letterbox
model = torch.load('/home/caizhai/python3_ws/src/grape_detection_yolov5/yolov5s_grape.pt', map_location='cpu')['model'].float().fuse().eval()

# 手眼转换矩阵
calibrateArray = [[1,   0,    0,   -0.02],  #左右，左偏增大，右偏减小
                                   [0,    1,   0,   -0.08],    #上下
                                   [0,    0,   1,    0.12],   #前后
                                   [0,    0,   0,       1]]
# 左臂拍照点矩阵
end2baseL = [[3.00773947e-06, -1.33629425e-02, 9.99910712e-01, 3.67820000e-02],
                             [-2.27003276e-05, 9.99910712e-01, 1.33629426e-02, -3.00002000e-01],
                             [-1.00000000e+00, -2.27384929e-05, 2.70412774e-06, 1.80010000e-02],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
# 位置信息初始化
coordinateL = [1, 2, 3]
positionL = [1, 2, 3]
bridge1 = CvBridge()
fx = 608.4118
fy = 607.9635
ppx = 311.2207
ppy = 255.7856
low_depthl = np.array(200)
high_depthl = np.array(1000)


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
    # temp_img = None
    # print("pred shape:", pred.shape)    low_depthl = np.array(20)
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
                # print(bbox_new)
                box.append(((int(bbox_new[0] + (bbox_new[2] - bbox_new[0]) / 2)),
                            (int(bbox_new[1] + (bbox_new[3] - bbox_new[1]) / 2))))
                ht.append( int((bbox_new[3] - bbox_new[1]) / 2))
                num_boxes = num_boxes + 1
    return img_org, num_boxes, box,ht

def socket_service_dataL(tableL):
    s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s1.connect(("192.168.1.108",6699))
    while True:
        buf_l1= s1.recv(2048) 
        buf_l1 = buf_l1.decode()  
        if str(buf_l1) =="L":
            s1.send(tableL.encode())
            rospy.loginfo("send ok")
        elif str(buf_l1) =="m":
            rospy.loginfo("pick ok")
            break
        else:
            rospy.logerr ("Connection  error!!!")

def callback1(data1, data2):
    real_zl_list=[]
    positionL_list=[]
    table_all=''
    num=0
    color_imagel = bridge1.imgmsg_to_cv2(data2, 'bgr8')
    depth_imagel = bridge1.imgmsg_to_cv2(data1, '16UC1')
    maskl = cv2.inRange(depth_imagel, low_depthl, high_depthl)
    imagel = cv2.bitwise_and(color_imagel, color_imagel, mask=maskl)
    imagel[:,480:640]=0
    frame, num_boxes, box,ht = predict(model, imagel)
    #cv2.imshow("frame",frame)
    #cv2.waitKey(0)
    #cv2.imwrite(str(time.time())+".png",frame)
    rospy.loginfo("num grapes:%d", num_boxes)
    if num_boxes==0:
        rospy.set_param("zl",1)
        rospy.loginfo("foward_r:1m")
    else:
        for i in range(num_boxes):
            real_zl = depth_imagel[box[i][1], box[i][0]] * 0.001
            real_xl = (box[i][0] - ppx) / fx * real_zl
            real_yl = (box[i][1]-ht[i]- ppy) / fy * real_zl
            rospy.loginfo("camera position:x1=%f,y1=%f,z1=%f", real_xl, real_yl, real_zl)
            if real_zl == 0:
                continue
            elif real_zl<=0.65:
                positionL_list.append((real_xl,real_yl,real_zl))
            else:
                real_zl_list.append(real_zl)
        if positionL_list==[]:
            rospy.loginfo("pickunable")
        else:
            positionL_list.sort(key=lambda positionL_list: positionL_list[2])  # 基于Z坐标由近到远排序
            # print(positionL_list)
            rospy.loginfo("pickable grapes:%d",len(positionL_list))
            for j in range(len(positionL_list)):
                num+=1
                for i in range(3):
                    coordinateL[i] = positionL_list[j][0] * calibrateArray[i][0] + positionL_list[j][1] * calibrateArray[i][1] + positionL_list[j][2] * \
                                    calibrateArray[i][2] + calibrateArray[i][3] * 1
                rospy.loginfo("end position:xl=%f,yl=%f,zl=%f", coordinateL[0], coordinateL[1], coordinateL[2])
                for i in range(3):
                    positionL[i] = coordinateL[0] * end2baseL[i][0] + coordinateL[1] * end2baseL[i][1] + coordinateL[2] * \
                                end2baseL[i][2] + end2baseL[i][3] * 1
                rospy.loginfo("base position:xl=%f,yl=%f,zl=%f", positionL[0], positionL[1], positionL[2])
                tabelL = str(positionL[0]) + ',' + str(positionL[1]) + ',' + str(positionL[2])
                table_all=table_all+tabelL+','
            table_all=str(num)+','+table_all
            socket_service_dataL(table_all)
        if real_zl_list==[]:
            return
        else:
            foward_l=min(real_zl_list) 
            rospy.set_param('zl', float(foward_l))
            rospy.loginfo("foward_l:%fm",foward_l)
    # cv2.imwrite(str(time.time())+".png",frame)
    while True:
        if rospy.get_param("zl")==0:
            break

if __name__ == '__main__':
    rospy.init_node('get_image1', anonymous=True)
    # rospy.set_param("z",0)
    # color1 = message_filters.Subscriber("/camera1/color/image_raw", Image, queue_size=1)
    # depth1 = message_filters.Subscriber("/camera1/aligned_depth_to_color/image_raw", Image, queue_size=1)
    color1 = message_filters.Subscriber("/camera1/color/image_raw", Image, queue_size=1, buff_size=52428800)
    depth1 = message_filters.Subscriber("/camera1/aligned_depth_to_color/image_raw", Image, queue_size=1,buff_size=52428800)
    # color_depth1 = message_filters.TimeSynchronizer([color1, depth1], 10)
    color_depth1 = message_filters.TimeSynchronizer([depth1, color1], queue_size=1)
    # color_depth1 = message_filters.ApproximateTimeSynchronizer([color1, depth1], 1, 0.1, allow_headerless=True)
    color_depth1.registerCallback(callback1)
    rospy.spin()

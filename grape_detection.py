import time
import os
import cv2
import torch
import numpy as np
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.datasets import letterbox
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('weights/yolov5m_grape.pt', map_location=device)['model'].float().fuse().eval()
model.to(device)

labels = ['Grape Chardonnay','Grape Cabernet Franc', 'Grape Cabernet Sauvignon', 'Grape Sauvignon Blanc', 'Grape Syrah']


def predict(model, img):
    img_org = img.copy()
    h,w,s = img_org.shape
    img = letterbox(img, new_shape = 640)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1) 
    image = img.astype(np.float32)
    image /= 255.0
    image = np.expand_dims(image, 0)

    image = torch.from_numpy(image)

    image = image.to(device)
    pred = model(image)[0]
    temp_img = None
    pred = non_max_suppression(pred, 0.5, 0.5,None)
    num_boxes = 0
    boxes = []  
    for i, det in enumerate(pred):
            im0 = img_org
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  
            if len(det):
                det[:, :4] = scale_coords(image.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    check = True 
                    bbox = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
                    bbox_new = bbox.clone() if isinstance(bbox, torch.Tensor) else np.copy(bbox)
                    bbox_new[0] = bbox[0] - bbox[2] / 2  # top left x
                    bbox_new[1] = bbox[1] - bbox[3] / 2  # top left y
                    bbox_new[2] = bbox[0] + bbox[2] / 2  # bottom right x
                    bbox_new[3] = bbox[1] + bbox[3] / 2  # bottom right y

                    bbox_new[0] = bbox_new[0] * w
                    bbox_new[2] = bbox_new[2] * w
                    bbox_new[1] = bbox_new[1] * h
                    bbox_new[3] = bbox_new[3] * h
                    cv2.rectangle(img_org,(int(bbox_new[0]), int(bbox_new[1])), (int(bbox_new[2]), int(bbox_new[3])), (0,255,0), 3)
                    boxes.append(bbox_new)

    return img_org, num_boxes, boxes   


def main():
    frame_rate = 1
    prev = 0
    cap = cv2.VideoCapture("grape.mp4")
    cap.set(cv2.CAP_PROP_FPS, 1)
    while(True):
        time_elapsed = time.time() - prev
        ret, frame = cap.read()
        if not ret:
            print("视频读取结束或出错")
            break
        if time_elapsed > 1./frame_rate:
            prev = time.time()
            frame, num_boxes, boxes = predict(model, frame)
            frame_t = cv2.putText(frame, "grapes detected: " + str(num_boxes), (00, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)
            cv2.imshow('frame',frame_t)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
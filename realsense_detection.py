import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.datasets import letterbox


# 配置深度和彩色流
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 获取深度传感器的深度比例
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('weights/yolov5m_grape.pt', map_location=device)['model'].float().fuse().eval()
model.to(device)

labels = ['Grape Chardonnay', 'Grape Cabernet Franc', 'Grape Cabernet Sauvignon', 'Grape Sauvignon Blanc', 'Grape Syrah']

align = rs.align(rs.stream.color)

def predict(model, depth_frame, color_frame, depth_scale):
    color_image = np.asanyarray(color_frame.get_data())
    img_org = color_image.copy()
    h, w, s = img_org.shape
    img = letterbox(img_org, new_shape=640)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)
    image = img.astype(np.float32)
    image /= 255.0
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image).to(device)

    pred = model(image)[0]
    pred = non_max_suppression(pred, 0.7, 0.5, None)
    boxes_3d = []
    for i, det in enumerate(pred):
        im0 = img_org
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        if len(det):
            det[:, :4] = scale_coords(image.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                bbox = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                bbox_new = [bbox[0] * w - bbox[2] * w / 2, bbox[1] * h - bbox[3] * h / 2, bbox[0] * w + bbox[2] * w / 2, bbox[1] * h + bbox[3] * h / 2]

                # 获取边界框中心点的像素坐标
                center_x = int((bbox_new[0] + bbox_new[2]) / 2)
                center_y = int((bbox_new[1] + bbox_new[3]) / 2)

                # 获取深度值
                depth_value = depth_frame.get_distance(center_x, center_y)

                # 获取相机内参
                depth_intrin = depth_frame.profile.as_video_stream_profile().get_intrinsics()

                # 2D像素+深度转3D坐标
                center_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [center_x, center_y], depth_value)
                center_3d = [int(coord * 1000) for coord in center_3d]  # 转换为毫米，保留整数
                cv2.putText(img_org,f"{center_3d}", (center_x, center_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.rectangle(img_org, (int(bbox_new[0]), int(bbox_new[1])), (int(bbox_new[2]), int(bbox_new[3])), (0, 255, 0), 2)
                cv2.circle(img_org, (center_x, center_y), 5, (0, 0, 255), -1)
                boxes_3d.append(center_3d)

    return img_org, boxes_3d

def main():
    frame_rate = 1
    prev = 0
    while True:
        # 等待深度和彩色帧
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # cv2.imshow('color frame', np.asanyarray(color_frame.get_data()))
        if not depth_frame or not color_frame:
            continue

        # time_elapsed = time.time() - prev
        # if time_elapsed > 1. / frame_rate:
        #     prev = time.time()
        frame, boxes_3d = predict(model, depth_frame, color_frame, depth_scale)
        frame = cv2.putText(frame, f"grapes detected: {len(boxes_3d)}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # 停止流
    pipeline.stop()


if __name__ == '__main__':
    main()

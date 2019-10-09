import time
import sys
import pathlib
import logging

import cv2
import torch 
import caffe
import numpy as np

from tools.test_detect import MtcnnDetector
from tools.utils import convert_to_minimum_square, convert_to_square
import config

logger = logging.getLogger("app")
formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
console_handler.formatter = formatter  # 也可以直接给formatter赋值 


def draw_images(net, img, bboxs, landmarks):  # 在图片上绘制人脸框及特征点
    num_face = bboxs.shape[0]
    if num_face == 0:
        return img
    bboxs = convert_to_minimum_square(bboxs)
    for i in range(num_face):
        x1 = int(bboxs[i, 0])
        y1 = int(bboxs[i, 1])
        x2 = int(bboxs[i, 2])
        y2 = int(bboxs[i, 3])
        print(x1, y1, x2, y2)
        w = x2 - x1
        h_center =  (y1 + y2) // 2
        y11 = h_center - w // 2
        y22 = h_center + w // 2
        cv2.rectangle(img, (int(bboxs[i, 0]), int(bboxs[i, 1])), (int(
            bboxs[i, 2]), int(bboxs[i, 3])), (0, 255, 0), 3)
        
        if y1 >= y2 or x1 >= x2 or x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            return img
        roi = img[y1: y2, x1: x2+1, :]
        gray_img = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        w, h = 60, 60
        
        res = cv2.resize(gray_img, (w, h), 0.0, 0.0, interpolation=cv2.INTER_CUBIC)
        resize_mat = np.float32(res)
                
        m = np.zeros((w, h))
        sd = np.zeros((w, h))
        mean, std_dev = cv2.meanStdDev(resize_mat, m, sd)
        new_m = mean[0][0]
        new_sd = std_dev[0][0]
        new_img = (resize_mat - new_m) / (0.000001 + new_sd)

        if new_img.shape[0] != net.blobs['data'].data[0].shape or new_img.shape[1] != net.blobs['data'].data[1].shape:
            print("Incorrect dimension, resize to correct dimensions.")

        net.blobs['data'].data[...] = new_img
        landmark_time_start = time.time()
        out = net.forward()
        landmark_time_end = time.time()
        landmark_time = landmark_time_end - landmark_time_start
        print("landmark time is {}".format(landmark_time))
        points = net.blobs['Dense3'].data[0].flatten()

        point_pair_l = len(points)
        print("num points:", point_pair_l)
        for i in range(point_pair_l // 2):
            x = points[2*i] * (x2 - x1) + x1
            y = points[2*i+1] * (y2 - y1) + y1
            cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 1)

    return img


if __name__ == '__main__':
    mtcnn_detector = MtcnnDetector(min_face_size=24, use_cuda=False)  # 加载模型参数，构造检测器
    mtcnn_detector.pnet_detector.load_state_dict(torch.load('./results/pnet/check_point/5_landmarks_model_050.pth'))
    mtcnn_detector.rnet_detector.load_state_dict(torch.load('./results/rnet/check_point/5_landmarks_model_050.pth'))
    mtcnn_detector.onet_detector.load_state_dict(torch.load('./results/onet/check_point/5_landmarks_model_050.pth'))
    logger.info("Init the MtcnnDetector.")
    project_root = pathlib.Path()
    inputPath = pathlib.Path('/dev/data/img_ibug_300W_tmp')
    # inputPath = pathlib.Path('/home/ubuntu/dataset/user/')
    outputPath = project_root / "data" / "you_result" / "onet"
    outputPath.mkdir(exist_ok=True)
    caffe.set_mode_cpu()
    net = caffe.Net('../temp/face-landmark/model/landmark_deploy.prototxt', 
                    '../temp/face-landmark/model/VanFace.caffemodel',
                    caffe.TEST)

    start = time.time()
    for num, input_img_filename in enumerate(inputPath.iterdir()):
        logger.info("Start to process No.{} image.".format(num))
        img_name = input_img_filename.name
        logger.info("The name of the image is {}.".format(img_name))
        img = cv2.imread(str(input_img_filename))
        # img = cv2.resize(img, (300, 400))
        RGB_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxs, landmarks = mtcnn_detector.detect_face(RGB_image)  # 检测得到bboxs以及特征点
        img = draw_images(net, img, bboxs, landmarks)  # 得到绘制人脸框及特征点的图片
        savePath = outputPath / img_name  # 图片保存路径
        logger.info("Process complete. Save image to {}.".format(str(savePath)))

        cv2.imwrite(str(savePath), img)  # 保存图片

    logger.info("Finish all the images.")
    logger.info("Elapsed time: {:.3f}s".format(time.time() - start))

import time
import sys
import pathlib
import logging

import cv2
import torch

from tools.test_detect import MtcnnDetector

logger = logging.getLogger("app")
formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
console_handler.formatter = formatter  # 也可以直接给formatter赋值 


def draw_images(img, dets, bboxs):  # 在图片上绘制人脸框及特征点
    num_face = dets.shape[0]
    print("num_face:", num_face)
    for i in range(num_face):
            cv2.rectangle(img, (int(bboxs[i, 0]), int(bboxs[i, 1])), (int(
                bboxs[i, 2]), int(bboxs[i, 3])), (0, 255, 0), 3)
    # for p in landmarks:
    #    for i in range(5):
    #        cv2.circle(img, (int(p[2 * i]), int(p[2 * i + 1])), 3, (0, 0, 255), -1)
    return img


if __name__ == '__main__':
    mtcnn_detector = MtcnnDetector(min_face_size=24, threshold=[0.99, 0.9, 0.9], use_cuda=False)  # 加载模型参数，构造检测器
    mtcnn_detector.pnet_detector.load_state_dict(torch.load('./results/pnet/check_point/98_landmarks_model_050.pth'))
    logger.info("Init the MtcnnDetector.")
    project_root = pathlib.Path()
    # inputPath = project_root / "data" / "test_images"
    inputPath = pathlib.Path('/home/ubuntu/dataset/user/')
    outputPath = project_root / "data" / "you_result" / "pnet"
    outputPath.mkdir(exist_ok=True)

    start = time.time()
    for num, input_img_filename in enumerate(inputPath.iterdir()):
        logger.info("Start to process No.{} image.".format(num))
        img_name = input_img_filename.name
        logger.info("The name of the image is {}.".format(img_name))
        img = cv2.imread(str(input_img_filename))
        RGB_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets, bboxs = mtcnn_detector.detect_pnet(RGB_image)  # 检测得到bboxs以及特征点
        if dets is not None:
            img = draw_images(img, dets, bboxs)  # 得到绘制人脸框及特征点的图片
        savePath = outputPath / img_name  # 图片保存路径
        logger.info("Process complete. Save image to {}.".format(str(savePath)))

        cv2.imwrite(str(savePath), img)  # 保存图片

    logger.info("Finish all the images.")
    logger.info("Elapsed time: {:.3f}s".format(time.time() - start))

# import cv2
#
# imgFile = "./img/lena.jpg"
# img1 = cv2.imread(imgFile, flags=1)  # flags=1 读取彩色图像(BGR)
# xmin, ymin, w, h = 200, 200, 200, 200  # 矩形裁剪区域 (ymin:ymin+h, xmin:xmin+w) 的位置参数
# imgCrop = img1[ymin:ymin + h, xmin:xmin + w].copy()  # 切片获得裁剪后保留的图像区域
# cv2.imshow("CropDemo", imgCrop)  # 在窗口显示 彩色随机图像
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
#
# imgFile = "img/lena.jpg"
# img1 = cv2.imread(imgFile, flags=1)  # flags=1 读取彩色图像(BGR)
# roi = cv2.selectROI(img1, showCrosshair=True, fromCenter=False)
# xmin, ymin, w, h = roi  # 矩形裁剪区域 (ymin:ymin+h, xmin:xmin+w) 的位置参数
# imgROI = img1[ymin:ymin + h, xmin:xmin + w].copy()  # 切片获得裁剪后保留的图像区域
# cv2.imshow("RIODemo", imgROI)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# from PIL import Image
# import matplotlib.pyplot as plt
#
# imgFile = "./img/lena.jpg"
# img = Image.open(imgFile)  # W*H
# plt.rcParams['font.sans-serif'] = ['FangSong']  # 支持中文标签
# plt.subplot(221), plt.title("原图"), plt.axis('off')
# plt.imshow(img)
# img_c = img.crop([img.size[0] / 4, img.size[1] / 4, img.size[0] * 3 / 4, img.size[1] * 3 / 4])
# plt.rcParams['font.sans-serif'] = ['FangSong']  # 支持中文标签
# plt.subplot(222), plt.title("裁切之后"), plt.axis('off')
# plt.imshow(img_c)
# plt.show()


# from PIL import Image
#
# imgFile = "./img/lena.jpg"
# img = Image.open(imgFile)
# size = img.size
# print(size)
# # 准备将图片切割成9张小图片
# weight = int(size[0] // 3)
# height = int(size[1] // 3)
# # 切割后的小图的宽度和高度
# print(weight, height)
# for j in range(3):
#     for i in range(3):
#         box = (weight * i, height * j, weight * (i + 1), height * (j + 1))
#         region = img.crop(box)
#         region.save('{}{}.png'.format(j, i))

#  opencv版本号为 4.10.0
#  安装 hyperlpr库 python -m pip install hyperlpr
#  安装 opencv库   python -m pip install opencv_pythonq
#  复制工程下的hyperlpr_replace.py到包目录下(C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages\hyperlpr)
#  删除原来的hyperlpr.py 然后重新命名hyperlpr_replace.py为hyperlpr.py
#导入包
from hyperlpr import *
#导入OpenCV库
import cv2
#读入图片
image = cv2.imread("2.png")
#识别结果
print(HyperLPR_plate_recognition(image))


# coding:utf-8
# 导入包
from hyperlpr import *
# 导入OpenCV库
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 定义画图函数
def drawRectBox(image, rect, addText, fontC):
    """
    车牌识别，绘制矩形框与结果
    :param image: 原始图像
    :param rect: 矩形框坐标
    :param addText:车牌号
    :param fontC: 字体
    :return:
    """
    # 绘制车牌位置方框
    cv2.rectangle(image, (int(round(rect[0])), int(round(rect[1]))),
                 (int(round(rect[2]) + 15), int(round(rect[3]) + 15)),
                 (0, 0, 255), 2)
    # 绘制字体背景框
    cv2.rectangle(image, (int(rect[0] - 1), int(rect[1]) - 25), (int(rect[0] + 120), int(rect[1])), (0, 0, 255), -1, cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((int(rect[0] + 1), int(rect[1] - 25)), addText, (255, 255, 255), font=fontC)
    imagex = np.array(img)
    return imagex

# 读取选择的图片
image = cv2.imread('2.png')
all_res = HyperLPR_plate_recognition(image)
# 车牌标注的字体
fontC = ImageFont.truetype("Font/msyhl.ttc", 20, 0)
# all_res为多个车牌信息的列表，取第一个车牌信息
lisence, conf, boxes = all_res[0]
image = drawRectBox(image, boxes, lisence, fontC)
cv2.imshow('RecognitionResult', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 读取摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# 车牌标注的字体
fontC = ImageFont.truetype("Font/msyhl.ttc", 20, 0)

while True:
    ref, frame = cap.read()
    if ref:
        # 识别车牌
        all_res = HyperLPR_plate_recognition(frame)
        if len(all_res) > 0:
            lisence, conf, boxes = all_res[0]
            frame = drawRectBox(frame, boxes, lisence, fontC)
        cv2.imshow("RecognitionResult", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break  # 退出
    else:
        break







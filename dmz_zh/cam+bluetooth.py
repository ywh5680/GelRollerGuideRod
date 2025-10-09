import cv2
import numpy as np
import time
import os
import subprocess
import threading
from calibration import calibration
from test_poisson_wukuang import marker_detection,contact_detection,matching_v2,fast_poisson


# ----------------- 参数配置 ------------------
CROP = False
COMPENSATE = False
[upleft_x, upleft_y, downright_x, downright_y] = [1, 1, 1, 1]

ref_img_path = "/home/pi/Desktop/dmz_zh/tong6/0.jpg"
table_path = "/home/pi/Desktop/dmz_zh/table_smooth.npy"

# ----------------- 加载参考数据 ------------------
ref_img = cv2.imread(ref_img_path)
table = np.load(table_path)

# ----------------- 卷积核 ------------------
def make_kernal(radius, shape='circle'):
    k_size = 2 * radius + 1
    kernal = np.zeros((k_size, k_size), dtype=np.uint8)
    for i in range(k_size):
        for j in range(k_size):
            if shape == 'circle':
                if (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
                    kernal[i, j] = 1
            elif shape == 'square':
                kernal[i, j] = 1
    return kernal

kernel1 = make_kernal(3, 'circle')
kernel2 = make_kernal(25, 'circle')

# ----------------- 图像处理函数 ------------------
def Process_single_image_from_array(ref_img, test_img, table2):
    if CROP:
        ref_img = ref_img[upleft_y:downright_y + 1, upleft_x:downright_x + 1]
    if COMPENSATE:
        ref_img = color_mean(ref_img)

    if CROP:
        test_img = test_img[upleft_y:downright_y + 1, upleft_x:downright_x + 1]
    img = test_img.copy()
    if COMPENSATE:
        test_img = color_mean(test_img)

    marker = cali.mask_marker(ref_img)
    keypoints = cali.find_dots(marker)
    marker_mask = cali.make_mask(ref_img, keypoints)
    marker_image = np.dstack((marker_mask, marker_mask, marker_mask))
    ref_img = cv2.inpaint(ref_img, marker_mask, 3, cv2.INPAINT_TELEA)

    red_mask = (ref_img[:, :, 2] > 12).astype(np.uint8)
    ref_blur = cv2.GaussianBlur(ref_img.astype(np.float32), (3, 3), 0) + 1
    blur_inverse = 1 + ((np.mean(ref_blur) / ref_blur) - 1) * 2

    test_img = cv2.GaussianBlur(test_img.astype(np.float32), (3, 3), 0)
    marker_mask = marker_detection(test_img)
    marker_mask = cv2.dilate(marker_mask, kernel1, iterations=1)
    contact_mask = contact_detection(test_img, ref_blur, marker_mask, kernel2)
    grad_img2 = matching_v2(test_img, ref_blur, cali, table2, blur_inverse)
    grad_img2[:, :, 0] = grad_img2[:, :, 0] * (1 - marker_mask) * red_mask
    grad_img2[:, :, 1] = grad_img2[:, :, 1] * (1 - marker_mask) * red_mask
    depth2 = fast_poisson(grad_img2[:, :, 0], grad_img2[:, :, 1])
    depth2[depth2 < 0] = 0

    print("当前帧最大深度值（单位：cm）：", depth2.max())
    ca=depth2.max()-depth2.min()
    print("当前帧最大差值（单位：cm）：", ca)
    if ca-2> 0.5:
        subprocess.call(["espeak", "-v", "zh","-s", "300","前方道路不平整"])

# ----------------- 拍照 + 处理线程 ------------------
def camera_loop():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("无法打开摄像头")
        return
    for i in range(15):
        ret, frame = cap.read()
    while True:
        
        ret, frame = cap.read()
        
        if not ret:
            print("图像采集失败")
            continue

        try:
            Process_single_image_from_array(ref_img, frame, table)
        except Exception as e:
            print("图像处理异常：", e)

        time.sleep(5)  # 每10秒处理一次图像

    cap.release()

# ----------------- 主函数入口 ------------------
if __name__ == '__main__':
    

    cali = calibration()
    camera_thread = threading.Thread(target=camera_loop)
    camera_thread.daemon = True
    camera_thread.start()

    while True:
        time.sleep(1)  # 主线程保持运行

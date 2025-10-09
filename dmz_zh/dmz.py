import RPi.GPIO as GPIO
import time
import subprocess
import serial
import threading
import queue

import os
import sys
import warnings
import numpy as np
from fast_poisson import fast_poisson
import cv2
from calibration import calibration

# ================= GPIO 设置 =================
SENSORS = [
    {'name': '前方', 'trig': 17, 'echo': 18},
    {'name': '左侧', 'trig': 22, 'echo': 23},
    {'name': '右侧', 'trig': 24, 'echo': 25}
]
ENA = 12  # 马达控制引脚

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

for s in SENSORS:
    GPIO.setup(s['trig'], GPIO.OUT)
    GPIO.setup(s['echo'], GPIO.IN)
    GPIO.output(s['trig'], GPIO.LOW)

GPIO.setup(ENA, GPIO.OUT)
motor_pwm = GPIO.PWM(ENA, 50)
motor_pwm.start(0)

# ================= 状态变量 =================
OBSTACLE_THRESHOLD = 0.1   #cm
alert_interval = 3
last_alert_time = [0, 0, 0]
spoken_state = set()
last_height = None

# ================= 播报队列 =================
speak_queue = queue.Queue()



sys.path.append('/Code/Calib')



def matching_v2(test_img, ref_blur, cali, table, blur_inverse):
    diff_temp1 = test_img - ref_blur
    diff_temp2 = diff_temp1 * blur_inverse
    diff_temp2[:, :, 0] = (diff_temp2[:, :, 0] -
                           cali.zeropoint[0]) / cali.lookscale[0]
    diff_temp2[:, :, 1] = (diff_temp2[:, :, 1] -
                           cali.zeropoint[1]) / cali.lookscale[1]
    diff_temp2[:, :, 2] = (diff_temp2[:, :, 2] -
                           cali.zeropoint[2]) / cali.lookscale[2]
    diff_temp3 = np.clip(diff_temp2, 0, 0.999)
    diff = (diff_temp3 * cali.bin_num).astype(int)
    grad_img = table[diff[:, :, 0], diff[:, :, 1], diff[:, :, 2], :]
    return grad_img



def contact_detection(raw_image, ref_blur, marker_mask, kernel):
    diff_img = np.max(np.abs(raw_image.astype(np.float32) - ref_blur), axis=2)
    contact_mask = (diff_img > 25).astype(np.uint8)  # *(1-marker_mask)
    contact_mask = cv2.dilate(contact_mask, kernel, iterations=1)
    contact_mask = cv2.erode(contact_mask, kernel, iterations=1)
    return contact_mask


def marker_detection(raw_image_blur):
    m, n = raw_image_blur.shape[1], raw_image_blur.shape[0]
    raw_image_blur = cv2.pyrDown(raw_image_blur).astype(np.float32)
    ref_blur = cv2.GaussianBlur(raw_image_blur, (25, 25), 0)
    diff = ref_blur - raw_image_blur
    diff *= 16.0
    diff[diff < 0.] = 0.
    diff[diff > 255.] = 255.
    mask_b = diff[:, :, 0] > 150
    mask_g = diff[:, :, 1] > 150
    mask_r = diff[:, :, 2] > 150
    mask = (mask_b * mask_g + mask_b * mask_r + mask_g * mask_r) > 0
    mask = cv2.resize(mask.astype(np.uint8), (m, n))
    return mask


def make_kernal(n, k_type):
    if k_type == 'circle':
        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    else:
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))
    return kernal


def Process_single_image(ref_img_path, tag_img_path, table2):
    
    ref_img = cv2.imread(ref_img_path)
    if CROP:
        ref_img = ref_img[upleft_y:downright_y + 1, upleft_x:downright_x + 1]
    if COMPENSATE:
        ref_img = color_mean(ref_img)
    

    test_img = cv2.imread(tag_img_path)
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
   
    print('imge:',depth2.max())
 




def speak_worker():
    while True:
        text = speak_queue.get()
        if text is None:
            break
        subprocess.call(['espeak', '-v', 'zh', '-s', '300', text])
        speak_queue.task_done()

def speak_async(text):
    speak_queue.put(text)

def vibrate_motor(duration=0.3, strength=60):
    motor_pwm.ChangeDutyCycle(strength)
    time.sleep(duration)
    motor_pwm.ChangeDutyCycle(0)


# ================= 图像处理线程 =================
def image_processing_thread_loop():
    while True:
        try:
            # 模拟你定时或定条件处理图片
            Process_single_image(
                '/home/pi/Desktop/dmz_zh/tong6/0.jpg',
                '/home/pi/Desktop/dmz_zh/tong6/test/14.jpg',
                table
            )
        except Exception as e:
            print("图像处理出错：", e)
        time.sleep(2)  # 控制频率，避免资源占用过高



# ================= 超声波检测 =================
def measure_distance(trig, echo):
    GPIO.output(trig, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(trig, GPIO.LOW)

    timeout = time.time() + 0.04
    pulse_start, pulse_end = time.time(), time.time()

    while GPIO.input(echo) == 0 and time.time() < timeout:
        pulse_start = time.time()
    while GPIO.input(echo) == 1 and time.time() < timeout:
        pulse_end = time.time()

    duration = pulse_end - pulse_start
    return (duration * 34300) / 2 / 100

def ultrasonic_thread_loop():
    global last_alert_time
    while True:
        now = time.time()
        for i, s in enumerate(SENSORS):
            dist = measure_distance(s['trig'], s['echo'])
            if dist < OBSTACLE_THRESHOLD:
                if now - last_alert_time[i] > alert_interval:
                    speak_async(f"{s['name']}有障碍物，距离约{int(dist * 100)}厘米")
                    vibrate_motor()
                    last_alert_time[i] = now
        time.sleep(0.2)

# ================= IMU 数据处理 =================
def parse_angle_from_frame(data):
    if len(data) != 10 or data[0] != 0x55 or data[1] != 0x53:
        return None
    roll = ((data[3] << 8) | data[2]) / 32768 * 180
    pitch = ((data[5] << 8) | data[4]) / 32768 * 180
    yaw = ((data[7] << 8) | data[6]) / 32768 * 180
    return roll, pitch, yaw

def parse_height_from_pressure_frame(data):
    if len(data) != 10 or data[0] != 0x55 or data[1] != 0x56:
        return None
    pressure = (data[5] << 24) | (data[4] << 16) | (data[3] << 8) | data[2]
    height_cm = (data[9] << 24) | (data[8] << 16) | (data[7] << 8) | data[6]
    return pressure, height_cm / 100.0

def detect_slope(pitch):
    print("角度：",pitch)
    if pitch > 180:
        pitch -= 360
    if pitch > 5:
        return "上坡"
    elif pitch < -5:
        return "下坡"
    else:
        return "平路"



def detect_step(current_height):
    global last_height
    if last_height is None:
        last_height = current_height
        return None
    delta = current_height - last_height
    last_height = current_height
    if abs(delta) > 0.1:
        return "上台阶" if delta > 0 else "下台阶"
    return None

# ================= 主程序 =================

def main():
    try:
        ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=0.5)
        print("系统启动中...")

        # 启动语音线程
        threading.Thread(target=speak_worker, daemon=True).start()

        # 启动超声波线程
        threading.Thread(target=ultrasonic_thread_loop, daemon=True).start()

        # ✅ 启动图像处理线程
        threading.Thread(target=image_processing_thread_loop, daemon=True).start()

        # 主线程读取 IMU 数据
        while True:
            if ser.in_waiting:
                head = ser.read(1)
                if head != b'\x55':
                    continue
                type_byte = ser.read(1)
                if not type_byte:
                    continue
                data = ser.read(8)
                frame = head + type_byte + data
                frame_type = type_byte[0]

                if frame_type == 0x53:
                    result = parse_angle_from_frame(frame)
                    if result:
                        roll, pitch, yaw = result
                        slope = detect_slope(pitch)
                        if slope != "平路" and slope not in spoken_state:
                            speak_async(f"检测到{slope}")
                            spoken_state.clear()
                            spoken_state.add(slope)

                elif frame_type == 0x56:
                    result = parse_height_from_pressure_frame(frame)
                    if result:
                        _, height = result
                        step_state = detect_step(height)
                        if step_state and step_state not in spoken_state:
                            speak_async(f"检测到{step_state}")
                            spoken_state.clear()
                            spoken_state.add(step_state)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("程序已终止。")
    finally:
        speak_queue.put(None)  # 停止语音线程
        ser.close()
        motor_pwm.stop()
        GPIO.cleanup()

# ================ 程序入口 ================
if __name__ == '__main__':
    CROP = False
    batch_mode = True
    COMPENSATE = False
    cali = calibration()
    kernel1 = make_kernal(3, 'circle')
    kernel2 = make_kernal(25, 'circle')
    [upleft_x, upleft_y, downright_x, downright_y] = [1, 1, 1, 1]
    table = np.load('/home/pi/Desktop/dmz_zh/table_smooth.npy')
    main()

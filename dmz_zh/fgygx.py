import cv2
import numpy as np
import time
import os
import subprocess
import threading
import serial
import queue
import atexit
import sys
from dataclasses import dataclass
from collections import deque
from calibration import calibration
from fast_poisson import fast_poisson
import RPi.GPIO as GPIO

# =============== 配置 ===============
TEMP_REF_PATH = "/home/pi/Desktop/ref.jpg"
TABLE_PATH = "/home/pi/Desktop/dmz_zh/tong8/table_smooth.npy"
CROP = False
COMPENSATE = False
SAMPLE_WINDOW = 20
HEIGHT_WINDOW = 40
ANGLE_THRESH = 4.0
HEIGHT_RATE_THRESH = 0.05
HEIGHT_STEP_THRESH = 0.12
OBSTACLE_THRESHOLD = 0.1
ALERT_INTERVAL = 3


kernel1 = np.ones((7, 7), np.uint8)
kernel2 = np.ones((25, 25), np.uint8)
table = np.load(TABLE_PATH)

# =============== GPIO 硬件 ===============
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)


@dataclass
class UltrasonicSensor:
    name: str
    trig: int
    echo: int
    key: str


class HardwareConfig:
    SENSORS = [
        UltrasonicSensor("前方", 17, 18, "front"),
        UltrasonicSensor("左侧", 22, 23, "left"),
        UltrasonicSensor("右侧", 24, 25, "right")
    ]
    MOTOR_PIN = 12

    @classmethod
    def setup(cls):
        for s in cls.SENSORS:
            GPIO.setup(s.trig, GPIO.OUT)
            GPIO.setup(s.echo, GPIO.IN)
            GPIO.output(s.trig, GPIO.LOW)
        GPIO.setup(cls.MOTOR_PIN, GPIO.OUT)
        cls.motor_pwm = GPIO.PWM(cls.MOTOR_PIN, 50)
        cls.motor_pwm.start(0)
        print("硬件初始化完成")


# =============== 语音系统 ===============
class VoiceSystem:
    PRIORITY_LEVELS = {
        'emergency': {'speed': 400, 'vibrate': 80},
        'warning': {'speed': 300, 'vibrate': 60},
        'info': {'speed': 200, 'vibrate': 40}
    }

    def __init__(self):
        self.queue = queue.PriorityQueue()
        self.active_messages = set()
        threading.Thread(target=self._speak_worker, daemon=True).start()

    def add_message(self, text, priority='info'):
        if text not in self.active_messages:
            self.queue.put((
                self.PRIORITY_LEVELS[priority]['speed'],
                time.time(),
                text,
                self.PRIORITY_LEVELS[priority]['vibrate']
            ))
            self.active_messages.add(text)

    def _speak_worker(self):
        while True:
            speed, _, text, vibrate = self.queue.get()
            try:
                if vibrate > 0:
                    HardwareConfig.motor_pwm.ChangeDutyCycle(vibrate)
                    time.sleep(0.3)
                    HardwareConfig.motor_pwm.ChangeDutyCycle(0)
                subprocess.run(['espeak', '-v', 'zh', '-s', str(speed), text])
                self.active_messages.discard(text)
            except Exception as e:
                print("[语音系统] 播报失败:", e)
            finally:
                self.queue.task_done()


# =============== 图像处理 ===============
def marker_detection(raw_image_blur):
    m, n = raw_image_blur.shape[1], raw_image_blur.shape[0]
    raw_image_blur = cv2.pyrDown(raw_image_blur).astype(np.float32)
    ref_blur = cv2.GaussianBlur(raw_image_blur, (25, 25), 0)
    diff = (ref_blur - raw_image_blur) * 16.0
    diff = np.clip(diff, 0, 255)
    mask = np.any(diff > 150, axis=2).astype(np.uint8)
    return cv2.resize(mask, (m, n))


def contact_detection(raw_image, ref_blur, marker_mask, kernel):
    diff_img = np.max(np.abs(raw_image.astype(np.float32) - ref_blur), axis=2)
    contact_mask = (diff_img > 25).astype(np.uint8)
    contact_mask = cv2.dilate(contact_mask, kernel, iterations=1)
    contact_mask = cv2.erode(contact_mask, kernel, iterations=1)
    return contact_mask


def matching_v2(test_img, ref_blur, cali, table, blur_inverse):
    diff_temp = (test_img - ref_blur) * blur_inverse
    for i in range(3):
        diff_temp[:, :, i] = (diff_temp[:, :, i] - cali.zeropoint[i]) / cali.lookscale[i]
    diff_temp = np.clip(diff_temp, 0, 0.999)
    diff = (diff_temp * cali.bin_num).astype(int)
    return table[diff[:, :, 0], diff[:, :, 1], diff[:, :, 2], :]


def process_camera(cali, msg_system):
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("正在拍摄参考图像...")
    ret, ref_img = cap.read()
    if not ret:
        print("参考图像拍摄失败")
        return
    cv2.imwrite(TEMP_REF_PATH, ref_img)
    print("参考图像保存完成，按 q 退出")

    ref_blur = cv2.GaussianBlur(ref_img.astype(np.float32), (3, 3), 0) + 1
    blur_inverse = 1 + ((np.mean(ref_blur) / ref_blur) - 1) * 2

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            test_img = cv2.GaussianBlur(frame.astype(np.float32), (3, 3), 0)
            marker_mask = marker_detection(test_img)
            marker_mask = cv2.dilate(marker_mask, kernel1, iterations=1)
            contact_mask = contact_detection(test_img, ref_blur, marker_mask, kernel2)
            grad_img = matching_v2(test_img, ref_blur, cali, table, blur_inverse)
            red_mask = (ref_img[:, :, 2] > 12).astype(np.uint8)
            for i in range(2):
                grad_img[:, :, i] *= (1 - marker_mask) * red_mask
            depth = fast_poisson(grad_img[:, :, 0], grad_img[:, :, 1])
            print(depth.max())
            if depth.max() - 2.5 > 0.29:
                msg_system.add_message("前方道路不平整", priority='warning')
        except Exception as e:
            print("图像处理异常：", e)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if os.path.exists(TEMP_REF_PATH):
                os.remove(TEMP_REF_PATH)
            break
    cap.release()
    cv2.destroyAllWindows()


# =============== IMU 处理 ===============
gyro_step_state = {}
def parse_angle_frame(frame):
    pitch = ((frame[5] << 8) | frame[4]) / 32768 * 180
    if pitch > 180:
        pitch -= 360
    return pitch


def parse_height_frame(frame):
    h_cm = (frame[9] << 24) | (frame[8] << 16) | (frame[7] << 8) | frame[6]
    return h_cm / 89.0

# 角速度
def parse_gyro_frame(frame):
    """
    解析陀螺仪数据帧 (类型0x52)
    返回: (wx, wy, wz) 单位: °/s
    
    参数:
        frame: 10字节数据帧 (包含帧头0x55和类型0x52)
    """
    wx = ((frame[3] << 8) | frame[2]) / 32768 * 2000
    wy = ((frame[5] << 8) | frame[4]) / 32768 * 2000
    wz = ((frame[7] << 8) | frame[6]) / 32768 * 2000
    
    # 处理溢出情况
    if wx > 2000:
        wx -= 4000
    if wy > 2000:
        wy -= 4000
    if wz > 2000:
        wz -= 4000
        
    return wx

def detect_step_by_gyro(wx):
    """
    改进版：只检测单向峰值（不要求反转），支持 1 秒冷却防连发
    """
    global gyro_step_state

    # 初始化状态机
    if 'phase' not in gyro_step_state:
        gyro_step_state['phase'] = 'IDLE'       # 当前状态：IDLE/POSITIVE/NEGATIVE
        gyro_step_state['count'] = 0            # 连续达标帧数计数
        gyro_step_state['threshold'] = 20       # 角速度阈值(°/s)
        gyro_step_state['required_count'] = 3   # 触发所需连续帧数
        gyro_step_state['cooldown'] = False     # 冷却状态标志
        gyro_step_state['cooldown_start'] = 0   # 冷却开始时间
        gyro_step_state['cooldown_time'] = 1.0  # 冷却时长(秒)

    # 冷却状态，直接跳过
    if gyro_step_state['cooldown']:
        if time.time() - gyro_step_state['cooldown_start'] >= gyro_step_state['cooldown_time']:
            gyro_step_state['cooldown'] = False  # 冷却结束
            print("[STEP] 冷却结束，可以继续检测。")
        else:
            return None

    phase = gyro_step_state['phase']
    threshold = gyro_step_state['threshold']
    required_count = gyro_step_state['required_count']

    # 正向检测（上台阶）
    if wx > threshold:
        if phase == 'IDLE' or phase == 'POSITIVE':
            gyro_step_state['phase'] = 'POSITIVE'
            gyro_step_state['count'] += 1
            if gyro_step_state['count'] >= required_count:
                print(f"[STEP] 检测到：下台阶（峰值检测，wx={wx:.2f} °/s）")
                gyro_step_state['phase'] = 'IDLE'
                gyro_step_state['count'] = 0
                gyro_step_state['cooldown'] = True
                gyro_step_state['cooldown_start'] = time.time()
                return "up"
    # 负向检测（下台阶）
    elif wx < -threshold:
        if phase == 'IDLE' or phase == 'NEGATIVE':
            gyro_step_state['phase'] = 'NEGATIVE'
            gyro_step_state['count'] += 1
            if gyro_step_state['count'] >= required_count:
                print(f"[STEP] 检测到：上台阶（峰值检测，wx={wx:.2f} °/s）")
                gyro_step_state['phase'] = 'IDLE'
                gyro_step_state['count'] = 0
                gyro_step_state['cooldown'] = True
                gyro_step_state['cooldown_start'] = time.time()
                return "down"
    else:
        # 小角速度时重置
        gyro_step_state['phase'] = 'IDLE'
        gyro_step_state['count'] = 0

    return None

def process_imu(msg_system):
    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=0.5)
    window = deque(maxlen=SAMPLE_WINDOW)
    height_window = deque(maxlen=HEIGHT_WINDOW)
    last_pitch, last_height = None, None
    angle_change = 0
    height_jump = 0
    wx = 0
    last_event_time = 0
    EVENT_GAP = 1.0  # 秒
    now = time.time()
    spoken_state = set()
    while True:
        try:
            if ser.in_waiting >= 10:
                if ser.read(1) != b'\x55':
                    continue
                t = ser.read(1)[0]
                data = ser.read(8)
                frame = b'\x55' + bytes([t]) + data
                now = time.time()

                if t == 0x53:
                    last_pitch = parse_angle_frame(frame)
                elif t == 0x56:
                    last_height = parse_height_frame(frame)
                elif t == 0x52:
                    wx = parse_gyro_frame(frame)

                if last_pitch is not None and last_height is not None:
                    window.append((now, last_pitch))
                    height_window.append((now,last_height))

            if len(window) < 2 or len(height_window) < 2:
                time.sleep(0.005)
                continue

            times = [x[0] for x in window]
            pitches = [x[1] for x in window]
            angle_change = max(pitches) - min(pitches)


            height_times = [x[0] for x in height_window]
            heights = [x[1] for x in height_window]
            duration = height_times[-1] - height_times[0] or 1e-6
            height_rate = abs(heights[-1] - heights[0]) / duration
            height_jump = abs(heights[-1] - heights[0])
            height_diff = heights[-1] - heights[0]

                      # 状态检测
            current_time = time.time()
            state = None

            # 台阶检测（优先级高）
            if height_jump > HEIGHT_STEP_THRESH:
                step_result = detect_step_by_gyro(wx)
                # 确保高度变化方向与陀螺仪检测一致
                if (height_diff > 0 and step_result == "up") or (height_diff < 0 and step_result == "down"):
                    state = "上台阶" if height_diff > 0 else "下台阶"
            
            # 坡道检测（优先级低）
            elif angle_change > ANGLE_THRESH and height_rate < HEIGHT_RATE_THRESH:
                state = "下坡" if pitches[-1] > pitches[0] else "上坡"

            # 事件触发（带防抖）
            if state and (current_time - last_event_time > EVENT_GAP):
                msg_system.add_message(f"检测到{state}", priority='info')
                last_event_time = current_time

            time.sleep(0.005)
            
        except Exception as e:
            print("[IMU] 异常:", e)
            time.sleep(1)





# =============== 超声波服务 ===============
def process_ultrasound(msg_system):
    last_alert_time = 0
    while True:
        try:
            now = time.time()
            obstacles = []
            for s in HardwareConfig.SENSORS:
                GPIO.output(s.trig, GPIO.HIGH)
                time.sleep(0.00001)
                GPIO.output(s.trig, GPIO.LOW)
                timeout = time.time() + 0.04
                while GPIO.input(s.echo) == 0 and time.time() < timeout:
                    pulse_start = time.time()
                while GPIO.input(s.echo) == 1 and time.time() < timeout:
                    pulse_end = time.time()
                duration = pulse_end - pulse_start
                dist = (duration * 34300) / 2 / 100
                if dist < OBSTACLE_THRESHOLD:
                    obstacles.append(s.name)
            if obstacles and now - last_alert_time > ALERT_INTERVAL:
                msg = "和".join(obstacles) + "有障碍物"
                msg_system.add_message(msg, priority='emergency')
                last_alert_time = now
            time.sleep(0.1)
        except Exception as e:
            print("[超声波] 异常：", e)
            time.sleep(1)


# =============== 主函数入口 ===============
if __name__ == '__main__':
    atexit.register(GPIO.cleanup)
    HardwareConfig.setup()
    msg_system = VoiceSystem()
    cali = calibration()

    threads = [
        threading.Thread(target=process_imu, args=(msg_system,), daemon=True),
        #threading.Thread(target=process_ultrasound, args=(msg_system,), daemon=True),
        #threading.Thread(target=process_camera, args=(cali, msg_system), daemon=True),
    ]

    for t in threads:
        t.start()

    print("系统启动完成")
    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("系统已退出")

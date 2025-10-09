#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import RPi.GPIO as GPIO
import time
import serial
import threading
import queue
import numpy as np
from collections import deque, OrderedDict
from dataclasses import dataclass
import subprocess
import atexit
import sys

# ===== 硬件配置 =====
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

@dataclass
class UltrasonicSensor:
    name: str
    trig: int
    echo: int
    key: str

class HardwareConfig:
    # 超声波传感器配置
    SENSORS = [
        UltrasonicSensor("前方", 17, 18, "front"),
        UltrasonicSensor("左侧", 22, 23, "left"),
        UltrasonicSensor("右侧", 24, 25, "right")
    ]
    
    # 电机震动引脚
    MOTOR_PIN = 12
    
    # 串口配置
    SERIAL_PORT = '/dev/ttyUSB0'
    SERIAL_BAUDRATE = 9600
    
    @classmethod
    def setup(cls):
        """初始化所有硬件"""
        try:
            # 设置超声波引脚
            for s in cls.SENSORS:
                GPIO.setup(s.trig, GPIO.OUT)
                GPIO.setup(s.echo, GPIO.IN)
                GPIO.output(s.trig, GPIO.LOW)
            
            # 设置电机PWM
            GPIO.setup(cls.MOTOR_PIN, GPIO.OUT)
            cls.motor_pwm = GPIO.PWM(cls.MOTOR_PIN, 50)
            cls.motor_pwm.start(0)
            
            print("硬件初始化完成")
        except Exception as e:
            print(f"硬件初始化失败: {e}")
            sys.exit(1)

# ===== IMU校准系统 =====
class IMUCalibrator:
    def __init__(self, calibration_samples=10):
        self.calibration_samples = calibration_samples
        self.reference_pitch = None
        self.reference_roll = None
    
    def calibrate(self, ser):
        """自动校准初始角度"""
        print("\n[IMU校准] 请将设备水平放置并保持静止...")
        pitch_samples = []
        roll_samples = []
        start_time = time.time()
        
        while len(pitch_samples) < self.calibration_samples:
            if ser.in_waiting >= 10:
                data = ser.read(10)
                if data[0] == 0x55 and data[1] == 0x53:  # 角度数据帧
                    pitch = ((data[5]<<8)|data[4])/32768 * 180
                    roll = ((data[3]<<8)|data[2])/32768 * 180
                    
                    # 实时显示校准进度
                    progress = len(pitch_samples)/self.calibration_samples*100
                    print(f"\r[IMU校准] 进度: {progress:.1f}% | 当前俯仰角: {pitch:.2f}°", end='')
                    
                    pitch_samples.append(pitch)
                    roll_samples.append(roll)
                time.sleep(0.01)
        
        # 计算初始角度基准（去除异常值）
        self.reference_pitch = self._get_robust_mean(pitch_samples)
        self.reference_roll = self._get_robust_mean(roll_samples)
        
        print(f"\n[IMU校准] 完成! 基准俯仰角: {self.reference_pitch:.2f}° 基准横滚角: {self.reference_roll:.2f}°")
    
    def _get_robust_mean(self, data):
        """使用IQR方法去除异常值后计算均值"""
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5*iqr
        upper_bound = q3 + 1.5*iqr
        filtered = [x for x in data if lower_bound <= x <= upper_bound]
        return np.mean(filtered)
    
    def get_relative_angle(self, raw_pitch, raw_roll):
        """计算相对于初始角度的变化量"""
        if self.reference_pitch is None:
            return raw_pitch, raw_roll
        
        # 计算相对角度并归一化到[-180,180]
        rel_pitch = (raw_pitch - self.reference_pitch + 180) % 360 - 180
        rel_roll = (raw_roll - self.reference_roll + 180) % 360 - 180
        
        return rel_pitch, rel_roll

# ===== 数据处理模块 =====
class IMUParser:
    @staticmethod
    def parse(data):
        """解析IMU数据包"""
        if len(data) != 10 or data[0] != 0x55:
            return None
        
        if data[1] == 0x53:  # 角度数据
            return {
                'type': 'angle',
                'raw_pitch': ((data[5]<<8)|data[4])/32768 * 180,
                'raw_roll': ((data[3]<<8)|data[2])/32768 * 180,
                'timestamp': time.time()
            }
        elif data[1] == 0x56:  # 高度数据
            height = ((data[9]<<24)|(data[8]<<16)|(data[7]<<8)|data[6])/100.0
            return {
                'type': 'height',
                'height': height if 0 <= height <= 10 else None,  # 物理范围检查
                'timestamp': time.time()
            }
        return None

# ===== 台阶检测系统 =====
class StepDetector:
    def __init__(self):
        self.last_height = None
        self.last_pitch = None
        self.min_step_height = 0.05  # 5cm
        self.min_angle_change = 3.0  # 2度
        self.last_detection_time = 0
        self.detection_interval = 1.0  # 检测间隔(秒)
    
    def detect(self, current_height, current_pitch):
        """检测台阶，需要同时满足高度和角度变化条件"""
        now = time.time()
        if now - self.last_detection_time < self.detection_interval:
            return None
        
        if self.last_height is None or self.last_pitch is None:
            self.last_height = current_height
            self.last_pitch = current_pitch
            return None
        
        height_diff = current_height - self.last_height
        pitch_diff = current_pitch - self.last_pitch
        
        # 更新上次记录
        self.last_height = current_height
        self.last_pitch = current_pitch
        
        # 上台阶条件：高度增加>5cm且角度减小(负值)
        if height_diff > self.min_step_height and pitch_diff < self.min_angle_change:
            self.last_detection_time = now
            return 'up'
        
        # 下台阶条件：高度减少>5cm且角度增加(正值)
        elif height_diff < -self.min_step_height and pitch_diff > self.min_angle_change:
            self.last_detection_time = now
            return 'down'
        
        return None

# ===== 坡度检测系统 =====
class SlopeDetector:
    def __init__(self, confirm_frames=5, slope_threshold=5):
        self.confirm_frames = confirm_frames
        self.slope_threshold = slope_threshold
        self.pitch_history = deque(maxlen=confirm_frames)
        self.calibrator = IMUCalibrator()
        self.last_alert_time = 0
        self.alert_interval = 2  # 警报间隔(秒)
    
    def calibrate(self, ser):
        """执行IMU校准"""
        self.calibrator.calibrate(ser)
    
    def detect(self, raw_pitch, raw_roll):
        """检测有效坡度变化"""
        if raw_pitch is None:
            return None
        
        # 计算相对于校准角度的变化
        rel_pitch, _ = self.calibrator.get_relative_angle(raw_pitch, raw_roll)
        self.pitch_history.append(rel_pitch)
        
        # 需要连续多帧确认
        if len(self.pitch_history) == self.confirm_frames:
            avg_pitch = np.mean(self.pitch_history)
            
            # 检查警报间隔
            now = time.time()
            if abs(avg_pitch) > self.slope_threshold and now - self.last_alert_time > self.alert_interval:
                self.last_alert_time = now
                return 'up' if avg_pitch > 0 else 'down'
        return None

# ===== 语音消息系统 =====
class VoiceSystem:
    PRIORITY_LEVELS = {
        'emergency': {'speed': 400, 'vibrate': 80},  # 障碍物
        'warning': {'speed': 300, 'vibrate': 60},    # 坡度/台阶
        'info': {'speed': 200, 'vibrate': 0}         # 系统消息
    }
    
    def __init__(self):
        self.queue = queue.PriorityQueue()
        self.active_messages = set()  # 防止重复消息
        self.worker = threading.Thread(target=self._speak_worker, daemon=True)
        self.worker.start()
    
    def add_message(self, text, priority='info'):
        """添加消息到队列"""
        if text not in self.active_messages:
            self.queue.put((
                self.PRIORITY_LEVELS[priority]['speed'],
                time.time(),
                text,
                self.PRIORITY_LEVELS[priority]['vibrate']
            ))
            self.active_messages.add(text)
    
    def _speak_worker(self):
        """语音工作线程"""
        while True:
            speed, _, text, vibrate = self.queue.get()
            try:
                # 震动反馈
                if vibrate > 0:
                    HardwareConfig.motor_pwm.ChangeDutyCycle(vibrate)
                    time.sleep(0.3)
                    HardwareConfig.motor_pwm.ChangeDutyCycle(0)
                
                # 语音播报
                subprocess.run(
                    ['espeak', '-v', 'zh', '-s', str(speed), text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )
                
                # 从活跃消息中移除
                if text in self.active_messages:
                    self.active_messages.remove(text)
                    
            except Exception as e:
                print(f"[语音系统] 播报失败: {e}")
            finally:
                self.queue.task_done()

# ===== 超声波服务 =====
class UltrasonicService:
    OBSTACLE_THRESHOLD = 0.01  # 10cm
    ALERT_INTERVAL = 3        # 警报间隔(秒)
    
    @staticmethod
    def measure_distance(trig, echo):
        """测量距离(cm)"""
        try:
            GPIO.output(trig, GPIO.HIGH)
            time.sleep(0.00001)
            GPIO.output(trig, GPIO.LOW)

            timeout = time.time() + 0.04
            while GPIO.input(echo) == 0 and time.time() < timeout:
                pulse_start = time.time()
            while GPIO.input(echo) == 1 and time.time() < timeout:
                pulse_end = time.time()

            duration = pulse_end - pulse_start
            return (duration * 34300) / 2 / 100
        except Exception as e:
            print(f"[超声波] 测量失败: {e}")
            return float('inf')
    
    @classmethod
    def run(cls, msg_system):
        """超声波监测服务"""
        GPIO.setmode(GPIO.BCM)  # 确保线程中模式正确
        last_alert_time = 0
        
        while True:
            try:
                now = time.time()
                active_obstacles = []
                
                # 检测所有传感器
                for s in HardwareConfig.SENSORS:
                    dist = cls.measure_distance(s.trig, s.echo)
                    if dist < cls.OBSTACLE_THRESHOLD:
                        active_obstacles.append(s.key)
                
                # 处理障碍物警报
                if active_obstacles and now - last_alert_time > cls.ALERT_INTERVAL:
                    # 生成自然语言描述
                    if len(active_obstacles) > 1:
                        names = [s.name for s in HardwareConfig.SENSORS 
                                if s.key in active_obstacles]
                        message = "同时检测到" + "和".join(names) + "有障碍物"
                    else:
                        sensor = next(s for s in HardwareConfig.SENSORS 
                                    if s.key in active_obstacles)
                        message = f"{sensor.name}有障碍物"
                    
                    # 发送紧急消息
                    msg_system.add_message(message, priority='emergency')
                    last_alert_time = now
                
                time.sleep(0.1)
            except Exception as e:
                print(f"[超声波] 服务异常: {e}")
                time.sleep(1)

# ===== 主导航系统 =====
class NavigationSystem:
    def __init__(self):
        # 初始化硬件
        HardwareConfig.setup()
        
        # 初始化消息系统
        self.msg_system = VoiceSystem()
        
        # 初始化IMU串口
        try:
            self.ser = serial.Serial(
                HardwareConfig.SERIAL_PORT,
                HardwareConfig.SERIAL_BAUDRATE,
                timeout=0.5
            )
            print("[系统] IMU串口已连接")
        except Exception as e:
            print(f"[系统] IMU串口连接失败: {e}")
            sys.exit(1)
        
        # 初始化各检测模块
        self.slope_detector = SlopeDetector()
        self.step_detector = StepDetector()
        
        # 启动时自动校准IMU
        self.slope_detector.calibrate(self.ser)
        
        # 启动服务线程
        self.threads = [
            threading.Thread(target=self._ultrasonic_service, daemon=True),
            threading.Thread(target=self._imu_service, daemon=True),
            threading.Thread(target=self._status_monitor, daemon=True)
        ]
        
        for t in self.threads:
            t.start()
        
        # 系统启动提示
        self.msg_system.add_message("导航系统已启动", priority='info')
        print("[系统] 所有服务已启动")

    def _ultrasonic_service(self):
        """超声波避障服务"""
        UltrasonicService.run(self.msg_system)
    
    def _imu_service(self):
        """IMU数据处理服务"""
        current_height = None
        current_pitch = None
        
        while True:
            try:
                if self.ser.in_waiting >= 10:
                    data = self.ser.read(10)
                    imu_data = IMUParser.parse(data)
                    
                    if imu_data:
                        # 处理高度数据
                        if imu_data['type'] == 'height' and imu_data['height'] is not None:
                            current_height = imu_data['height']
                            
                            # 检测台阶（需要同时有高度和角度数据）
                            if current_pitch is not None:
                                step = self.step_detector.detect(current_height, current_pitch)
                                if step:
                                    self.msg_system.add_message(
                                        f"检测到{'上' if step=='up' else '下'}台阶", 
                                        priority='warning'
                                    )
                        
                        # 处理角度数据
                        elif imu_data['type'] == 'angle':
                            current_pitch = imu_data['raw_pitch']
                            
                            # 检测坡度（仅使用角度数据）
                            slope = self.slope_detector.detect(
                                imu_data['raw_pitch'],
                                imu_data['raw_roll']
                            )
                            if slope:
                                self.msg_system.add_message(
                                    f"检测到{'上' if slope=='up' else '下'}坡", 
                                    priority='warning'
                                )
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"[IMU服务] 异常: {e}")
                time.sleep(1)
    
    def _status_monitor(self):
        """系统状态监控"""
        while True:
            time.sleep(5)
            # 这里可以添加系统健康检查逻辑
            # 例如：检查线程状态、传感器响应等

# ===== 主程序 =====
if __name__ == '__main__':
    print(""" 
   _____ 导航辅助系统 _____

""")
    
    # 注册退出清理函数
    def cleanup():
        print("\n[系统] 正在清理资源...")
        GPIO.cleanup()
        print("[系统] 资源已释放")
    
    atexit.register(cleanup)
    
    # 启动系统
    nav = NavigationSystem()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[系统] 正在关闭...")
    finally:
        sys.exit(0)
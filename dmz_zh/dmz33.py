#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import RPi.GPIO as GPIO
import time
import serial
import threading
import queue
import numpy as np
from collections import deque
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
    SENSORS = [
        UltrasonicSensor("前方", 17, 18, "front"),
        UltrasonicSensor("左侧", 22, 23, "left"),
        UltrasonicSensor("右侧", 24, 25, "right")
    ]
    
    MOTOR_PIN = 12
    SERIAL_PORT = '/dev/ttyUSB0'
    SERIAL_BAUDRATE = 9600
    
    @classmethod
    def setup(cls):
        try:
            for s in cls.SENSORS:
                GPIO.setup(s.trig, GPIO.OUT)
                GPIO.setup(s.echo, GPIO.IN)
                GPIO.output(s.trig, GPIO.LOW)
            
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
        print("\n[IMU校准] 请将设备水平放置并保持静止...")
        pitch_samples = []
        roll_samples = []
        
        while len(pitch_samples) < self.calibration_samples:
            if ser.in_waiting >= 10:
                data = ser.read(10)
                if data[0] == 0x55 and data[1] == 0x53:  # 角度数据帧
                    pitch = ((data[5]<<8)|data[4])/32768 * 180
                    roll = ((data[3]<<8)|data[2])/32768 * 180
                    pitch_samples.append(pitch)
                    roll_samples.append(roll)
                    print(f"\r[IMU校准] 进度: {len(pitch_samples)/self.calibration_samples*100:.1f}%", end='')
                time.sleep(0.01)
        
        self.reference_pitch = np.mean(pitch_samples)
        self.reference_roll = np.mean(roll_samples)
        print(f"\n[IMU校准] 完成! 基准角度: 俯仰={self.reference_pitch:.2f}° 横滚={self.reference_roll:.2f}°")
    
    def get_relative_angle(self, raw_pitch, raw_roll):
        if self.reference_pitch is None:
            return raw_pitch, raw_roll
        return raw_pitch - self.reference_pitch, raw_roll - self.reference_roll

# ===== IMU数据解析 =====
class IMUParser:
    @staticmethod
    def parse(data):
        if len(data) != 10 or data[0] != 0x55:
            return None
        
        if data[1] == 0x53:  # 角度数据
            return {
                'type': 'angle',
                'raw_pitch': ((data[5]<<8)|data[4])/32768 * 180,
                'raw_roll': ((data[3]<<8)|data[2])/32768 * 180,
                'timestamp': time.time()
            }
        elif data[1] == 0x52:  # 陀螺仪数据
            return {
                'type': 'imu',
                'gyro_x': ((data[3]<<8)|data[2])/32768 * 2000,  # 度/秒
                'gyro_y': ((data[5]<<8)|data[4])/32768 * 2000,
                'timestamp': time.time()
            }
        return None

# ===== 地形检测系统 =====
class TerrainDetector:
    def __init__(self, imu_calibrator):
        self.calibrator = imu_calibrator
        self.angle_history = deque(maxlen=15)
        self.slope_threshold = 5.0  # 坡度阈值(度)
        self.current_slope = 'level'
        self.slope_confidence = 0
        
        # 基于陀螺仪的台阶检测
        self.step_gyro_threshold = 25.0  # 度/秒
        self.step_roll_threshold = 5.0    # 度
        self.last_roll = 0
        self.step_cooldown = 0.4  # 秒

    def update(self, imu_data):
        if imu_data is None:
            return None
            
        if imu_data['type'] == 'angle':
            return self._process_slope(imu_data)
        elif imu_data['type'] == 'imu':
            return self._process_step(imu_data)
        return None
    
    def _process_slope(self, angle_data):
        pitch, roll = self.calibrator.get_relative_angle(
            angle_data['raw_pitch'], angle_data['raw_roll']
        )
        self.angle_history.append(pitch)
        self.last_roll = roll  # 更新横滚角记录
        
        weights = np.linspace(0.3, 1.0, len(self.angle_history))  # 越新权重越高
        avg_pitch = np.average(list(self.angle_history), weights=weights)
        self.currentp = pitch -avg_pitch
        print(f"Current Pitch: {pitch:.2f}°, Avg Pitch: {abs(avg_pitch):.2f}°,currentP: {currentp:.2f}°")
        new_slope = 'level'
        if currentp > self.slope_threshold:
            new_slope = 'up'
        elif currentp < -self.slope_threshold:
            new_slope = 'down'
        
        if new_slope != self.current_slope:
            self.current_slope = new_slope
            if new_slope != 'level':
                return {
                    'type': 'terrain',
                    'subtype': 'slope',
                    'direction': new_slope,
                    'intensity': abs(avg_pitch),
                    'timestamp': time.time()
                }
        return None

    def _process_step(self, imu_data):
        current_time = time.time()
        if current_time - getattr(self, 'last_step_time', 0) < self.step_cooldown:
            return None
            
        current_gyro_x = imu_data['gyro_x']
        print(current_gyro_x)
        roll_change = self.currentp
        
        if abs(current_gyro_x) > self.step_gyro_threshold and roll_change > self.step_roll_threshold:
            self.last_step_time = current_time
            direction = 'up' if current_gyro_x > 0 else 'down'
            return {
                'type': 'terrain',
                'subtype': 'step',
                'direction': direction,
                'intensity': abs(current_gyro_x),
                'timestamp': current_time
            }
        return None

# ===== 语音系统 =====
class VoiceSystem:
    def __init__(self):
        self.queue = queue.Queue()
        self.worker = threading.Thread(target=self._speak_worker, daemon=True)
        self.worker.start()
    
    def add_message(self, text, priority='info'):
        self.queue.put((text, priority))
    
    def _speak_worker(self):
        while True:
            text, priority = self.queue.get()
            try:
                speed = 300 if priority == 'warning' else 200
                subprocess.run(['espeak', '-v', 'zh', '-s', str(speed), text],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"语音播报失败: {e}")

# ===== 主系统 =====
class NavigationSystem:
    def __init__(self):
        HardwareConfig.setup()
        self.voice = VoiceSystem()
        
        try:
            self.ser = serial.Serial(
                HardwareConfig.SERIAL_PORT,
                HardwareConfig.SERIAL_BAUDRATE,
                timeout=0.1
            )
            print("IMU串口已连接")
        except Exception as e:
            print(f"串口连接失败: {e}")
            sys.exit(1)
        
        self.calibrator = IMUCalibrator()
        self.calibrator.calibrate(self.ser)
        self.detector = TerrainDetector(self.calibrator)
        
        # 启动服务线程
        threading.Thread(target=self._imu_service, daemon=True).start()
 
        
        self.voice.add_message("导航系统已就绪")

    def _imu_service(self):
        buffer = b''
        while True:
            try:
                buffer += self.ser.read(self.ser.in_waiting or 1)
                while len(buffer) >= 10:
                    if buffer[0] == 0x55:
                        frame = buffer[:10]
                        buffer = buffer[10:]
                        imu_data = IMUParser.parse(frame)
                        if imu_data:
                            event = self.detector.update(imu_data)
                            if event:
                                self._handle_event(event)
                    else:
                        buffer = buffer[1:]
                time.sleep(0.01)
            except Exception as e:
                print(f"IMU服务异常: {e}")
                time.sleep(1)
    
    def _ultrasonic_service(self):
        while True:
            try:
                for sensor in HardwareConfig.SENSORS:
                    dist = self._measure_distance(sensor.trig, sensor.echo)
                    if dist < 0.5:  # 0.5米障碍物检测
                        self.voice.add_message(f"{sensor.name}检测到障碍物", 'warning')
                time.sleep(0.2)
            except Exception as e:
                print(f"超声波异常: {e}")
                time.sleep(1)
    
    def _handle_event(self, event):
        if event['subtype'] == 'slope':
            self.voice.add_message(
                f"检测到{'上' if event['direction']=='up' else '下'}坡",
                'warning'
            )
        elif event['subtype'] == 'step':
            self.voice.add_message(
                f"检测到{'上' if event['direction']=='up' else '下'}台阶", 
                'warning'
            )

    @staticmethod
    def _measure_distance(trig, echo):
        GPIO.output(trig, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(trig, GPIO.LOW)

        while GPIO.input(echo) == 0:
            pulse_start = time.time()
        while GPIO.input(echo) == 1:
            pulse_end = time.time()

        return (pulse_end - pulse_start) * 17150  # 厘米换算

if __name__ == "__main__":
    print("导航辅助系统启动中...")
    atexit.register(lambda: GPIO.cleanup())
    
    try:
        nav = NavigationSystem()
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\n系统关闭中...")
    finally:
        sys.exit(0)
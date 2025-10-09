#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import RPi.GPIO as GPIO
import time
import threading
import queue
import subprocess
import atexit
import sys
from dataclasses import dataclass

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

# ===== 语音消息系统 =====
class VoiceSystem:
    PRIORITY_LEVELS = {
        'emergency': {'speed': 400, 'vibrate': 80},
        'warning': {'speed': 300, 'vibrate': 60},
        'info': {'speed': 200, 'vibrate': 40}
    }
    
    def __init__(self):
        self.queue = queue.PriorityQueue()
        self.active_messages = set()
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
                
                if text in self.active_messages:
                    self.active_messages.remove(text)
                    
            except Exception as e:
                print(f"[语音系统] 播报失败: {e}")
            finally:
                self.queue.task_done()

# ===== 超声波服务 =====
class UltrasonicService:
    OBSTACLE_THRESHOLD = 0.1  # 10cm
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
        GPIO.setmode(GPIO.BCM)
        last_alert_time = 0
        
        while True:
            try:
                now = time.time()
                active_obstacles = []
                
                for s in HardwareConfig.SENSORS:
                    dist = cls.measure_distance(s.trig, s.echo)
                    if dist < cls.OBSTACLE_THRESHOLD:
                        active_obstacles.append(s.key)
                
                if active_obstacles and now - last_alert_time > cls.ALERT_INTERVAL:
                    if len(active_obstacles) > 1:
                        names = [s.name for s in HardwareConfig.SENSORS 
                                if s.key in active_obstacles]
                        message = "同时检测到" + "和".join(names) + "有障碍物"
                    else:
                        sensor = next(s for s in HardwareConfig.SENSORS 
                                    if s.key in active_obstacles)
                        message = f"{sensor.name}有障碍物"
                    
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
        
        # 启动服务线程
        self.threads = [
            threading.Thread(target=self._ultrasonic_service, daemon=True),
            threading.Thread(target=self._status_monitor, daemon=True)
        ]
        
        for t in self.threads:
            t.start()
        
        print("[系统] 超声波服务已启动")

    def _ultrasonic_service(self):
        """超声波避障服务"""
        UltrasonicService.run(self.msg_system)
    
    def _status_monitor(self):
        """系统状态监控"""
        while True:
            time.sleep(5)
            # 可添加系统健康检查逻辑

# ===== 主程序 =====
if __name__ == '__main__':
    print(""" 
   _____ 超声波避障系统 _____
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
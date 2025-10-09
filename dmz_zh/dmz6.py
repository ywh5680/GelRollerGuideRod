import time
import subprocess
import serial
import threading
import queue
from collections import deque

# ================= 参数配置 =================
SAMPLE_WINDOW       = 20      # 缓存帧数，约 0.5 秒 @50Hz
ANGLE_THRESH        = 5.0     # 坡度累计变化阈值（度）
HEIGHT_RATE_THRESH  = 0.05    # 坡道条件下最大高度速率（米/秒）
HEIGHT_STEP_THRESH  = 0.10    # 台阶跃变阈值（米）
# 语音去重
spoken_state = set()
# 播报队列
speak_queue = queue.Queue()

def speak_worker():
    while True:
        text = speak_queue.get()
        if text is None:
            break
        subprocess.call(['espeak', '-v', 'zh', '-s', '200', text])
        speak_queue.task_done()

def speak_async(text):
    speak_queue.put(text)
    
    # ================= 串口 IMU 解析 =================
def parse_angle_frame(frame: bytes) -> float:
    pitch = ((frame[5] << 8) | frame[4]) / 32768 * 180
    if pitch > 180:
        pitch -= 360
    return pitch

def parse_height_frame(frame: bytes) -> float:
    h_cm = (frame[9] << 24) | (frame[8] << 16) | (frame[7] << 8) | frame[6]
    return h_cm / 100.0  # 转换为 米

# ================= 主程序 =================
def main():
    threading.Thread(target=speak_worker, daemon=True).start()
    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=0.5)
    print("启动：坡道 vs 台阶 检测 (固定帧窗口)")

    # 缓存 (timestamp, pitch, height) 的固定长度队列
    window = deque(maxlen=SAMPLE_WINDOW)

    last_pitch  = None
    last_height = None
    try:
        while True:
            # 读取 IMU 帧
            if ser.in_waiting >= 10:
                head = ser.read(1)
                if head != b'\x55':
                    continue
                t    = ser.read(1)[0]
                data = ser.read(8)
                frame = b'\x55' + bytes([t]) + data
                now = time.time()

                if t == 0x53:  # 角度帧
                    pitch = parse_angle_frame(frame)
                    last_pitch = pitch
                elif t == 0x56:  # 高度帧
                    height = parse_height_frame(frame)
                    last_height = height
                else:
                    continue

                # 同时更新到缓存
                if last_pitch is not None and last_height is not None:
                    window.append((now, last_pitch, last_height))

            # 如果数据不够，跳过
            if len(window) < 2:
                time.sleep(0.005)
                continue

            # 拆分
            times   = [x[0] for x in window]
            pitches = [x[1] for x in window]
            heights = [x[2] for x in window]

            # 计算 sprint 变化量和高度速率
            angle_change = max(pitches) - min(pitches)
            duration     = times[-1] - times[0] or 1e-6
            height_rate  = abs(heights[-1] - heights[0]) / duration

            # 计算台阶跃变（首尾帧差值）
            height_jump  = abs(heights[-1] - heights[0])

            # 决策：台阶优先
            if height_jump > HEIGHT_STEP_THRESH:
                state = "上台阶" if (heights[-1] - heights[0]) > 0 else "下台阶"
                time.sleep(0.1)
            elif angle_change > ANGLE_THRESH and height_rate < HEIGHT_RATE_THRESH:
                state = "下坡" if (pitches[-1] - pitches[0]) > 0 else "上坡"
            else:
                state = None

            # 播报
            if state and state not in spoken_state:
                speak_async(f"检测到{state}")
                spoken_state.clear()
                spoken_state.add(state)

            # 短延时，保持高响应
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("退出程序")

    finally:
        speak_queue.put(None)
        ser.close()

if __name__ == '__main__':
    main()
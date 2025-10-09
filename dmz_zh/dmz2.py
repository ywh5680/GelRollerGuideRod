import time
import subprocess
import serial
import threading
import queue

# ================ 常量 & 状态 ================
STEP_HEIGHT_THRESHOLD  = 0.10    # 台阶高度突变阈值（米）
STEP_CONFIRM_TIME      = 0.02    # 突变后持续稳定时间（秒）
STEP_STABLE_TOLERANCE  = 1   # 稳定判断容差（米）
PITCH_UP_THRESHOLD     = 5.0    # 上坡俯仰角阈值（°）
PITCH_DOWN_THRESHOLD   = -5.0   # 下坡俯仰角阈值（°）
last_pitch = None
i= 0
# 全局状态
spoken_state       = set()     # 记录已播报状态，防重复
pending_step       = None      # "上台阶"/"下台阶" 待确认
pending_step_time  = None
pending_height_ref = None

# 语音播报队列
speak_queue = queue.Queue()

# ================ 语音播报 ================
def speak_worker():
    while True:
        text = speak_queue.get()
        if text is None:
            break
        subprocess.call(['espeak', '-v', 'zh', '-s', '200', text])
        speak_queue.task_done()

def speak_async(text):
    speak_queue.put(text)

# ================ IMU 帧解析 ================
def parse_angle_from_frame(data):
    # data 为 10 字节：0x55,0x53,然后8字节
    roll  = ((data[3]<<8)|data[2]) / 32768 * 180
    pitch = ((data[5]<<8)|data[4]) / 32768 * 180
    yaw   = ((data[7]<<8)|data[6]) / 32768 * 180
    return roll, pitch, yaw

def parse_height_from_pressure_frame(data):
    # data 为 10 字节：0x55,0x56,然后8字节
    height_cm = (data[9]<<24)|(data[8]<<16)|(data[7]<<8)|data[6]
    return height_cm / 90

# ================ 坡度检测 ================
def detect_slope(pitch):
    global last_pitch,i,mt
    i +=1
    if pitch > 180:
        pitch -= 360
    if last_pitch is None:  
        last_pitch = pitch
    delta = pitch-last_pitch
    print("当前角度： ",pitch)
    print("上一角度： ",last_pitch)
    print("delta： ",delta)
    if i==60:
        last_pitch = pitch
        i=0
    if delta > PITCH_UP_THRESHOLD:
        return "下坡"
    elif delta < PITCH_DOWN_THRESHOLD:
        return "上坡"
    else:
        return "平路"

# ================ 台阶检测（带确认机制） ================
def detect_step_with_confirmation(current_h):
    """
    1) 首次记录 last_h，不做判断。
    2) 若高度与 last_h 差值 > 阈值，则进入“待确认”状态。
    3) 在“待确认”中，检测高度是否在基准高度±容差内持续超过确认时间，才返回“上台阶”或“下台阶”。
    4) 否则重置，返回 None。
    """
    global pending_step, pending_step_time, pending_height_ref

    # 初始化 last_h
    if not hasattr(detect_step_with_confirmation, "last_h") or detect_step_with_confirmation.last_h is None:
        detect_step_with_confirmation.last_h = current_h
        print(f"[STEP] init last_h = {current_h:.3f}")
        return None

    last_h = detect_step_with_confirmation.last_h
    delta  = current_h - last_h
    print(f"[STEP] current_h={current_h:.3f}, last_h={last_h:.3f}, Δ={delta:.3f}, pending_step={pending_step}")
    # 第一次检测到突变
    if pending_step is None:
        if abs(delta) > STEP_HEIGHT_THRESHOLD:
            pending_step       = "上台阶" if delta>0 else "下台阶"
            pending_step_time  = time.time()
            pending_height_ref = current_h
            print(f"[STEP] 进入待确认: {pending_step} at {pending_height_ref:.3f}")
    else:
        # 检查稳定
        stable_delta = abs(current_h - pending_height_ref)
        elapsed      = time.time() - pending_step_time
        print(f"[STEP] 待确认中: stable_delta={stable_delta:.3f}, elapsed={elapsed:.2f}")
        if stable_delta < STEP_STABLE_TOLERANCE:
            if elapsed >= STEP_CONFIRM_TIME:
                confirmed = pending_step
                print(f"[STEP] 确认台阶: {confirmed}")
                pending_step       = None
                pending_step_time  = None
                pending_height_ref = None
                detect_step_with_confirmation.last_h = current_h
                return confirmed
        else:
            print("[STEP] 波动过大，重置")
            pending_step       = None
            pending_step_time  = None
            pending_height_ref = None

    detect_step_with_confirmation.last_h = current_h
    return None
 
# 初始化静态属性
detect_step_with_confirmation.last_h = None

# ================ 主循环 ================
def main():
    # 启动语音播报线程
    threading.Thread(target=speak_worker, daemon=True).start()

    # 打开 IMU 串口
    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=0.5)
    print("开始测试：上下坡与上下台阶播报逻辑")

    # 用于控制“台阶优先于坡度”
    step_recently_detected = False
    step_recently_timer = 0
    STEP_PRIORITY_HOLD_TIME = 1.0  # 台阶优先时长（秒）

    try:
        while True:
            if ser.in_waiting >= 10:
                head = ser.read(1)
                if head != b'\x55':
                    continue
                t = ser.read(1)
                data = ser.read(8)
                frame = head + t + data

                step_state = None
                slope_state = None

                if t[0] == 0x53:  # 角度帧
                    _, pitch, _ = parse_angle_from_frame(frame)
                    print(pitch)
                    
                    slope_state = detect_slope(pitch)
                
                elif t[0] == 0x56:  # 高度帧
                    h = parse_height_from_pressure_frame(frame)
                    step_state = detect_step_with_confirmation(h)
                


                # 优先播报台阶
                if step_state and step_state not in spoken_state:
                    speak_async(f"检测到{step_state}")
                    spoken_state.clear()
                    spoken_state.add(step_state)
                    step_recently_detected = True
                    step_recently_timer = time.time()

                # 如果最近没有台阶检测，才允许坡度播报
                elif (not step_recently_detected 
                      and slope_state 
                      and slope_state != "平路" 
                      and slope_state not in spoken_state):
                    speak_async(f"检测到{slope_state}")
                    spoken_state.clear()
                    spoken_state.add(slope_state)

                # 清除台阶优先标志（超时）
                if step_recently_detected and (time.time() - step_recently_timer > STEP_PRIORITY_HOLD_TIME):
                    step_recently_detected = False

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("测试结束，退出。")

    finally:
        speak_queue.put(None)
        ser.close()

if __name__ == '__main__':
    main()
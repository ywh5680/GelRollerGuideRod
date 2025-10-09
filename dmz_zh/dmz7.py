import time
import subprocess
import serial
import threading
import queue

# ================ 常量 & 状态 ================
STEP_HEIGHT_THRESHOLD  = 0.10    # 台阶高度突变阈值（米）
STEP_CONFIRM_TIME      = 0.02    # 突变后持续稳定时间（秒）
STEP_STABLE_TOLERANCE  = 1   # 稳定判断容差（米）
PITCH_UP_THRESHOLD     = 4.0    # 上坡俯仰角阈值（°）
PITCH_DOWN_THRESHOLD   = -4.0   # 下坡俯仰角阈值（°）
last_pitch = None
i = 0
j = 0
# 全局状态
spoken_state       = set()     # 记录已播报状态，防重复
pending_step       = None      # "上台阶"/"下台阶" 待确认
pending_step_time  = None
pending_height_ref = None
# 台阶检测状态机
step_state_machine = {}

# 台阶检测历史高度
step_last_h = None
# 台阶检测状态
gyro_step_state = {}

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
    return height_cm / 89

# ================ 坡度检测 ================
def detect_slope(pitch,wx):
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
    if i==50:
        last_pitch = pitch
        i=0
    if delta > PITCH_UP_THRESHOLD and abs(wx)<10:
        return "下坡"
    elif delta < PITCH_DOWN_THRESHOLD and abs(wx)<10:
        return "上坡"
    else:
        return "平路"

# ================ 台阶检测（带确认机制） ================
def detect_step_with_confirmation_yuan(current_h,wx):
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
    if pending_step is None and abs(wx)>20:
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
# ================ 台阶检测（状态机安全版） ================
def detect_step_with_confirmation(current_h, wx):
    """
    状态机设计：
    - IDLE: 空闲状态
    - PENDING: 进入待确认状态
    """
    global step_state_machine

    # 初始化状态机
    if 'state' not in step_state_machine:
        step_state_machine['state'] = 'IDLE'
        step_state_machine['pending_step'] = None
        step_state_machine['pending_step_time'] = None
        step_state_machine['pending_height_ref'] = None
        step_state_machine['last_h'] = current_h
        print(f"[STEP] 初始化高度: {current_h:.3f}")
        return None

    # 取出状态
    state = step_state_machine['state']
    last_h = step_state_machine['last_h']

    delta = current_h - last_h
    print(f"[STEP] 当前高度={current_h:.3f}, 上次高度={last_h:.3f}, 差值Δ={delta:.3f}, 当前状态={state}")

    if state == 'IDLE':
        if abs(wx) > 25 and abs(delta) > STEP_HEIGHT_THRESHOLD:
            # 进入待确认状态
            step_state_machine['state'] = 'PENDING'
            step_state_machine['pending_step'] = "上台阶" if delta > 0 else "下台阶"
            step_state_machine['pending_step_time'] = time.time()
            step_state_machine['pending_height_ref'] = current_h
            print(f"[STEP] 进入待确认状态: {step_state_machine['pending_step']} at {current_h:.3f}")
    elif state == 'PENDING':
        stable_delta = abs(current_h - step_state_machine['pending_height_ref'])
        elapsed = time.time() - step_state_machine['pending_step_time']
        print(f"[STEP] 确认中: 高度波动Δ={stable_delta:.3f}, 已持续时间={elapsed:.2f}s")

        if stable_delta < STEP_STABLE_TOLERANCE:
            if elapsed >= STEP_CONFIRM_TIME:
                confirmed = step_state_machine['pending_step']
                print(f"[STEP] 确认台阶: {confirmed}")
                step_state_machine['state'] = 'IDLE'
                step_state_machine['last_h'] = current_h
                return confirmed
        else:
            print("[STEP] 波动过大，重置为 IDLE")
            step_state_machine['state'] = 'IDLE'

    step_state_machine['last_h'] = current_h
    return None
# ================ 台阶检测（简化版） ================
def detect_step(current_h, wx):
    """
    只要高度发生突变立即判断，不使用确认机制。
    """
    global step_last_h,j
    j+=1
    if j==10:
        step_last_h = current_h
        j=0
    print(j)
    # 初始化历史高度
    if step_last_h is None:
        step_last_h = current_h
        print(f"[STEP] 初始化高度: {current_h:.3f}")
        return None

    delta = current_h - step_last_h
    print(f"[STEP] 当前高度={current_h:.3f}, 上次高度={step_last_h:.3f}, 差值Δ={delta:.3f}")

    if abs(wx) > 15 and abs(delta) > STEP_HEIGHT_THRESHOLD:
        step_state = "上台阶" if delta > 0 else "下台阶"
        return step_state


    return None

# ================ 台阶检测（峰值法 + 冷却） ================
def detect_step_by_gyro(wx):
    """
    改进版：只检测单向峰值（不要求反转），支持 1 秒冷却防连发
    """
    global gyro_step_state

    # 初始化状态机
    if 'phase' not in gyro_step_state:
        gyro_step_state['phase'] = 'IDLE'
        gyro_step_state['count'] = 0
        gyro_step_state['threshold'] = 25
        gyro_step_state['required_count'] = 3  # 连续 N 帧确认
        gyro_step_state['cooldown'] = False
        gyro_step_state['cooldown_start'] = 0
        gyro_step_state['cooldown_time'] = 1.0  # 冷却时间 1 秒

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
                return "下台阶"
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
                return "上台阶"
    else:
        # 小角速度时重置
        gyro_step_state['phase'] = 'IDLE'
        gyro_step_state['count'] = 0

    return None










# 初始化静态属性
detect_step_with_confirmation.last_h = None

# ================ 主循环 ================
def main():
    wx =0
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

                if t[0] == 0x52:  # 陀螺仪帧
                    wx = ((data[3] << 8) | data[2]) / 32768 * 2000
                    wy = ((data[5] << 8) | data[4]) / 32768 * 2000
                    wz = ((data[7] << 8) | data[6]) / 32768 * 2000

                    if wx > 2000:
                        wx -= 4000
                    if wy > 2000:
                        wy -= 4000
                    if wz > 2000:
                        wz -= 4000

                    print(f"陀螺仪角速度 wx={wx:.2f} °/s, wy={wy:.2f} °/s, wz={wz:.2f} °/s")
                    step_state = detect_step_by_gyro(wx)
                elif t[0] == 0x53:  # 角度帧
                    _, pitch, _ = parse_angle_from_frame(frame)
                    print(pitch)
                        # 冷却时间内，禁止坡度检测
                    if gyro_step_state.get('cooldown', False):
                        print("[SLOPE] 冷却时间内，忽略坡度检测")
                        continue
                    slope_state = detect_slope(pitch,wx)






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


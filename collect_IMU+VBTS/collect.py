import serial
import cv2
import time
import csv
import threading
import os
import sys

# ---------------- 全局参数 ----------------
stop_flag = False  # 控制线程停止
duration = 12      # 采集时长（秒），可调整

# 允许的地面类别
ALLOWED_GROUNDS = ['GRAS', 'ICE', 'TILE', 'GLAS', 'TRACK', 'ASPH', 'MUD', 'SAND', 'GRVL']

# ---------------- IMU Functions ----------------
ACCData = [0.0]*8
GYROData = [0.0]*8
AngleData = [0.0]*8
FrameState = 0
Bytenum = 0
CheckSum = 0

acc = [0.0]*3
gyro = [0.0]*3
angle = [0.0]*3

def get_acc(datahex):
    k_acc = 16.0
    acc_x = (datahex[1] << 8 | datahex[0]) / 32768.0 * k_acc
    acc_y = (datahex[3] << 8 | datahex[2]) / 32768.0 * k_acc
    acc_z = (datahex[5] << 8 | datahex[4]) / 32768.0 * k_acc
    if acc_x >= k_acc: acc_x -= 2 * k_acc
    if acc_y >= k_acc: acc_y -= 2 * k_acc
    if acc_z >= k_acc: acc_z -= 2 * k_acc
    return acc_x, acc_y, acc_z

def get_gyro(datahex):
    k_gyro = 2000.0
    gx = (datahex[1] << 8 | datahex[0]) / 32768.0 * k_gyro
    gy = (datahex[3] << 8 | datahex[2]) / 32768.0 * k_gyro
    gz = (datahex[5] << 8 | datahex[4]) / 32768.0 * k_gyro
    if gx >= k_gyro: gx -= 2 * k_gyro
    if gy >= k_gyro: gy -= 2 * k_gyro
    if gz >= k_gyro: gz -= 2 * k_gyro
    return gx, gy, gz

def get_angle(datahex):
    k_angle = 180.0
    ax = (datahex[1] << 8 | datahex[0]) / 32768.0 * k_angle
    ay = (datahex[3] << 8 | datahex[2]) / 32768.0 * k_angle
    az = (datahex[5] << 8 | datahex[4]) / 32768.0 * k_angle
    if ax >= k_angle: ax -= 2 * k_angle
    if ay >= k_angle: ay -= 2 * k_angle
    if az >= k_angle: az -= 2 * k_angle
    return ax, ay, az

def DueData(inputdata, writer):
    global FrameState, Bytenum, CheckSum, acc, gyro, angle
    for data in inputdata:
        if FrameState == 0:
            if data == 0x55 and Bytenum == 0:
                CheckSum = data
                Bytenum = 1
                continue
            elif data == 0x51 and Bytenum == 1:
                CheckSum += data; FrameState = 1; Bytenum = 2
            elif data == 0x52 and Bytenum == 1:
                CheckSum += data; FrameState = 2; Bytenum = 2
            elif data == 0x53 and Bytenum == 1:
                CheckSum += data; FrameState = 3; Bytenum = 2
        elif FrameState == 1:  # acc
            if Bytenum < 10:
                ACCData[Bytenum-2] = data; CheckSum += data; Bytenum += 1
            else:
                if data == (CheckSum & 0xff):
                    acc = get_acc(ACCData)
                FrameState = 0; Bytenum = 0; CheckSum = 0
        elif FrameState == 2:  # gyro
            if Bytenum < 10:
                GYROData[Bytenum-2] = data; CheckSum += data; Bytenum += 1
            else:
                if data == (CheckSum & 0xff):
                    gyro = get_gyro(GYROData)
                FrameState = 0; Bytenum = 0; CheckSum = 0
        elif FrameState == 3:  # angle
            if Bytenum < 10:
                AngleData[Bytenum-2] = data; CheckSum += data; Bytenum += 1
            else:
                if data == (CheckSum & 0xff):
                    angle = get_angle(AngleData)
                    ts = time.time()
                    row = [ts] + list(acc) + list(gyro) + list(angle)
                    writer.writerow(row)
                FrameState = 0; Bytenum = 0; CheckSum = 0

# ---------------- IMU Thread ----------------
def imu_thread(csv_file="imu.csv"):
    global stop_flag
    try:
        ser = serial.Serial('/dev/ttyUSB0',115200, timeout=0.1)
    except Exception as e:
        print("打开串口失败:", e)
        return

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "acc_x", "acc_y", "acc_z",
                         "gyro_x", "gyro_y", "gyro_z",
                         "angle_x", "angle_y", "angle_z"])
        buffer=[]
        while not stop_flag:
            data = ser.read(1)
            if data:
                buffer.append(data[0])
                if len(buffer) >= 11:
                    DueData(buffer, writer)
                    buffer = []
                    f.flush()
    ser.close()

# ---------------- Camera Thread ----------------
def camera_thread(video_file="video.avi", ts_file="frame_timestamps.csv"):
    global stop_flag
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_file, fourcc, 30.0, (640, 480))

    with open(ts_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "timestamp"])
        frame_index = 0

        while not stop_flag:
            ret, frame = cap.read()
            if not ret:
                continue

            ts = time.time()
            out.write(frame)

            writer.writerow([frame_index, ts])
            frame_index += 1
            f.flush()

    cap.release()
    out.release()

# ---------------- Main ----------------
if __name__ == '__main__':
    # 用户设置（注意输入会被规范化为大写）
    category = input("请输入类别代码 (CS/CR/RS/RR/WET): ").strip().upper()
    ground_class = input("请输入地面类别 (GRAS/ICE/TILE/GLAS/TRACK/ASPH/MUD/SAND/GRVL): ").strip().upper()
    index = input("请输入样本编号 (如 001): ").zfill(3)

    # 简单校验
    if ground_class not in ALLOWED_GROUNDS:
        print(f"地面类别不在允许列表: {ALLOWED_GROUNDS}")
        sys.exit(1)
    if category == "":
        print("类别代码不能为空。")
        sys.exit(1)

    # 目录结构: category / ground_class_index / ...
    base_dir = os.path.join(category, f"{ground_class}_{index}")
    os.makedirs(base_dir, exist_ok=True)

    imu_file = os.path.join(base_dir, "imu.csv")
    video_file = os.path.join(base_dir, "video.avi")
    ts_file = os.path.join(base_dir, "frame_timestamps.csv")

    # 启动线程
    t1 = threading.Thread(target=imu_thread, args=(imu_file,))
    t2 = threading.Thread(target=camera_thread, args=(video_file, ts_file))
    start_time = time.time()
    t1.start()
    t2.start()

    # 自动停止
    while time.time() - start_time < duration:
        time.sleep(0.1)
    stop_flag = True

    t1.join()
    t2.join()
    print(f"数据采集完成: {base_dir}")
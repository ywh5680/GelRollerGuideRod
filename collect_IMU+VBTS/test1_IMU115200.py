import serial
import cv2
import time
import csv
import threading
import os
import sys

# ---------------- 全局参数 ----------------
stop_flag = False  # 控制线程停止
duration = 10       # 采集时长（秒），可调整

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
    ser = serial.Serial('/dev/ttyUSB0',115200, timeout=0.1)
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
def camera_thread(save_dir="frames"):
    global stop_flag
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            continue
        ts = time.time()
        filename = os.path.join(save_dir, f"{ts:.6f}.png")
        cv2.imwrite(filename, frame)
    cap.release()

# ---------------- Main ----------------
if __name__ == '__main__':
    # 用户设置类别和编号
    category = input("请输入类别代码 (CS/CR/RS/RR): ").strip()
    index = input("请输入样本编号 (如 001): ").zfill(3)

    base_dir = f"{category}_{index}"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "frames"), exist_ok=True)

    imu_file = os.path.join(base_dir, "imu.csv")
    frames_dir = os.path.join(base_dir, "frames")

    # 启动线程
    t1 = threading.Thread(target=imu_thread, args=(imu_file,))
    t2 = threading.Thread(target=camera_thread, args=(frames_dir,))
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
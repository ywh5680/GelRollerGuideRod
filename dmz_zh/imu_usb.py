# imu_reader.py
import serial
import threading
import queue

class IMUReader(threading.Thread):
    def __init__(self, port='/dev/ttyUSB2', baud=9600):
        super().__init__()
        self.ser = serial.Serial(port, baud, timeout=0.5)
        self.angle_queue = queue.Queue()
        self._stop_event = threading.Event()
        self.FrameState = 0
        self.Bytenum = 0
        self.CheckSum = 0
        self.AngleData = [0.0] * 8

    def run(self):
        while not self._stop_event.is_set():
            data = self.ser.read(33)
            self.DueData(data)

    def stop(self):
        self._stop_event.set()

    def DueData(self, inputdata):
        for data in inputdata:
            if self.FrameState == 0:
                if data == 0x55 and self.Bytenum == 0:
                    self.CheckSum = data
                    self.Bytenum = 1
                elif data == 0x53 and self.Bytenum == 1:
                    self.CheckSum += data
                    self.FrameState = 3
                    self.Bytenum = 2
                else:
                    self.Bytenum = 0
            elif self.FrameState == 3:
                if self.Bytenum < 10:
                    self.AngleData[self.Bytenum - 2] = data
                    self.CheckSum += data
                    self.Bytenum += 1
                else:
                    if data == (self.CheckSum & 0xff):
                        angle = self.get_angle(self.AngleData)
                        self.angle_queue.put(angle)  # 放入队列
                    self.CheckSum = 0
                    self.Bytenum = 0
                    self.FrameState = 0

    def get_angle(self, datahex):
        rxl, rxh, ryl, ryh, rzl, rzh = datahex[0:6]
        k_angle = 180.0
        angle_x = (rxh << 8 | rxl) / 32768.0 * k_angle
        angle_y = (ryh << 8 | ryl) / 32768.0 * k_angle
        angle_z = (rzh << 8 | rzl) / 32768.0 * k_angle
        if angle_x >= k_angle: angle_x -= 2 * k_angle
        if angle_y >= k_angle: angle_y -= 2 * k_angle
        if angle_z >= k_angle: angle_z -= 2 * k_angle
        return angle_x, angle_y, angle_z

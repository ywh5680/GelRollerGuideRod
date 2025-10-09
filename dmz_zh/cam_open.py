import cv2

# 创建摄像头对象（0表示默认摄像头）
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("成功打开摄像头")
print("按 'q' 键退出")

# 设置摄像头分辨率（可选）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

try:
    while True:
        # 读取一帧画面
        ret, frame = cap.read()
        
        # 检查是否成功获取画面
        if not ret:
            print("无法获取画面")
            break
        
        # 显示画面
        cv2.imshow('Camera Feed', frame)
        
        # 按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    print("摄像头已关闭")
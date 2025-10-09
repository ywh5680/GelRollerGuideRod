import cv2
import os

def video_to_frames(video_path, output_dir="frames"):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束

        # 保存为 PNG 图片，文件名按顺序编号
        frame_filename = os.path.join(output_dir, f"{frame_index}.png")
        cv2.imwrite(frame_filename, frame)
        frame_index += 1

        if frame_index % 100 == 0:
            print(f"已保存 {frame_index} 帧...")

    cap.release()
    print(f"完成！总共保存 {frame_index} 帧到文件夹 {output_dir}")

if __name__ == "__main__":
    video_path = "/home/pi/Desktop/collect_IMU+VBTS/cs_008/video.avi"  # 输入视频路径
    output_dir = "/home/pi/Desktop/collect_IMU+VBTS/cs_008/frames"     # 输出图片文件夹
    video_to_frames(video_path, output_dir)
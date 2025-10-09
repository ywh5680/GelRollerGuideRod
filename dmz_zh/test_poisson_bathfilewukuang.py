import glob
import os
import sys

sys.path.append('/Code/Calib')
import warnings
import numpy as np
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from tqdm import tqdm
# from color_com import color_mean
from fast_poisson import fast_poisson
import cv2
import matplotlib.pyplot as plt
from calibration import calibration
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import gc

class Fit_Guassian:
    def __init__(self, plot=None) -> None:
        self.plot = plot
        self.popt = None

    def save(self, img_path) -> None:
        header_file, file_num = os.path.split(img_path)
        file_num, _ = os.path.splitext(file_num)
        head_file, _ = os.path.split(header_file)
        save_file = head_file + "/data/"
        if not os.path.exists(save_file):
            os.makedirs(save_file)
        try:
            with open(head_file + '/result_.txt', "r") as file:
                force_list = file.read().splitlines()
            if len(force_list) > 0:
                F = np.float32(force_list[int(file_num) - 1])
                print(F)
                save_content = np.append(self.popt, F)
                np.savetxt(save_file + "popt.txt", save_content)
            else:
                warnings.warn("无法读出内容，请检查文件", Warning)
        except IOError:
            warnings.warn("文件打开失败！", IOError)

    def gaussian_3d(self, X, a, x0, y0, sigma_x, sigma_y, b) -> np.ndarray:
        return (a * np.exp(-(X[0] - x0) ** 2 / (2 * sigma_x ** 2) - (X[1] - y0) ** 2 / (2 * sigma_y ** 2)) + b).ravel()

    def show(self, X, Y, Z) -> None:
        def gaussian_3d(X, a, x0, y0, sigma_x, sigma_y, b):
            return (a * np.exp(-(X[0] - x0) ** 2 / (2 * sigma_x ** 2) - (X[1] - y0) ** 2 / (2 * sigma_y ** 2)) + b)

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax3d = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet')
        if self.popt:
            ax.plot_surface(X, Y, gaussian_3d(
                (X, Y), *(self.popt)), cmap='viridis')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        plt.show()

    def fit_guassian_3d(self, image, img_path=None) -> None:
        image = cv2.GaussianBlur(image.astype(np.float32), (3, 3), 0)
        image = cv2.resize(image, (50, 50))
        max_val = np.max(image)
        (y_max, x_max) = np.where(image == max_val)
        p0 = [max_val, x_max[0], y_max[0], 8, 8, -0.1]
        M, N = image.shape
        M, N = 50, 50
        x = np.arange(N)
        y = np.arange(M)
        X, Y = np.meshgrid(x, y)
        XX = np.expand_dims(X, 0)
        YY = np.expand_dims(Y, 0)
        xx = np.append(XX, YY, axis=0)
        self.popt, pcov = curve_fit(self.gaussian_3d, xx, image.ravel(), p0=p0)
        print(self.popt)
        if self.plot:
            self.show(X, Y, image)
        if img_path:
            self.save(img_path)


def makedir(dir_path):
    dir_path = os.path.dirname(dir_path)
    bool = os.path.exists(dir_path)
    if bool:
        pass
    else:
        os.makedirs(dir_path)


def matching(test_img, ref_blur, cali, table):
    diff = test_img - ref_blur
    diff[:, :, 0] = np.clip(
        (diff[:, :, 0] - cali.blue_range[0]) * cali.ratio, 0, cali.blue_bin - 1)
    diff[:, :, 1] = np.clip(
        (diff[:, :, 1] - cali.green_range[0]) * cali.ratio, 0, cali.green_bin - 1)
    diff[:, :, 2] = np.clip(
        (diff[:, :, 2] - cali.red_range[0]) * cali.ratio, 0, cali.green_bin - 1)
    diff = diff.astype(int)
    grad_img = table[diff[:, :, 0], diff[:, :, 1], diff[:, :, 2], :]
    return grad_img


def matching_v2(test_img, ref_blur, cali, table, blur_inverse):
    diff_temp1 = test_img - ref_blur
    diff_temp2 = diff_temp1 * blur_inverse
    diff_temp2[:, :, 0] = (diff_temp2[:, :, 0] -
                           cali.zeropoint[0]) / cali.lookscale[0]
    diff_temp2[:, :, 1] = (diff_temp2[:, :, 1] -
                           cali.zeropoint[1]) / cali.lookscale[1]
    diff_temp2[:, :, 2] = (diff_temp2[:, :, 2] -
                           cali.zeropoint[2]) / cali.lookscale[2]
    diff_temp3 = np.clip(diff_temp2, 0, 0.999)
    diff = (diff_temp3 * cali.bin_num).astype(int)
    grad_img = table[diff[:, :, 0], diff[:, :, 1], diff[:, :, 2], :]
    return grad_img


def show_depth(depth, figure_num):
    fig = plt.figure(figure_num)
    ax = fig.add_subplot(projection='3d')
    X = np.arange(0, depth.shape[1], 1) * 4.76 / 80
    Y = np.arange(0, depth.shape[0], 1) * 4.76 / 80
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, depth, cmap=cm.jet)


def contact_detection(raw_image, ref_blur, marker_mask, kernel):
    diff_img = np.max(np.abs(raw_image.astype(np.float32) - ref_blur), axis=2)
    contact_mask = (diff_img > 25).astype(np.uint8)  # *(1-marker_mask)
    contact_mask = cv2.dilate(contact_mask, kernel, iterations=1)
    contact_mask = cv2.erode(contact_mask, kernel, iterations=1)
    return contact_mask


def marker_detection(raw_image_blur):
    m, n = raw_image_blur.shape[1], raw_image_blur.shape[0]
    raw_image_blur = cv2.pyrDown(raw_image_blur).astype(np.float32)
    ref_blur = cv2.GaussianBlur(raw_image_blur, (25, 25), 0)
    diff = ref_blur - raw_image_blur
    diff *= 16.0
    diff[diff < 0.] = 0.
    diff[diff > 255.] = 255.
    mask_b = diff[:, :, 0] > 150
    mask_g = diff[:, :, 1] > 150
    mask_r = diff[:, :, 2] > 150
    mask = (mask_b * mask_g + mask_b * mask_r + mask_g * mask_r) > 0
    mask = cv2.resize(mask.astype(np.uint8), (m, n))
    return mask


def make_kernal(n, k_type):
    if k_type == 'circle':
        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    else:
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))
    return kernal


def Process_batch_files(files_path):
    # header_file = "./lump"
    subfolders = glob.glob(files_path + '/*')
    for folder in subfolders:  #
        file_ = folder + '/tactile128/'
        ref_file = file_ + "000.jpg"
        ref_img = cv2.imread(ref_file)
        if CROP:
            ref_img = ref_img[upleft_y:downright_y + 1, upleft_x:downright_x + 1]
        if COMPENSATE:
            ref_img = color_mean(ref_img)
        height = ref_img.shape[0]
        width = ref_img.shape[1]
        marker = cali.mask_marker(ref_img)
        keypoints = cali.find_dots(marker)
        marker_mask = cali.make_mask(ref_img, keypoints)
        marker_mask = cv2.dilate(marker_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        marker_image = np.dstack((marker_mask, marker_mask, marker_mask))
        ref_img = cv2.inpaint(ref_img, marker_mask, 5, cv2.INPAINT_TELEA)
        # ref_img = cali.crop_image(ref_img, pad)
        # if not os.path.exists(str(name) +'/depth/'):
        #     os.makedirs(str(name) +'/depth/')
        imgs_file = glob.glob(file_ + '/*.jpg')
        imgs_file.sort()
        progress_bar = tqdm(total=len(imgs_file), unit="num", unit_scale=True, desc="Processing " + str(file_))
        for img_file in imgs_file:
            test_img = cv2.imread(img_file)
            if CROP:
                test_img = test_img[upleft_y:downright_y + 1, upleft_x:downright_x + 1]
            red_mask = (ref_img[:, :, 2] > 12).astype(np.uint8)
            ref_blur = cv2.GaussianBlur(ref_img.astype(np.float32), (3, 3), 0) + 1
            blur_inverse = 1 + ((np.mean(ref_blur) / ref_blur) - 1) * 2
            # test_img = cali.crop_image(test_img, pad)
            test_img = cv2.GaussianBlur(test_img, (3, 3), 0)
            marker_mask = marker_detection(test_img)
            marker_mask = cv2.dilate(marker_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            grad_img2 = matching_v2(test_img, ref_blur, cali, table, blur_inverse)
            grad_img2[:, :, 0] = grad_img2[:, :, 0] * (1 - marker_mask) * red_mask
            grad_img2[:, :, 1] = grad_img2[:, :, 1] * (1 - marker_mask) * red_mask
            depth = fast_poisson(grad_img2[:, :, 0], grad_img2[:, :, 1])
            depth[depth < 0] = 0
            # print(np.max(depth))
            # plt.figure()
            # plt.imshow(depth)
            # plt.show()
            if 'sensor01' in img_file:
                depth_path = img_file.replace('sensor01', 'sensor01_depth')
            elif 'sensor02' in img_file:
                depth_path = img_file.replace('sensor02', 'sensor02_depth')
            elif 'sensor03' in img_file:
                depth_path = img_file.replace('sensor03', 'sensor03_depth')
            else:
                print('file path error ... ')
                break
            makedir(depth_path)
            cv2.imwrite(depth_path, (depth * 60).astype(np.uint8))
            progress_bar.update(1)
        progress_bar.close()


def Process_batch_images():
    pass


def Process_single_image(ref_img_path, tag_img_path, table2):
    # file_ = "./lump/test_003_001/tactile/"
    # ref_file = file_+"ref.jpg"
    ref_img = cv2.imread(ref_img_path)
    if CROP:
        ref_img = ref_img[upleft_y:downright_y + 1, upleft_x:downright_x + 1]
    if COMPENSATE:
        ref_img = color_mean(ref_img)
    # img_file = file_ + "070.jpg"
    fit_guassian = Fit_Guassian(plot=True)
    test_img = cv2.imread(tag_img_path)
    if CROP:
        test_img = test_img[upleft_y:downright_y + 1, upleft_x:downright_x + 1]
        img = test_img.copy()
    if COMPENSATE:
        test_img = color_mean(test_img)
    marker = cali.mask_marker(ref_img)
    keypoints = cali.find_dots(marker)
    marker_mask = cali.make_mask(ref_img, keypoints)
    marker_image = np.dstack((marker_mask, marker_mask, marker_mask))
    ref_img = cv2.inpaint(ref_img, marker_mask, 3, cv2.INPAINT_TELEA)
    red_mask = (ref_img[:, :, 2] > 12).astype(np.uint8)
    ref_blur = cv2.GaussianBlur(ref_img.astype(np.float32), (3, 3), 0) + 1
    blur_inverse = 1 + ((np.mean(ref_blur) / ref_blur) - 1) * 2
    # test_img = cali.crop_image(test_img, pad)
    test_img = cv2.GaussianBlur(test_img.astype(np.float32), (3, 3), 0)
    marker_mask = marker_detection(test_img)
    marker_mask = cv2.dilate(marker_mask, kernel1, iterations=1)
    contact_mask = contact_detection(test_img, ref_blur, marker_mask, kernel2)
    grad_img2 = matching_v2(test_img, ref_blur, cali, table2, blur_inverse)
    grad_img2[:, :, 0] = grad_img2[:, :, 0] * (1 - marker_mask) * red_mask
    grad_img2[:, :, 1] = grad_img2[:, :, 1] * (1 - marker_mask) * red_mask
    depth2 = fast_poisson(grad_img2[:, :, 0], grad_img2[:, :, 1])

    depth2[depth2 < 0] = 0

    # === 生成保存文件名 ===
    tag_name = os.path.splitext(os.path.basename(tag_img_path))[0]
    save_dir = 'E:/Calib_gu/tong6/test/depth'
    os.makedirs(save_dir, exist_ok=True)

    # ✅ 保存为 PNG 图片（无坐标轴）
    output_image_path = os.path.join(save_dir, f"{tag_name}.png")
    plt.figure()  # 创建新图像
    plt.imshow(depth2)  # 可加 colormap 美观显示
    plt.axis('off')  # 去除坐标轴
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭图像，防止内存泄漏

    # ✅ 保存为 .npy 文件
    output_npy_path = os.path.join(save_dir, f"{tag_name}.npy")
    np.save(output_npy_path, depth2)

    # plt.show()

    print(f"保存图片到: {output_image_path}")
    print(f"保存 .npy 文件到: {output_npy_path}")
    # print(1)
    # fit_guassian.fit_guassian_3d(depth2)

    # ✅ 释放内存，防止累积
    del ref_img, test_img, marker, depth2, grad_img2
    gc.collect()

    # ✅ 关闭 OpenCV 缓存（防止累积）
    cv2.ocl.setUseOpenCL(False)

    # ✅ 关闭所有 Matplotlib 窗口（防止窗口累积）
    plt.close('all')




def process_all_images(ref_img_path, input_folder, table):
    # 获取文件夹中所有的图片（支持 jpg、png 格式）
    image_paths = glob.glob(os.path.join(input_folder, '*.jpg')) + glob.glob(os.path.join(input_folder, '*.png'))

    # 按文件名排序，确保处理顺序
    image_paths.sort()

    print(f"检测到 {len(image_paths)} 张图片，开始处理...")

    for idx, tag_img_path in enumerate(image_paths):
        print(f"处理第 {idx + 1}/{len(image_paths)} 张图片: {tag_img_path}")

        try:
            # 调用函数进行处理
            Process_single_image(ref_img_path, tag_img_path, table)
            print(f"✅ 完成：{tag_img_path}")
        except Exception as e:
            print(f"❌ 处理失败: {tag_img_path}，原因：{e}")

    print("🎉 所有图片处理完成！")


if __name__ == '__main__':
    ## Params
    # pad = 20
    CROP = False
    batch_mode = True
    COMPENSATE = False
    cali = calibration()
    kernel1 = make_kernal(3, 'circle')
    kernel2 = make_kernal(25, 'circle')
    [upleft_x, upleft_y, downright_x, downright_y] = [1, 1, 1, 1]

    # table = np.load('Code/Calib/table/table_smooth_1.3_C3.npy')
    # Process_batch_files(files_path = 'Datasets/sensor01/')

    table = np.load('table_smooth.npy')

    ref_img_path = 'E:/Calib_gu/tong6/test/0.jpg'  # 参考图片路径
    input_folder = 'E:/Calib_gu/tong6/test'  # 包含待处理图片的文件夹

    process_all_images(ref_img_path, input_folder, table)


    # Process_batch_files(files_path = 'Datasets/sensor02/')
    # Process_single_image('E:/Calib_gu/calib3/128/00.jpg', 'E:/Calib_gu/calib3/128/test/45.jpg', table)
    # Process_single_image('E:/Calib_gu/calib3/128/00.jpg', 'E:/Calib_gu/caizen/frame_0011.jpg', table)

    # table = np.load('Code/Calib/table/table_smooth_1.5_C12.npy')
    # Process_batch_files(files_path = 'Datasets/sensor03/')


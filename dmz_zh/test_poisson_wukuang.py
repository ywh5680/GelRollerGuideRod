import os
import sys
import warnings
import numpy as np
from fast_poisson import fast_poisson
import cv2
from calibration import calibration

sys.path.append('/Code/Calib')


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





def Process_single_image(ref_img_path, tag_img_path, table2):
    
    ref_img = cv2.imread(ref_img_path)
    if CROP:
        ref_img = ref_img[upleft_y:downright_y + 1, upleft_x:downright_x + 1]
    if COMPENSATE:
        ref_img = color_mean(ref_img)
    

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
 
    test_img = cv2.GaussianBlur(test_img.astype(np.float32), (3, 3), 0)
    marker_mask = marker_detection(test_img)
    marker_mask = cv2.dilate(marker_mask, kernel1, iterations=1)
    contact_mask = contact_detection(test_img, ref_blur, marker_mask, kernel2)
    grad_img2 = matching_v2(test_img, ref_blur, cali, table2, blur_inverse)
    grad_img2[:, :, 0] = grad_img2[:, :, 0] * (1 - marker_mask) * red_mask
    grad_img2[:, :, 1] = grad_img2[:, :, 1] * (1 - marker_mask) * red_mask
    depth2 = fast_poisson(grad_img2[:, :, 0], grad_img2[:, :, 1])
    depth2[depth2 < 0] = 0
   
    print(depth2.max())
 


if __name__ == '__main__':
    
    CROP = False
    batch_mode = True
    COMPENSATE = False
    cali = calibration()
    kernel1 = make_kernal(3, 'circle')
    kernel2 = make_kernal(25, 'circle')
    [upleft_x, upleft_y, downright_x, downright_y] = [1, 1, 1, 1]
    table = np.load(r'D:\Users\Admin\Desktop\postgraduate\dmz\dmz_zh\Calib_gu\table_smooth.npy')

    Process_single_image(r'D:\Users\Admin\Desktop\postgraduate\dmz\dmz_zh\Calib_gu\tong6\0.jpg', r'D:\Users\Admin\Desktop\postgraduate\dmz\dmz_zh\Calib_gu\tong6\test\14.jpg', table)
 

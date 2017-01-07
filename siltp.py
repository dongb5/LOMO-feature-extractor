import cv2
import numpy as np

def SILTP4(img, R, tau):    

    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_pad = np.pad(img, R, 'edge')
    
    R_ = -1 * R
    img_u = img_pad[:2*R_, R:R_]
    img_d = img_pad[2*R:, R:R_]
    img_r = img_pad[R:R_, 2*R:]
    img_l = img_pad[R:R_, :2*R_]

    up_limit = (1 + tau) * img
    low_limit = (1 - tau) * img

    siltp = ((img_u < low_limit) + (img_u > up_limit) * 2) + \
            ((img_d < low_limit) + (img_d > up_limit) * 2) * 3 + \
            ((img_r < low_limit) + (img_r > up_limit) * 2) * (3 ** 2) + \
            ((img_l < low_limit) + (img_l > up_limit) * 2) * (3 ** 3) 
    
    return siltp

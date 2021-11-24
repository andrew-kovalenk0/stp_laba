from flask import Flask, render_template
from PIL import Image
import numpy as np
import time
import cv2


app = Flask(__name__)


def erode(img, kernel, er_img, bound):
    img_v = img.shape[0]
    img_h = img.shape[1]
    kl_v = kernel.shape[0]
    kl_h = kernel.shape[1]
    kl_cn_v = kl_v // 2
    kl_cn_h = kl_h // 2
    for i in range(kl_cn_v, img_v - (kl_v - kl_cn_v - 1)):
        for j in range(kl_cn_h, img_h - (kl_h - kl_cn_h - 1)):
            er_img[i, j] = np.min(img[i - kl_cn_v: i + kl_v - kl_cn_v, j - kl_cn_h: j + kl_h - kl_cn_h][kernel])

    # UP BOUNDARY
    for i in range(kl_cn_v):
        for j in range(img_h):
            if kl_cn_h - 1 < j < img_h - (kl_h - kl_cn_h - 1):
                er_img[i, j] = max(np.max(img[: i + kl_v - kl_cn_v, j - kl_cn_h: j + kl_h - kl_cn_h]
                                          [kernel[kl_cn_v - i:, :]]), bound)
            elif kl_cn_h - 1 >= j:
                er_img[i, j] = max(np.max(img[: i + kl_v - kl_cn_v, : j + kl_h - kl_cn_h]
                                          [kernel[kl_cn_v - i:, kl_cn_h - j:]]), bound)
            elif j >= img_h - (kl_h - kl_cn_h - 1):
                er_img[i, j] = max(np.max(img[: i + kl_v - kl_cn_v, j - kl_cn_h:]
                                          [kernel[kl_cn_v - i:, :img_h + kl_cn_h - j]]), bound)

    # DOWN BOUNDARY
    for i in range(img_v - (kl_v - kl_cn_v - 1), img_v):
        for j in range(img_h):
            if kl_cn_h - 1 < j < img_h - (kl_h - kl_cn_h - 1):
                er_img[i, j] = max(np.max(img[i - kl_cn_v:, j - kl_cn_h: j + kl_h - kl_cn_h]
                                          [kernel[:kl_cn_v + img_v - i, :]]), bound)
            elif kl_cn_h - 1 >= j:
                er_img[i, j] = max(np.max(img[i - kl_cn_v:, :j + kl_h - kl_cn_h]
                                          [kernel[:kl_cn_v + img_v - i, kl_cn_h - j:]]), bound)

            elif j >= img_h - (kl_h - kl_cn_h - 1):
                er_img[i, j] = max(np.max(img[i - kl_cn_v:, j - kl_cn_h:]
                                          [kernel[:kl_cn_v + img_v - i, :img_h + kl_cn_h - j]]), bound)

    # LEFT BOUNDARY
    for i in range(kl_cn_v):
        for j in range(kl_cn_h, img_v - (kl_v - kl_cn_v - 1)):
            er_img[j, i] = max(np.max(img[j - kl_cn_h: j + kl_h - kl_cn_h, :kl_v - kl_cn_v + i]
                                      [kernel[:, kl_cn_v - i:]]), bound)

    # RIGHT BOUNDARY
    for i in range(img_h - (kl_h - kl_cn_h - 1), img_h):
        for j in range(kl_cn_h, img_v - (kl_v - kl_cn_v - 1)):
            er_img[j, i] = max(np.max(img[j - kl_cn_h: j + kl_h - kl_cn_h, i - kl_cn_h:]
                                      [kernel[:, :kl_h - kl_cn_h + img_h - i - 1]]), bound)

    return er_img


def dilate(img, kernel, dil_img, bound):
    img_v = img.shape[0]
    img_h = img.shape[1]
    kl_v = kernel.shape[0]
    kl_h = kernel.shape[1]
    kl_cn_v = kl_v // 2
    kl_cn_h = kl_h // 2
    for i in range(kl_cn_v, img_v - (kl_v - kl_cn_v - 1)):
        for j in range(kl_cn_h, img_h - (kl_h - kl_cn_h - 1)):
            dil_img[i, j] = np.max(img[i - kl_cn_v: i + kl_v - kl_cn_v, j - kl_cn_h: j + kl_h - kl_cn_h] * kernel)

    # UP BOUNDARY
    for i in range(kl_cn_v):
        for j in range(img_h):
            if kl_cn_h - 1 < j < img_h - (kl_h - kl_cn_h - 1):
                dil_img[i, j] = max(np.max(img[: i + kl_v - kl_cn_v, j - kl_cn_h: j + kl_h - kl_cn_h] *
                                           kernel[kl_cn_v - i:, :]), bound)
            elif kl_cn_h - 1 >= j:
                dil_img[i, j] = max(np.max(img[: i + kl_v - kl_cn_v, : j + kl_h - kl_cn_h] *
                                           kernel[kl_cn_v - i:, kl_cn_h - j:]), bound)
            elif j >= img_h - (kl_h - kl_cn_h - 1):
                dil_img[i, j] = max(np.max(img[: i + kl_v - kl_cn_v, j - kl_cn_h:] *
                                           kernel[kl_cn_v - i:, :img_h + kl_cn_h - j]), bound)

    # DOWN BOUNDARY
    for i in range(img_v - (kl_v - kl_cn_v - 1), img_v):
        for j in range(img_h):
            if kl_cn_h - 1 < j < img_h - (kl_h - kl_cn_h - 1):
                dil_img[i, j] = max(np.max(img[i - kl_cn_v:, j - kl_cn_h: j + kl_h - kl_cn_h] *
                                           kernel[:kl_cn_v + img_v - i, :]), bound)
            elif kl_cn_h - 1 >= j:
                dil_img[i, j] = max(np.max(img[i - kl_cn_v:, :j + kl_h - kl_cn_h] *
                                           kernel[:kl_cn_v + img_v - i, kl_cn_h - j:]), bound)

            elif j >= img_h - (kl_h - kl_cn_h - 1):
                dil_img[i, j] = max(np.max(img[i - kl_cn_v:, j - kl_cn_h:] *
                                           kernel[:kl_cn_v + img_v - i, :img_h + kl_cn_h - j]), bound)

    # LEFT BOUNDARY
    for i in range(kl_cn_v):
        for j in range(kl_cn_h, img_v - (kl_v - kl_cn_v - 1)):
            dil_img[j, i] = max(np.max(img[j - kl_cn_h: j + kl_h - kl_cn_h, :kl_v - kl_cn_v + i] *
                                       kernel[:, kl_cn_v - i:]), bound)

    # RIGHT BOUNDARY
    for i in range(img_h - (kl_h - kl_cn_h - 1), img_h):
        for j in range(kl_cn_h, img_v - (kl_v - kl_cn_v - 1)):
            dil_img[j, i] = max(np.max(img[j - kl_cn_h: j + kl_h - kl_cn_h, i - kl_cn_h:] *
                                       kernel[:, :kl_h - kl_cn_h + img_h - i - 1]), bound)

    return dil_img


@app.route('/')
def hello_world():
    image = cv2.imread('static/image.png', 0)
    core = np.ones((3, 3)).astype('bool')

    dilate_image_own = image.copy()
    dilate_start_time_own = time.time_ns()
    dilate_image_own = dilate(image.copy(), core, dilate_image_own, 0)
    dilate_time_own = (time.time_ns() - dilate_start_time_own) / 1000
    Image.fromarray(dilate_image_own).save("static/dilate_image_own.png", "PNG")

    dilate_start_time_cv = time.time_ns()
    dilate_image_cv = cv2.dilate(image, core.astype(np.uint8))
    dilate_time_cv = (time.time_ns() - dilate_start_time_cv) / 1000
    Image.fromarray(dilate_image_cv).save("static/dilate_image_cv.png", "PNG")

    erode_image_own = image.copy()
    erode_start_time_own = time.time_ns()
    erode_image_own = erode(image.copy(), core, erode_image_own, 0)
    erode_time_own = (time.time_ns() - erode_start_time_own) / 1000
    Image.fromarray(erode_image_own).save("static/erode_image_own.png", "PNG")

    erode_start_time_cv = time.time_ns()
    erode_image_cv = cv2.erode(image, core.astype(np.uint8))
    erode_time_cv = (time.time_ns() - erode_start_time_cv) / 1000
    Image.fromarray(erode_image_cv).save("static/erode_image_cv.png", "PNG")

    return render_template('main_page.html', dilate_time_cv=dilate_time_cv, dilate_time_own=dilate_time_own,
                           erode_time_cv=erode_time_cv, erode_time_own=erode_time_own)


if __name__ == '__main__':
    app.run()

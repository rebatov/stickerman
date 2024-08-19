from autotrace import Bitmap, VectorFormat
import cv2 as cv
import sys
import numpy as np


# Showing image on resized window
def show_image(name, image):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, 600, 600)
    cv.imshow(name, image)
    cv.waitKey(0)

# Adding a border as dilation will add pixels around the image
def add_uniform_border(img):
    return cv.copyMakeBorder(img, 250,250,250,250, cv.BORDER_CONSTANT, value=[-255, 0, 0, 0])

def convert_binary_to_rgb(binary):
    return cv.cvtColor(binary, cv.COLOR_GRAY2RGB)

def convert_rgb_to_binary(binary):
    return cv.cvtColor(binary, cv.COLOR_BGR2GRAY)

# Identifying continuous or discrete shape
def is_continuous_shape(alpha):
    contours, hierarchy = cv.findContours(alpha, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    rgb = convert_binary_to_rgb(alpha)
    cv.drawContours(rgb, contours, -1, (255,255,255), thickness=5)
    if len(contours) > 1:
        print('Detected discrete shape')
        return False
    print('Detected continuous shape')
    return True
    # show_image('shape', rgb)

# Applying dilation, choose kernel type as you want but elliptical worked best
def morph_image(img, radius=200):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (radius, radius))
    dilation = cv.dilate(img, kernel, iterations=1)
    return dilation


# Layering the dilated image onto the original image
def layer_image(img, dilate):
    layer = np.zeros(img.shape, dtype=np.uint8)
    alpha = np.expand_dims(img[:, :, 3], 2)
    alpha = np.repeat(alpha, 3, 2)
    alpha = alpha / 255

    layer[dilate == 255] = (255, 255, 255, 255)
    layer[:, :, 0:3] = layer[:, :, 0:3] * (1 - alpha) + alpha * img[:, :, 0:3]

    return layer

# Filling the largest encompassing shape
def fill_contours(dilate):
    contours, hierarchy = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    dilate_rgb = convert_binary_to_rgb(dilate)
    c = max(contours, key = cv.contourArea)
    cv.drawContours(dilate_rgb, [c], -1, (255,255,255), thickness=-1)
    dilate_mask = convert_rgb_to_binary(dilate_rgb)
    return dilate_mask


# Eroding out the excess area because of chosen large kernel radius
def erode_image(filled):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (180, 180))
    erosion = cv.erode(filled, kernel, iterations=1)
    return erosion


def create_vector_outline(img):
    bitmap = Bitmap(np.asarray(img))
    vector = bitmap.trace()
    svg = vector.encode(VectorFormat.SVG)
    print(svg)


def main():
    img = sys.argv[1]
    img_color = cv.imread(img, cv.IMREAD_UNCHANGED)
    img_border = add_uniform_border(img_color)
    b, g, r, alpha = cv.split(img_border)
    continuos_shape = is_continuous_shape(alpha)
    if not(continuos_shape):
        img_dilate = morph_image(alpha, radius=200)
        img_filled = fill_contours(img_dilate)
        img_erosion = erode_image(img_filled)
        img_layer = layer_image(img_border, img_erosion)
        create_vector_outline(convert_binary_to_rgb(img_erosion))
    else:
        img_dilate = morph_image(alpha, radius=50)
        img_layer = layer_image(img_border, img_dilate)
        create_vector_outline(convert_binary_to_rgb(img_dilate))
    show_image('layer', img_layer)


if __name__ == "__main__":
    main()

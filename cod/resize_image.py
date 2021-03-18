import sys
import cv2 as cv
import numpy as np
import copy

from parameters import *
from select_path import *

import pdb


def compute_energy(img):
    """
    calculeaza energia la fiecare pixel pe baza gradientului
    :param img: imaginea initiala
    :return:E - energia
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    grad_x = cv.Sobel(src=gray, ddepth=cv.CV_64F, dx=1, dy=0, ksize=3, scale=1, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(src=gray, ddepth=cv.CV_64F, dx=0, dy=1, ksize=3, scale=1, borderType=cv.BORDER_DEFAULT)

    E = np.abs(grad_x) + np.abs(grad_y)

    return E


def show_path(img, path, color):
    new_image = img.copy()
    for row, col in path:
        new_image[row, col] = color

    E = compute_energy(img)
    new_image_E = img.copy()
    new_image_E[:,:,0] = E.copy()
    new_image_E[:,:,1] = E.copy()
    new_image_E[:,:,2] = E.copy()

    for row, col in path:
        new_image_E[row, col] = color
    cv.imshow('path img', np.uint8(new_image))
    cv.imshow('path E', np.uint8(new_image_E))
    cv.waitKey(1000)


def delete_path(img, path):
    """
    elimina drumul vertical din imagine
    :param img: imaginea initiala
    :path - drumul vertical
    return: updated_img - imaginea initiala din care s-a eliminat drumul vertical
    """
    updated_img = np.zeros((img.shape[0], img.shape[1] - 1, img.shape[2]), np.uint8)
    for i in range(img.shape[0]):
        col = path[i][1]
        # copiem partea din stanga
        updated_img[i, :col] = img[i, :col].copy()
        # copiem partea din dreapta
        updated_img[i, col:] = img[i, col+1:].copy()
        
    return updated_img


def decrease_width(params: Parameters, num_pixels):
    img = params.image.copy()  # copiaza imaginea originala
    for i in range(num_pixels):
        print('Eliminam drumul vertical numarul %i dintr-un total de %d.' % (i+1, num_pixels))

        # calculeaza energia dupa ecuatia (1) din articol                
        E = compute_energy(img)
        path = select_path(E, params.method_select_path)

        if params.show_path:
            show_path(img, path, params.color_path)
        img = delete_path(img, path)

    cv.destroyAllWindows()
    return img


def decrease_height(params: Parameters, num_pixels):
    img = params.image.copy()
    # rotate the image 90 degrees clockwise so we can simulate a decrease in height using the decrease_width method on
    # the rotated image and rotating it back after we are done
    img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)

    for i in range(num_pixels):
        print('Eliminam drumul vertical numarul %i dintr-un total de %d.' % (i+1, num_pixels))

        # calculeaza energia dupa ecuatia (1) din articol
        E = compute_energy(img)
        path = select_path(E, params.method_select_path)
        if params.show_path:
            show_path(img, path, params.color_path)
        img = delete_path(img, path)

    # rotate the image back to original shape
    img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)

    cv.destroyAllWindows()
    return img


def amplify_content(params: Parameters):
    # compute the resized image's height/width
    amplified_width = int(params.factor_amplification * params.image.shape[1])
    amplified_height = int(params.factor_amplification * params.image.shape[0])

    # number of pixels to be added on each dimension (= number of paths to be deleted on each dimension)
    num_pixels_width = amplified_width - params.image.shape[1]
    num_pixels_height = amplified_height - params.image.shape[0]

    # back-up for the original image
    img_copy = params.image.copy()

    # resize the image
    params.image = cv.resize(params.image, (amplified_width, amplified_height))

    # get the image to original width
    params.image = decrease_width(params, num_pixels_width)

    # get the image to original height
    params.image = decrease_height(params, num_pixels_height)

    # recover the original image and return the new image with amplified content
    img_copy, params.image = params.image.copy(), img_copy.copy()

    return img_copy


def decrease_width_modified(params: Parameters, w, h, x0, y0):
    img = params.image.copy()  # copiaza imaginea originala
    for i in range(w):
        print('Eliminam drumul vertical numarul %i dintr-un total de %d.' % (i+1, w))

        # calculeaza energia dupa ecuatia (1) din articol
        E = compute_energy(img)

        # make sure that all paths chosen will go through the region of interest by making the region of interest
        # contain only huge negative values
        maximum = E.shape[0] * np.max(E)
        E[y0: y0 + h, x0: x0 + w - i] -= maximum

        path = select_path(E, params.method_select_path)
        if params.show_path:
            show_path(img, path, params.color_path)
        img = delete_path(img, path)

    cv.destroyAllWindows()
    return img


def decrease_height_modified(params: Parameters, w, h, x0, y0):
    # x0 is the column and y0 is the line
    img = params.image.copy()
    # rotate the image 90 degrees clockwise so we can simulate a decrease in height using the decrease_width method on
    # the rotated image and rotating it back after we are done
    img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)

    # rotate the coordinates of the top-left corner. It becomes the top-right corner of the roi in the rotated image
    # x0_rot = column in the rotated image
    x0_rot = img.shape[1] - y0 - 1
    # get the column of the top-left corner
    x0_rot = x0_rot - h + 1
    # y0_rot = line in the rotated image
    y0_rot = x0

    for i in range(h):
        print('Eliminam drumul vertical numarul %i dintr-un total de %d.' % (i+1, h))

        # calculeaza energia dupa ecuatia (1) din articol
        E = compute_energy(img)

        # make sure that all paths chosen will go through the region of interest by making the region of interest
        # contain only huge negative values
        maximum = E.shape[0] * np.max(E)
        E[y0_rot: y0_rot + w - 1, x0_rot: x0_rot + h - i] -= maximum

        path = select_path(E, params.method_select_path)
        if params.show_path:
            show_path(img, path, params.color_path)
        img = delete_path(img, path)

    # rotate the image back to original shape
    img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)

    cv.destroyAllWindows()
    return img


def delete_object(params: Parameters, x0, y0, w, h):
    # if the roi has a larger (or equal) height than width, we decrease the width of the image,
    # otherwise we decrease the height
    if w <= h:
        img = decrease_width_modified(params, w, h, x0, y0)
    else:
        img = decrease_height_modified(params, w, h, x0, y0)
    return img


def resize_image(params: Parameters):

    if params.resize_option == 'micsoreazaLatime':
        # redimensioneaza imaginea pe latime
        resized_image = decrease_width(params, params.num_pixels_width)
        return resized_image

    elif params.resize_option == 'micsoreazaInaltime':
        # shrinks the height of the image
        resized_image = decrease_height(params, params.num_pixel_height)
        return resized_image
    
    elif params.resize_option == 'amplificaContinut':
        # amplifies the content in the image
        resized_image = amplify_content(params)
        return resized_image

    elif params.resize_option == 'eliminaObiect':
        # draw a bounding and remove everything inside the bounding box from the image
        img = params.image.copy()
        img = np.uint8(img)
        roi = cv.selectROI(img=img)
        cv.destroyAllWindows()

        resized_image = delete_object(params, roi[0], roi[1], roi[2], roi[3])

        return resized_image
    else:
        print('The option is not valid!')
        sys.exit(-1)
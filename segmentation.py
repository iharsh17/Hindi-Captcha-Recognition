import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image, ImageOps
import scipy.ndimage
import os


def character_segmentation(image, plot_images=False):
    def show_image(image):
        cv2_imshow(image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Extra feature
    # In testing
    def vertical_to_horizontal(image):
        # get hough lines and if the slope of longest line>3 rotate to make horizontal
        img = image.copy()
        # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 100, apertureSize=3)
        # cv2.imshow('edges',edges)
        # cv2.waitKey(0)
        minLineLength = 20
        maxLineGap = 10
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            15,
            minLineLength=minLineLength,
            maxLineGap=maxLineGap,
        )
        lines = sorted(
            lines,
            key=lambda x: ((x[0][1] - x[0][3]) ** 2 + (x[0][0] - x[0][2]) ** 2)
            ** (0.5),
        )
        cords = lines[-1]
        x1, y1, x2, y2 = cords[0]
        angle = (y2 - y1) / (x2 - x1)
        if angle > 3:
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        elif angle < -3:
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    # Extra feature
    def getSkewAngle(image):
        # Prep image, copy, convert to gray scale, blur, and threshold
        dilate = image.copy()
        contours, hierarchy = cv2.findContours(
            dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largestContour = contours[0]
        minAreaRect = cv2.minAreaRect(largestContour)
        angle = minAreaRect[-1]
        return -1.0 * angle

    # Extra feature
    def rotateImage(image, angle: float):
        # rotate according to angle obtained
        newImage = image.copy()
        (h, w) = newImage.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        newImage = cv2.warpAffine(
            newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return newImage

    # Deskew image
    # Extra feature
    def deskew(image):
        # parent_function
        angle = getSkewAngle(image)
        return rotateImage(image, -1.0 * angle)

    # Extra feature
    def repair_noise(dilated):
        dilated = scipy.ndimage.median_filter(dilated, (5, 1))
        dilated = scipy.ndimage.median_filter(dilated, (1, 5))
        dilated = scipy.ndimage.median_filter(dilated, (5, 1))
        dilated = scipy.ndimage.median_filter(dilated, (1, 5))
        return dilated

    def remove_white(image):
        # Load image, grayscale, Gaussian blur, Otsu's threshold
        original = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (25, 25), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # Perform morph operations, first open to remove noise, then close to combine
        noise = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise, iterations=3)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel, iterations=3)

        # Find enclosing boundingbox and crop ROI
        coords = cv2.findNonZero(close)
        x, y, w, h = cv2.boundingRect(coords)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
        crop = original[y : y + h, x : x + w]

        # cv2.imshow('thresh', thresh)
        # cv2.imshow('close', close)
        # cv2.imshow('image', image)
        # cv2.imshow('crop', crop)
        # cv2.waitKey()
        crop = cv2.resize(crop, (image.shape[1], image.shape[0]))
        return crop

    def getMeanArea(contours):
        meanArea = 0
        for contour in contours:
            meanArea += cv2.contourArea(contour)
        meanArea = (meanArea) / len(contours)
        return meanArea

    characters = []
    # input is binary image
    # image = cv2.imread(image_path)
    image = cv2.resize(image, (int(1024 * (image.shape[1] / image.shape[0])), 1024))
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
    image = remove_white(image)

    # show_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )  # threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.erode(thresh, kernel, iterations=1)
    dilated = repair_noise(dilated)
    dilated_orig = dilated.copy()

    # show_image(dilated)
    # dilated=deskew(dilated) #(EXTRA FEATURE)
    # dilated=vertical_to_horizontal(dilated) #(EXTRA FEATURE)

    val = np.count_nonzero(dilated, axis=1).argmax()
    delta = 80
    dilated[val - delta // 2 : val + delta + 1, :] = 0

    (contours, hierarchy) = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    coords = []
    count = 0
    meanArea = getMeanArea(contours)
    meanArea = getMeanArea(contours)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 0.05 * meanArea:
            if h > 300 and w > 300:
                if w / h < 2:
                    coords.append((x, y, w, h))
                    count = count + 1
                else:
                    coords.append((x, y, w // 2, h))
                    coords.append((x + w // 2, y, w // 2, h))
                    count = count + 2
    coords = sorted(coords, key=lambda x: x[0])
    for cor in coords:
        [x, y, w, h] = cor
        t = dilated_orig[max(0, y - delta) : y + h, x : x + w]
        characters.append(t)
    if plot_images:
        print("number of char in image:", count)
        w = 10
        h = 10
        fig = plt.figure(figsize=(8, 8))
        for i in range(1, count + 1):
            img = characters[i - 1]
            fig.add_subplot(1, count, i)
            plt.imshow(img, cmap="gray")
        plt.show()
    return characters

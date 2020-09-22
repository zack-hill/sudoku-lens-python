import cv2 as cv2
import math as math
import numpy as np
import pytesseract as tess
from board import Board

CELL_BORDER_BUFFER = 3


def int_try_parse(value):
    try:
        return int(value), True
    except ValueError:
        return value, False


def calc_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_largest_contour(img, output=None):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = None
    largest_contour_area = 0
    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > largest_contour_area):
            largest_contour = contour
            largest_contour_area = area
        if output is not None:
            cv2.drawContours(output, contours, index, (255, 255, 0), 2)
    return largest_contour, largest_contour_area


def contour_touches_border(contour, width, height):
    for point in (x[0] for x in contour):
        px, py = point[0], point[1]
        if px <= CELL_BORDER_BUFFER or px >= width - CELL_BORDER_BUFFER:
            return True
        if py <= CELL_BORDER_BUFFER or py >= height - CELL_BORDER_BUFFER:
            return True
    return False


def get_contour_corners(contour):
    moments = cv2.moments(contour)
    center = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])
    cx, cy = center[0], center[1]
    corners = [None, None, None, None]
    max_distance = [0, 0, 0, 0]

    for point in (x[0] for x in contour):
        px, py = point[0], point[1]
        distance = calc_distance(center, point)
        if px > cx and py < cy:
            quadrant = 1
        elif px < cx and py < cy:
            quadrant = 0
        elif px < cx and py > cy:
            quadrant = 3
        else:
            quadrant = 2
        if distance > max_distance[quadrant]:
            max_distance[quadrant] = distance
            corners[quadrant] = point
    return corners


def warp_mat(mat, corners):
    new_height = int(max(calc_distance(corners[1], corners[2]), calc_distance(corners[3], corners[0])))
    new_width = int(max(calc_distance(corners[0], corners[1]), calc_distance(corners[2], corners[3])))

    src = np.array([
            corners[0],
            corners[1],
            corners[2],
            corners[3],], dtype = "float32")
    dst = np.array([
            [0, 0],
            [new_width - 1, 0],
            [new_width - 1, new_height - 1],
            [0, new_height - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(mat, M, (new_width, new_height))
    return warped, M


def get_cell_value_contour(cell_mat):
    cell_height = cell_mat.shape[0]
    cell_width = cell_mat.shape[1]
    cell_area = cell_height * cell_width
    min_area = cell_area * .02
    largest_contour = None
    largest_contour_area = 0
    contours, _ = cv2.findContours(cell_mat, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if not contour_touches_border(contour, cell_width, cell_height):
            area = cv2.contourArea(contour)
            if area > largest_contour_area and area > min_area:
                largest_contour = contour
                largest_contour_area = area
    return largest_contour


def get_roi_from_contour(contour, img):
    x, y, w, h = cv2.boundingRect(contour)
    dim = 0
    x -= dim
    y -= dim
    w += dim * 2
    h += dim * 2
    return img[y:y+h, x:x+w]


def read_board(path):
    img = cv2.imread(path)
    values = np.zeros(shape=(9, 9), dtype=np.int)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (3, 3))
    thresh = cv2.bitwise_not(blurred)
    thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 47, -7)

    contour_output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    largest_contour, _ = get_largest_contour(thresh, contour_output)
    corners = get_contour_corners(largest_contour)
    if any(x is None for x in corners):
        return values

    cv2.line(img, tuple(corners[0]), tuple(corners[1]), (255, 0, 255), 1)
    cv2.line(img, tuple(corners[1]), tuple(corners[2]), (255, 0, 255), 1)
    cv2.line(img, tuple(corners[2]), tuple(corners[3]), (255, 0, 255), 1)
    cv2.line(img, tuple(corners[3]), tuple(corners[0]), (255, 0, 255), 1)
    cv2.line(img, tuple(corners[0]), tuple(corners[2]), (255, 0, 255), 1)
    cv2.line(img, tuple(corners[1]), tuple(corners[3]), (255, 0, 255), 1)

    warped, matrix = warp_mat(thresh, corners)
    warped = cv2.bitwise_not(warped)

    cell_width = warped.shape[1] / 9.0
    cell_height = warped.shape[0] / 9.0
    threads = []
    for y in range(9):
        for x in range(9):
            x1 = int(x * cell_width)
            y1 = int(y * cell_height)
            x2 = x1 + int(cell_width)
            y2 = y1 + int(cell_height)
            cell_mat = warped[y1:y2, x1:x2]        
            cell_mat = cv2.bitwise_not(cell_mat)
            value_contour = get_cell_value_contour(cell_mat)
            if value_contour is not None:
                vx, vy, vw, vh = cv2.boundingRect(value_contour)
                roi = get_roi_from_contour(value_contour, cell_mat)
                roi = cv2.bitwise_not(roi)

                scale_percent = 32.0 / roi.shape[0]
                width = int(roi.shape[1] * scale_percent)
                height = int(roi.shape[0] * scale_percent)
                roi = cv2.resize(roi, (width, height), interpolation=cv2.INTER_AREA)

                border_size = 10
                roi = cv2.copyMakeBorder(
                    roi,
                    top=border_size,
                    bottom=border_size,
                    left=border_size,
                    right=border_size,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[255, 255, 255])
                string = tess.image_to_string(roi, lang='digits', config='--oem 1 --psm 10 -c tessedit_char_whitelist=123456789')
                value, success = int_try_parse(string)
                if success:
                    values[x][y] = value

    return Board(values)

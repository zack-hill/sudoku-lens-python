import cv2 as cv2
import math as math
import numpy as np
import pytesseract as tess

MIN_IMAGE_SIZE = 450
MAX_IMAGE_SIZE = 900
CELL_BORDER_BUFFER = 3

def int_try_parse(value):
    try:
        return int(value), True
    except ValueError:
        return value, False

def calc_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_largest_contour(img, output=None):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    return warped

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
    return img[y:y+h, x:x+w]

original = cv2.imread('boards/board_21.jpg')

height = original.shape[0]
width = original.shape[1]
scaled = original
if height > MAX_IMAGE_SIZE or width > MAX_IMAGE_SIZE:
    scale = max(height, width) / MAX_IMAGE_SIZE
    scaled = cv2.resize(original, (int(width / scale), int(height / scale)), 0, 0, cv2.INTER_AREA)
elif height < MIN_IMAGE_SIZE or width < MIN_IMAGE_SIZE:
    scale = min(height, width) / MAX_IMAGE_SIZE
    scaled = cv2.resize(original, (int(width / scale), int(height / scale)), 0, 0, cv2.INTER_CUBIC)

gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
denoised = cv2.blur(gray, (3, 3))
denoised = cv2.fastNlMeansDenoising(denoised, 3, 9, 21)
thresholded = cv2.bitwise_not(denoised)
thresholded = cv2.adaptiveThreshold(thresholded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -7)

contour_output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
largest_contour, area = get_largest_contour(thresholded, contour_output)
corners = get_contour_corners(largest_contour)
warped = warp_mat(thresholded, corners)
warped = cv2.bitwise_not(warped)
output = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

cell_width = warped.shape[1] / 9.0
cell_height = warped.shape[0] / 9.0
cell_area = cell_width * cell_height
for y in range(9):
    for x in range(9):
        x1 = int(x * cell_width)
        y1 = int(y * cell_height)
        x2 = x1 + int(cell_width)
        y2 = y1 + int(cell_height)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0))
        cell_mat = warped[y1:y2, x1:x2]        
        cell_mat = cv2.bitwise_not(cell_mat)
        value_contour = get_cell_value_contour(cell_mat)
        if value_contour is not None:
            vx, vy, vw, vh = cv2.boundingRect(value_contour)
            cv2.rectangle(output, (vx + x1, vy + y1), (vx + x1 + vw, vy + y1 + vh), (255, 0, 0), 2)
            roi = get_roi_from_contour(value_contour, cell_mat)
            roi = cv2.bitwise_not(roi)
            string = tess.image_to_string(roi, lang='eng', config='--oem 0 --psm 10 -c tessedit_char_whitelist=123456789')
            value, success = int_try_parse(string)
            if success:
                print(value, end='')
                cv2.putText(output, str(value), (x1 + 7, y1 + int(cell_height) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
            #else:
            #    cv2.imshow(str(x) + ',' + str(y) + '  ' + str(value), roi)

#cv2.imshow('Gray', gray)
#cv2.imshow('Denoised', denoised)
#cv2.imshow('Thresholded', thresholded)
#cv2.imshow('Contours', contour_output)
cv2.imshow('Output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

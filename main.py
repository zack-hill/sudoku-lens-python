import cv2 as cv2
import math as math
import numpy as np
import pytesseract as tess

MIN_IMAGE_SIZE = 450
MAX_IMAGE_SIZE = 900

original = cv2.imread('boards/board_10.jpg')

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
thresholded = cv2.adaptiveThreshold(thresholded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -2)
#canny = cv2.Canny(thresholded, 65, 130, apertureSize=3, L2gradient=True)
contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
largest_contour = None
largest_contour_area = 0
for index, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if (area > largest_contour_area):
        largest_contour = contour
        largest_contour_area = area
    cv2.drawContours(contour_output, contours, index, (255, 255, 0), 1)

moments = cv2.moments(largest_contour)
center = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])
cx, cy = center[0], center[1]
corners = [None, None, None, None]
max_distance = [0, 0, 0, 0]

def calc_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

for point in (x[0] for x in largest_contour):
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
warped = cv2.warpPerspective(denoised, M, (new_width, new_height))
#warped_thresh = cv2.bitwise_not(warped)
warped_thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 4)

def int_try_parse(value):
    try:
        return int(value), True
    except ValueError:
        return value, False

cell_width = new_width / 9
cell_height = new_height / 9
hor_padding = cell_width / 10
ver_padding = cell_height / 15
for y in range(9):
    for x in range(9):
        x1 = int(x * cell_width + hor_padding)
        y1 = int(y * cell_height + ver_padding)
        x2 = x1 + int(cell_width - hor_padding * 2)
        y2 = y1 + int(cell_height - ver_padding * 2)
        cell_mat = warped_thresh[y1:y2, x1:x2]
        config = ("--oem 1 --psm 10")
        string = ''
        string = tess.image_to_string(cell_mat, config=config)
        value, success = int_try_parse(string)
        #cv2.imshow(str(x) + ' ' + str(y), cell_mat)
        if success:
            print(value, end='')
        else:
            print(' ', end='')
            #cv2.imshow(str(x) + ' ' + str(y), cell_mat)
    print('')


#cv2.imshow('Gray', gray)
#cv2.imshow('Denoised', denoised)
#cv2.imshow('Thresholded', thresholded)
#cv2.imshow('Canny', canny)
#cv2.imshow('Contours', contour_output)
cv2.imshow('Warped', warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
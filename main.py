import cv2 as cv2

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

cv2.imshow('Gray', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
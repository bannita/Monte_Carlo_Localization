import cv2
import numpy as np

img = cv2.imread("map/map_clean.jpg", 0)

_, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV) #convert image into black&white

kernel = np.ones((3, 3), np.uint8)
binary = cv2.dilate(binary, kernel, iterations=1) #make walls thicker

grid_map = (binary > 0).astype(int) #convert image into binary

#add outer border walls
grid_map[0, :] = 1
grid_map[-1, :] = 1
grid_map[:, 0] = 1
grid_map[:, -1] = 1

display_map = (grid_map * 255).astype(np.uint8) #convert back for visualization

cv2.imshow("Original", img)
cv2.imshow("Grid Map", display_map)

cv2.imwrite("map/map_processed.png", display_map)

cv2.waitKey(0)
cv2.destroyAllWindows()

print(grid_map.shape)
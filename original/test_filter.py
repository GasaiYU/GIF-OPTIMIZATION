import numpy as np
import cv2
import filter

if __name__ == "__main__":
    bgr = cv2.imread("./cat.png")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    img = list(img)
    gif = filter.GuidedFilter(img)
    img = gif.filt(img)
    img = np.array(img)
    cv2.imwrite("./filter.png", img)
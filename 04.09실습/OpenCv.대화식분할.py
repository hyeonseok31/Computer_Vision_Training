import skimage
import numpy as np
import cv2
from matplotlib import pyplot as pyplot

src = skimage.data.coffee()

mask = np.zeros(src.shape[:2], np.uint8)
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)
iterCount = 1
mode = cv2.GC_INIT_WITH_RECT

rc = cv2.selectROI(src)

cv2.grabCut(src, mask, rc, bgdModel, fgdModel, iterCount, mode)

mask2 = np.where((mask == 0) | (mask ==2), 0, 1).astype('uint8')
dst = src * mask2[:, :, np.newaxis]

cv2.imshow('dst', dst)
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Load image
img = cv.imread('../Resources/Photos/cats.webp')
cv.imshow('Original Cats', img)

# Create blank mask
blank = np.zeros(img.shape[:2], dtype='uint8')

# Circle mask (centered)
mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 120, 255, -1)

# Apply mask
masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Masked Cats', masked)

# Colour histogram with labels and background color
plt.figure(figsize=(8,4), facecolor='lightyellow')   # set figure background
plt.title('Colour Histogram with Mask')
plt.xlabel('Bins')
plt.ylabel('Pixel Count')

colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], mask, [256], [0,256])
    plt.plot(hist, color=col, label=f'{col.upper()} channel')
    plt.xlim([0,256])

plt.legend()
plt.gca().set_facecolor('whitesmoke')   # set plot area background
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()

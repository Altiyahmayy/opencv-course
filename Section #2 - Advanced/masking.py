import cv2 as cv
import numpy as np

# Load image
img = cv.imread('../Resources/Photos/dogcat.jpg')
cv.imshow('Original Image', img)

# Blank canvas
blank = np.zeros(img.shape[:2], dtype='uint8')

# Circle + Rectangle
circle = cv.circle(blank.copy(), (img.shape[1]//2 + 45, img.shape[0]//2), 100, 255, -1)
rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)

# Combine shapes with XOR for “char2” effect
weird_shape = cv.bitwise_xor(circle, rectangle)
cv.imshow('Weird Shape (XOR)', weird_shape)

# Apply mask
masked = cv.bitwise_and(img, img, mask=weird_shape)

# Add text overlay
cv.putText(masked, 'Char2 Masked!', (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

# Show side-by-side comparison
combined = np.hstack([img, masked])
cv.imshow('Original vs Masked', combined)

cv.waitKey(0)
cv.destroyAllWindows
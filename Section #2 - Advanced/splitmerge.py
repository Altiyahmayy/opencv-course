#pylint:disable=no-member
import cv2 as cv
import numpy as np

img = cv.imread('../Resources/Photos/hehe.jpg')
cv.imshow('Park', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

# Split channels
b, g, r = cv.split(img)

# Create colored channel images
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

# Overlay text labels
cv.putText(blue, 'Blue Channel', (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
cv.putText(green, 'Green Channel', (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
cv.putText(red, 'Red Channel', (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

# Show individual windows
cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

# Side-by-side comparison
combined = np.hstack([blue, green, red])
cv.imshow('Channels Combined', combined)

print("Original shape:", img.shape)
print("Blue shape:", b.shape)
print("Green shape:", g.shape)
print("Red shape:", r.shape)

# Merge back to original
merged = cv.merge([b, g, r])
cv.imshow('Merged Image', merged)

cv.waitKey(0)
cv.destroyAllWindows()

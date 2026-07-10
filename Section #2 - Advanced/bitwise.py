# pylint:disable=no-member

import cv2 as cv
import numpy as np

# Create a blank color image
blank = np.zeros((500, 500, 3), dtype='uint8')

# Draw a blue rectangle (larger)
rectangle = cv.rectangle(
    blank.copy(),
    (50, 50),      # Top-left corner
    (450, 400),    # Bottom-right corner
    (255, 0, 0),   # Blue (BGR)
    -1
)

# Draw a yellow circle (smaller)
circle = cv.circle(
    blank.copy(),
    (300, 250),    # Center
    100,           # Radius
    (0, 255, 255), # Yellow (BGR)
    -1
)

# Display the shapes
cv.imshow('Blue Rectangle', rectangle)
cv.imshow('Yellow Circle', circle)

# Bitwise AND --> intersecting regions
bitwise_and = cv.bitwise_and(rectangle, circle)
cv.imshow('Bitwise AND', bitwise_and)

# Bitwise OR --> union of both shapes
bitwise_or = cv.bitwise_or(rectangle, circle)
cv.imshow('Bitwise OR', bitwise_or)

# Bitwise XOR --> non-overlapping regions
bitwise_xor = cv.bitwise_xor(rectangle, circle)
cv.imshow('Bitwise XOR', bitwise_xor)

# Bitwise NOT
bitwise_not = cv.bitwise_not(circle)
cv.imshow('Circle NOT', bitwise_not)

cv.waitKey(0)
cv.destroyAllWindows()
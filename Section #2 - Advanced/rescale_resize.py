import cv2 as cv
import numpy as np

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

capture = cv.VideoCapture('../Resources/Videos/videoni.mp4')

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break

    # Resized frame (mas gamay)
    frame_resized = rescaleFrame(frame, scale=.4)

    # Pad resized frame aron match ang height sa original
    h_diff = frame.shape[0] - frame_resized.shape[0]
    top_pad = h_diff // 2
    bottom_pad = h_diff - top_pad

    padded_resized = cv.copyMakeBorder(frame_resized, top_pad, bottom_pad, 0, 0,
                                       cv.BORDER_CONSTANT, value=(0,0,0))

    # Combine side-by-side (pareho na ang height tungod sa padding)
    combined = np.hstack([frame, padded_resized])

    cv.imshow('Original vs Resized (with padding)', combined)

    key = cv.waitKey(20) & 0xFF
    if key == ord('q'):   # quit
        break
    elif key == ord('s'): # save frame
        cv.imwrite('saved_frame.jpg', padded_resized)

capture.release()
cv.destroyAllWindows()

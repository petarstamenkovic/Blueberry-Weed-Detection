# This code detects and outlines the bluberry weed on the image samples

# Libraries import
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a sample image
#sampleImage = cv2.imread("imgs/realWorldSample/rw_sample3.png")
sampleImage = cv2.imread("imgs/labSamples/lab_sample4.jpg")
resizedSampleImage = cv2.resize(sampleImage, (640,480))
cv2.imshow("Sample Image",resizedSampleImage)
cv2.waitKey(0)

# Cropped image - Slice out the top 1/3 of the picture since it does not feature blueberry bank
imageSize = resizedSampleImage.shape # Return format is (h,w,ch_num)
x_start , y_start, x_end , y_end = 0 , int(imageSize[0] / 3) , imageSize[1] , imageSize[0]
sampleImageCropped = resizedSampleImage[y_start:y_end , x_start:x_end]
cv2.imshow("Cropped Sample Image", sampleImageCropped)
cv2.waitKey(0)

# Transform to HSV colorspace
hsvSampleImage = cv2.cvtColor(sampleImageCropped, cv2.COLOR_BGR2HSV)
cv2.imshow("Sample Image in HSV Colorspace",hsvSampleImage)
cv2.waitKey(0)

# Define lower and upper color threshold - HSV color circle
## STANDARD GREEN COLOR RANGE ##
green_lower = np.array([60,40,40])
green_higher = np.array([90,255,255])

# Create a mask for detecting color green
mask = cv2.inRange(hsvSampleImage, green_lower, green_higher)
cv2.imshow("Green mask", mask)
cv2.waitKey(0)

# Draw out contours with selected color range
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Number of contours detected: {len(contours)}")
result = sampleImageCropped.copy()
cv2.drawContours(result, contours, -1, (255,0,255), 2)
cv2.imshow("Final image", result)
cv2.waitKey(0)

# Draw out bounding boxes for x and y coordinatesq
boundingBoxImage = sampleImageCropped.copy()

# Go through all detected contours - Extract coordinates and draw bounding boxes
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    center = ((x+x+w)/2 , (y+y+w)/2)
    if cv2.contourArea(c) > 100: # Set the minimal area of a contour - ignore smaller
        cv2.rectangle(boundingBoxImage, (x,y) , (x+w, y+h), (255,0,255), 2)
        print(f"Coordinates for contour center are (in pixels) -> x : {center[0]}: y : {center[1]} and area : {cv2.contourArea(c)}")

cv2.imshow("Bounded box image", boundingBoxImage)
cv2.waitKey(0)

# Remove all opened OpenCV windows
cv2.destroyAllWindows()
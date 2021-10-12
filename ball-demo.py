import sys
import cv2
import numpy as np
import time
import math
from matplotlib import pyplot as plt

# Record Start time for total frame processing
start = time.time()*1000.0


# Define Upper-Lower Bound of threshold for outer edge (GRAY)
# threshvalgray = ((0,0,0),(255,255,20))

# Define Upper-Lower Bound of threshold for blue box (BLUE)
# threshvalblue = ((89,95,158),(98,155,200))

# Define Upper-Lower Bound of threshold for green ball (GREEN)
#threshvalgreen = ((20,90,50),(90,210,250))
threshvalgreen = ((25,90,100),(95,255,255))

# Image Resizing function
def imgresize(timg, scale_percent):
	width = int(timg.shape[1] * scale_percent / 100)
	height = int(timg.shape[0] * scale_percent / 100)
	dim = (width, height)

	# resize image
	return cv2.resize(timg, dim, interpolation = cv2.INTER_AREA) 

# CODE USED FOR READING FILE FROM SINGLE IMAGE
# Read File name
file = sys.argv[1]

# CODE USED FOR READING FILE FROM VIDEO FILE
cap = cv2.VideoCapture(file)

file_template = sys.argv[2]

print('Reading from file: ',file)


# Show steps
# 1 is true, 0 is false
disp = 1

# Scale for image
scale = 50


# Process Template (Will be made into a seperate function later)
# Very simple Process, should not be done more than once on code excecution
img_template = imgresize(cv2.imread(file_template, cv2.IMREAD_UNCHANGED), scale)
hsv_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2HSV)
thresh_template = cv2.inRange(hsv_template, threshvalgreen[0], threshvalgreen[1])
kernel = np.ones((2,2),np.uint8)
ret_template = cv2.dilate(thresh_template,kernel,iterations = 1)
contours_template, hierarchy = cv2.findContours(ret_template, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_template, contours_template, -1, (0,255,0), 1)
cv2.imshow('Template', img_template)
x,y,w,h = cv2.boundingRect(contours_template[0])
extern_template = cv2.contourArea(contours_template[0]) / (w * h)




# CODE USED FOR READING FILE FROM SINGLE IMAGE
# Read an image
# img_gray = cv2.imread(file, cv2.IMREAD_UNCHANGED)
# img_blue = cv2.imread(file, cv2.IMREAD_UNCHANGED)
# imgorig = imgresize(cv2.imread(file, cv2.IMREAD_UNCHANGED), 25)

# Track Number of frames
framecount = 0

while cap.isOpened():
	framecount = framecount + 1
	# CODE USED FOR READING FILE FROM VIDEO FILE
	ret, img_green = cap.read()

	# If ret is false, video is on last frame
	if ret == False:
		break
	imgorig = img_green.copy()

	# Resize the image
	img_green = imgresize(img_green, scale)
	imgorig = imgresize(imgorig, scale)

	# Show both images
	if disp == 1:
		cv2.imshow('Resized Image', img_green)

	# Convert to hsv Colorspace
	hsv_green = cv2.cvtColor(img_green, cv2.COLOR_BGR2HSV)

	# Threshhold image
	thresh_green = cv2.inRange(hsv_green, threshvalgreen[0], threshvalgreen[1])

	# Display Threshold
	if disp == 1:
		cv2.imshow('Threshold Green Outline', thresh_green)

	# Dilate image to help identify edge
	kernel = np.ones((2,2),np.uint8)
	ret_green = cv2.dilate(thresh_green,kernel,iterations = 1)

	# Display Morphology
	if disp == 1:
		cv2.imshow('Morphology Green Outline', ret_green)


	# Calculate Contours
	contours_green, hierarchy = cv2.findContours(ret_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


	# Display Contours
	if disp == 1:
		cv2.drawContours(imgorig, contours_green, -1, (0,0,255), 1)
		cv2.imshow('Contours', imgorig)


	# ------The Actual Important Code------
	# Set up for finding edge of jig

	# Best Match Score
	# If there is no singificant match, it will stay 1
	bestMatch = 2

	# Match Score
	match = 0

	# Countour Number
	contnum = 0

	# Run through all contours found, and check for a match with the template contour
	for i in range(len(contours_green)):
		match = cv2.matchShapes(contours_template[0],contours_green[i],1,0.0)
		if math.isinf(match):
			match = 0
		# print('Match: ', match)
		# cv2.drawContours(imgorig, contours_gray, i, (255,0,0), 1)
		# cv2.imshow('test', imgorig)
		x,y,w,h = cv2.boundingRect(contours_green[i])
		extern_contour = cv2.contourArea(contours_green[i]) / (w * h)
		# print('Extern Percentage: ',  abs(extern_contour - extern_template) / extern_template)
		if match < bestMatch and match < 1.5 and abs(extern_contour - extern_template) / extern_template <= .2:
			bestMatch = match
			contnum = i
		# cv2.waitKey()

	# Print to see if we have a match
	# print('best match:',  bestMatch)
	if bestMatch == 2:
		print('MATCH FAILED, CHECK FOCUS AND MAKE SURE THERE ARE NO OBSTRUCTIONS')
		cv2.putText(imgorig, 'MATCH FAILED', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
	if len(contours_green) > 0:
		# Display the best contour on image
		# cv2.drawContours(imgorig, contours_gray[contnum], -1, (0,255,0), 1)
		x,y,w,h = cv2.boundingRect(contours_green[contnum])
		cv2.rectangle(imgorig, (x, y), (x+w, y+h), (0,0,255),2)
		cv2.putText(imgorig, 'Distance:' , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
		cv2.imshow('Final Contour', imgorig)

	ch = cv2.waitKey(1)
	if ch == ord('q'):
	        break


# Print total time for frame
print('total time: ', time.time()*1000.0-start)

print('Average FPS', framecount/((time.time()*1000.0-start)/1000))

cap.release()
cv2.destroyAllWindows()

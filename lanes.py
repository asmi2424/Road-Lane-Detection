import cv2
import numpy as np

# --- Configuration (Define your paths here) ---
VIDEO_PATH = "test_video.mp4"  # Replace with your video path
IMAGE_PATH = "test_image.jpg"  # Replace with your image path


def make_coordinates(image, line_parameters):
    if line_parameters is None:  # Handle the case where line_parameters is None
        return None
    slope, intercept = line_parameters
    y1 = image.shape[0]  # Height
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is not None:  # Check if lines is not None
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0) if left_fit else None # Handle empty lists
    right_fit_average = np.average(right_fit, axis=0) if right_fit else None # Handle empty lists

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line]) if left_line is not None and right_line is not None else None  # Return None if either line is None

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Use BGR2GRAY consistently
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Kernel size is 5x5
    canny = cv2.Canny(blur, 50, 150)  # Adjust these thresholds if needed
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)  # Reshaping all the lines to a 1D array.
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)  # Draw a Blue Line(BGR in OpenCV)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]  # Adjust these points as needed
    ])
    mask = np.zeros_like(image)  # Create a black mask to apply the above Triangle on.
    cv2.fillPoly(mask, polygons, 255)  # A complete white triangle polygon on a black mask.
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# --- Image Processing ---
image = cv2.imread(IMAGE_PATH)
if image is not None: # Check if the image was loaded successfully
    lane_image = np.copy(image)  # Always make a copy when working with arrays rather than directly assigning
    canny_image = canny(lane_image)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=10, maxLineGap=5) # Consistent Hough parameters
    averaged_lines = average_slope_intercept(lane_image, lines)
    line_image = display_lines(lane_image, averaged_lines)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)  # Imposing the line_image on the original image

    cv2.imshow('Result Image', combo_image)
    cv2.waitKey(0)

# --- Video Processing ---
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video: {VIDEO_PATH}")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=10, maxLineGap=5) # Consistent Hough parameters

    if lines is not None: # Check if lines is not None
        averaged_lines = average_slope_intercept(frame, lines)
        if averaged_lines is not None: # Check if averaged_lines is not None
            line_image = display_lines(frame, averaged_lines)
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)  # Imposing the line_image on the original image
            cv2.imshow('Result Video', combo_image)

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

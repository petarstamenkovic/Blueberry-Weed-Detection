# Module where all functions used are stored in

import json
import math
import time
import numpy as np
import depthai as dai
import open3d as o3d

# Parameters for plane comparison
normal_vector_threshold = 0.5
offset_threshold = 0.5

# Camera intrinsics - Extracted from calibration.py
#fx, fy = 570.33, 570.3   # Focal lenghts
#cx, cy = 655.38, 358.85  # Camera principle values 

# Parameters for OAK D - Camera number 9
fx = 856.8466
fy = 857.0558
cx = 632.1884
cy = 366.1514

# Function to get stereo pair and configure it 
def getStereoPair(pipeline, monoLeft, monoRight):
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setExtendedDisparity(False)
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    return stereo

# Function that tries to maintain the focused plain
def updatePlane(prevPlane, newPlane):
    # Current plane Carthesian parameters and normal vector
    a,b,c,d = newPlane
    new_normal_vector = np.array([a,b,c])
    new_normal_vector = new_normal_vector / np.linalg.norm(new_normal_vector)

    # In first iteration, there is no 'previous' plane
    if prevPlane is None:
        return newPlane
    
    # Previous plane Carthesian parameters and normal vector
    prev_a, prev_b, prev_c, prev_d = prevPlane
    old_normal_vector = np.array([prev_a, prev_b, prev_c])
    old_normal_vector = old_normal_vector / np.linalg.norm(old_normal_vector)

    # Calculate the similarities of the normal vectors of a previous and a current plane
    normal_vectors_alligned = abs(np.dot(new_normal_vector, old_normal_vector))

    # Calculate the difference in offsets of planes
    offset_diffence = abs(d - prev_d)

    # Comparison of the planes under conditions
    if normal_vectors_alligned > (1 - normal_vector_threshold) and offset_diffence < offset_threshold:
        return newPlane  # Update the plane with a new one 
    else:
        return prevPlane # Keep the previous plane
    

# Function that converts REAL-WORLD 3D point cloud back to 2D and returns PIXEL coordinates in a tuple
def backtrackTo2D(endpoint):
    X, Y, Z = endpoint
    
    # Skip if Z is 0 due to division
    if Z == 0:
        return None
    
    # Backtrack - Both u and v are in pixel values - these will limit the ROI 
    u = int((X * fx) / Z + cx)
    v = int((Y * fy) / Z + cy)

    return (u,v)

# Convert PCA lines to Open3D lines
def lineToO3dLine(line_set, mean_point, direction, length=0.5, color=[0,1,0]):
    direction = direction / np.linalg.norm(direction)

    # Create endpoints of the line segment
    endpoint1 = mean_point - (direction * length / 2)
    endpoint2 = mean_point + (direction * length / 2)

    """
    print("Endpoint 1 : ", endpoint1)
    print("-------------------------------------")
    print("Endpoint 2 : ", endpoint2)
    print("=====================================")
    """

    # Define line parameters
    line_points = [endpoint1, endpoint2] # Actual points of the line
    lines = [[0, 1]]  # Single line connecting the two points - index based ; "connect 0-th and 1-st point"

    # Points in 2D - For line equation and ROI configuration later in the process
    point1 = backtrackTo2D(endpoint1)
    point2 = backtrackTo2D(endpoint2)

    # Add line parameters
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color])
    
    return line_set, point1, point2


# Function that calculates 2D line equation
def calculateLineEquation(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    A = y1 - y2
    B = x2 - x1

    m = round(A / B , 2)

    return m, x1, y1

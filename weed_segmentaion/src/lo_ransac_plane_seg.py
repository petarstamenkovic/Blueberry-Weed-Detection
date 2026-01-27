# This code uses a built LO-RANSAC to enhance the fitted plane.

# Imports 
import numpy as np

# ========= FUNCTIONS ========= #

# Fitting a plane - 3 points needed
def fit_a_plane(p1,p2,p3) : 
    v1 = p2 - p1
    v2 = p3 - p1

    n = np.cross(v1,v2)

    if np.linalg.norm(n) == 0:
        return None # Colinear points
    
    n = n / np.linalg.norm(n)

    d = -np.dot(n,p1)
    return n,d

# Compute residuals - distance from the plane to the points (format N x 3)
def point_plane_distance(points, normal, d):
    distances = np.abs(points @ normal + d)
    return distances

# Standard RANSAC loop
def ransac_plane(points, threshold = 0.01, max_iter = 1000):
    best_inliers = []
    best_plane = None

    N = len(points)
    for i in range(max_iter):
        # Random 3 points
        idx = np.random.choice(N,3,replace=False)
        p1, p2, p3 = points[idx]

        plane = fit_a_plane(p1,p2,p3)

        if plane is None: 
            continue

        normal,d = plane

        # Inliers calculation
        distances = point_plane_distance(points, normal, d)
        inliers = np.where(distances < threshold)[0]

        # Check if best
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (normal,d)
    
    return best_plane, best_inliers

# Useful to research about a SVD
# Local optimization - fit a plane to ALL inlier points from the previous fitting
def refit_plane(points, inliers):
    pts = points[inliers]

    # Plane fit via SVD (least squares)
    centroid = np.mean(pts, axis=0)
    _, _, vh = np.linalg.svd(pts - centroid) # last singular vector is plane normal
    normal = vh[2, :] 
    d = -np.dot(normal, centroid)
    return normal,d


# LO-RANSAC
def fit_lo_ransac(points, threshold = 0.01, max_iter = 1000):
    # Standard RANSAC first
    plane, inliers = ransac_plane(points, threshold, max_iter)

    if plane is None:
        return None, []
    
    # Local optimization
    normal, d = refit_plane(points, inliers)

    # Recompute inliers 
    distances = point_plane_distance(points, normal, d)
    inliers = np.where(distances < threshold)[0]

    return (normal,d), inliers


# HOW TO USE IT 
# plane, inliers = fit_lo_ransac(points, threshold = 0.01, max_iter = 1000)

import numpy as np


#1. Declare points that appear both on cameras and field
COORDINATES = {
    0: {
        'cam': np.array([[  0, 390, 358],
                         [154, 152, 375],
                         [  1,   1,   1]]), 
        'field': np.array([[-37.5, -37.5, 0],
                           [   55,     0, 0],
                           [    1,     1, 1]])
        },
    1: {
        'cam': np.array([[414,  29,  61],
                         [160, 159, 381],
                         [  1,   1,   1]]), 
        'field': np.array([[-37.5, -37.5, 0],
                           [  -55,     0, 0],
                           [    1,     1, 1]])
        },
    2: {
        'cam': np.array([[  5, 385, 351],
                         [163, 167, 386],
                         [  1,   1,   1]]), 
        'field': np.array([[37.5, 37.5, 0],
                           [ -55,    0, 0],
                           [   1,    1, 1]])
        },
    3: {
        'cam': np.array([[405,  19,  56],
                         [156, 162, 381],
                         [  1,   1,   1]]), 
        'field': np.array([[37.5, 37.5, 0],
                           [  55,    0, 0],
                           [   1,    1, 1]])
        },
}

NUM_CAMERAS = len(COORDINATES)

#2. Find the perspective transformation coefficients for each camera
coeffs = {}
for cam_id in range(NUM_CAMERAS):
    cam_coords = COORDINATES[cam_id]['cam']
    field_coords = COORDINATES[cam_id]['field']
    transformation_coeffs = np.dot(field_coords, np.linalg.inv(cam_coords))
    coeffs[cam_id] = transformation_coeffs


def cam_to_field(cam_id, x, y):
    """transforms camera coordinates into field coordinates"""
    cam_coords = np.array([x, y, 1])
    field_coords = np.dot(coeffs[cam_id], cam_coords)
    return field_coords[0], field_coords[1]

#!/usr/bin/python
# -*- coding: utf-8 -*-
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2019 Intel Corporation. All Rights Reserved.

#####################################################
##           librealsense T265 example             ##
#####################################################

# First import the library
import pyrealsense2 as rs
import numpy as np
import cv2
import time
from math import pi, tan


"""
Returns R, T transform from src to dst
"""
def get_extrinsics(src, dst):
    extrinsics = src.get_extrinsics_to(dst)
    R = np.reshape(extrinsics.rotation, [3,3]).T
    T = np.array(extrinsics.translation)
    return (R, T)

"""
Returns a camera matrix K from librealsense intrinsics
"""
def camera_matrix(intrinsics):
    return np.array([[intrinsics.fx,             0, intrinsics.ppx],
                     [            0, intrinsics.fy, intrinsics.ppy],
                     [            0,             0,              1]])

"""
Returns the fisheye distortion from librealsense intrinsics
"""
def fisheye_distortion(intrinsics):
    return np.array(intrinsics.coeffs[:4])


# Declare RealSense pipeline, encapsulating the actual device and sensors
pipe = rs.pipeline()

# Build config object and request pose data
cfg = rs.config()
cfg.enable_stream(rs.stream.fisheye,1)
cfg.enable_stream(rs.stream.fisheye,2)

# Start streaming with requested config
pipe.start(cfg)


profiles = pipe.get_active_profile()
streams = {"left"  : profiles.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
            "right" : profiles.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()}
intrinsics = {"left"  : streams["left"].get_intrinsics(),
                "right" : streams["right"].get_intrinsics()}

# Print information about both cameras
print("Left camera:",  intrinsics["left"])
print("Right camera:", intrinsics["right"])

# Translate the intrinsics from librealsense into OpenCV
K_left  = camera_matrix(intrinsics["left"])
D_left  = fisheye_distortion(intrinsics["left"])
K_right = camera_matrix(intrinsics["right"])
D_right = fisheye_distortion(intrinsics["right"])
(width, height) = (intrinsics["left"].width, intrinsics["left"].height)

# Get the relative extrinsics between the left and right camera
(R, T) = get_extrinsics(streams["left"], streams["right"])

# We need to determine what focal length our undistorted images should have
# in order to set up the camera matrices for initUndistortRectifyMap.  We
# could use stereoRectify, but here we show how to derive these projection
# matrices from the calibration and a desired height and field of view

# We calculate the undistorted focal length:
#
#         h
# -----------------
#  \      |      /
#    \    | f  /
#     \   |   /
#      \ fov /
#        \|/
stereo_fov_rad = 90 * (pi/180)  # 90 degree desired fov
stereo_height_px = 800          # 300x300 pixel stereo output
stereo_focal_px = stereo_height_px/2 / tan(stereo_fov_rad/2)

# We set the left rotation to identity and the right rotation
# the rotation between the cameras
R_left = np.eye(3)
R_right = R
max_disp = 112
# The stereo algorithm needs max_disp extra pixels in order to produce valid
# disparity on the desired output region. This changes the width, but the
# center of projection should be on the center of the cropped image
stereo_width_px = stereo_height_px + max_disp
stereo_size = (stereo_width_px, stereo_height_px)
stereo_cx = (stereo_height_px - 1)/2 + max_disp
stereo_cy = (stereo_height_px - 1)/2

# Construct the left and right projection matrices, the only difference is
# that the right projection matrix should have a shift along the x axis of
# baseline*focal_length
P_left = np.array([[stereo_focal_px, 0, stereo_cx, 0],
                    [0, stereo_focal_px, stereo_cy, 0],
                    [0,               0,         1, 0]])
P_right = P_left.copy()
P_right[0][3] = T[0]*stereo_focal_px

# Construct Q for use with cv2.reprojectImageTo3D. Subtract max_disp from x
# since we will crop the disparity later
Q = np.array([[1, 0,       0, -(stereo_cx - max_disp)],
                [0, 1,       0, -stereo_cy],
                [0, 0,       0, stereo_focal_px],
                [0, 0, -1/T[0], 0]])

# Create an undistortion map for the left and right camera which applies the
# rectification and undoes the camera distortion. This only has to be done
# once
m1type = cv2.CV_32FC1
(lm1, lm2) = cv2.fisheye.initUndistortRectifyMap(K_left, D_left, R_left, P_left, stereo_size, m1type)
(rm1, rm2) = cv2.fisheye.initUndistortRectifyMap(K_right, D_right, R_right, P_right, stereo_size, m1type)
undistort_rectify = {"left"  : (lm1, lm2),
                        "right" : (rm1, rm2)}


fisheye1_video_writer = cv2.VideoWriter('fisheye1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (800,800))
fisheye2_video_writer = cv2.VideoWriter('fisheye2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (800,800))
try:
    while True:
        # Wait for the next set of frames from the camera
        frames = pipe.wait_for_frames()
        frame1 =np.asanyarray(frames.get_fisheye_frame(1).as_video_frame().get_data())
        frame2 =np.asanyarray(frames.get_fisheye_frame(2).as_video_frame().get_data())

        #undistort them
        center_undistorted = {"left" : cv2.remap(src = frame1,
                                          map1 = undistort_rectify["left"][0],
                                          map2 = undistort_rectify["left"][1],
                                          interpolation = cv2.INTER_LINEAR),
                                  "right" : cv2.remap(src = frame2,
                                          map1 = undistort_rectify["right"][0],
                                          map2 = undistort_rectify["right"][1],
                                          interpolation = cv2.INTER_LINEAR)}
        #convert to three channel
        color_image = cv2.cvtColor(center_undistorted["left"][:,max_disp:], cv2.COLOR_GRAY2RGB)
        color_image2 = cv2.cvtColor(center_undistorted["right"][:,max_disp:], cv2.COLOR_GRAY2RGB)

        #save, show video
        fisheye1_video_writer.write(color_image)
        fisheye2_video_writer.write(color_image2)

        cv2.imshow("frame1", color_image)
        cv2.imshow("frame2", color_image2)
        key = cv2.waitKey(1)

        if key == ord("q"):
            break

except KeyboardInterrupt:
    fisheye1_video_writer.release()
    fisheye2_video_writer.release()
finally:
    pipe.stop()

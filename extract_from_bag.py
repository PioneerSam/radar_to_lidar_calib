import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np

from rosbags.typesys import get_types_from_msg, register_types, Stores, get_typestore
from rosbags.serde import serialize_cdr

import cv2
import cv_bridge

import matplotlib.pyplot as plt
from helper import *

# Lidar is 10 Hz, and radar is 4 Hz
# we can use the timestamp to align them
# do all the radar image processing
# and choose lidar pair based on the timestamps

# Extract Radar
print("The current working directory is: ", os.getcwd())

calibration_rosbag_path = "rosbags/sam_radar_lidar/"

radar_fft_data, radar_azimuths, radar_timesteps, radar_times = get_radar_scan_images_and_timestamps(calibration_rosbag_path)

radar_data = list(zip(radar_fft_data, radar_azimuths, radar_timesteps))

# print("radar_times: ", len(radar_times))
# print("len(radar_azimuth): ", np.array(radar_azimuths).shape)
# print("len(radar_timesteps): ", np.array(radar_timesteps).shape)
# print("radar time steps first 10", radar_times[:10])

# outpath = "result/radar_bev.npz"

# cv2.imshow("radar", radar_fft_data[0])
# cv2.waitKey(0)

# Extract Lidar according to closest radar timestamp (10 lidar)
get_lidar_radar_pair(calibration_rosbag_path, radar_times, radar_fft_data,radar_azimuths, radar_timesteps)



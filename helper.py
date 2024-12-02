import os
import os.path as osp
import numpy as np
# from pyboreas.utils.utils import load_lidar
# from pyboreas.data.sensors import Lidar, Radar
# import open3d as o3d
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm
# from pyboreas import BoreasDataset
# from pyboreas.utils.utils import get_inverse_tf
import pylgmath.so3.operations as so3op

from rosbags.highlevel import AnyReader
from pathlib import Path
import utm
from scipy.spatial.transform import Rotation, Slerp
from scipy import interpolate

from rosbags.typesys import get_types_from_msg, register_types, Stores, get_typestore
from rosbags.serde import serialize_cdr
from rosbags.serde import deserialize_cdr

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

import cv2
import cv_bridge


# I will need to register the custom navtech message type
# define messages
SCAN_MSG = """
# A ROS message carrying a B Scan and its associated metadata (e.g. timestamps, encoder IDs)
# B Scan from one rotation of the radar, also holds the time stamp information
sensor_msgs/Image b_scan_img

# The encoder values encompassed by the b scan
uint16[] encoder_values

# The timestamps of each azimuth in the scan
uint64[] timestamps
"""

FFT_MSG = """
# A ROS message based on an FFT data message from a radar Network order means big endian

# add a header message to hold message timestamp
std_msgs/Header header

# angle (double) represented as a network order (uint8_t) byte array (don't use)
uint8[] angle

# azimuth (uint16_t) represented as a network order (uint8_t) byte array (encoder tick number)
uint8[] azimuth

# sweep_counter (uint16_t) represented as a network order (uint8_t) byte array
uint8[] sweep_counter

# ntp_seconds (uint32_t) represented as a network order (uint8_t) byte array
uint8[] ntp_seconds

# ntp_split_seconds (uint32_t) represented as a network order (uint8_t) byte array
uint8[] ntp_split_seconds

# data (uint8_t) represented as a network order (uint8_t) byte array
uint8[] data

# data_length (uint16_t) represented as a network order (uint8_t) byte array
uint8[] data_length """


# helper function to manipulate radar and lidar points
# ** assumes lidar points already in radar frame
def project_lidar_onto_radar(points, max_elev=0.05):
    # find points that are within the radar scan FOV
    points_out = []
    for i in range(points.shape[0]):
        elev = np.arctan2(points[i, 2], np.sqrt(points[i, 0]**2 + points[i, 1]**2))
        if np.abs(elev) <= max_elev:
            points_out.append(points[i, :])
    points = np.array(points_out)
    # project to 2D (spherical projection)
    for i in range(points.shape[0]):
        rho = np.sqrt(points[i, 0]**2 + points[i, 1]**2 + points[i, 2]**2)
        phi = np.arctan2(points[i, 1], points[i, 0])
        points[i, 0] = rho * np.cos(phi)
        points[i, 1] = rho * np.sin(phi)
        points[i, 2] = 0.0
    return points



def polar_to_cartesian_points(
    azimuths: np.ndarray,
    polar_points: np.ndarray,
    radar_resolution: float,
    downsample_rate=1,
    range_offset = -0.31,
) -> np.ndarray:
    """Converts points from polar coordinates to cartesian coordinates
    Args:
        azimuths (np.ndarray): The actual azimuth of reach row in the fft data reported by the Navtech sensor
        polar_points (np.ndarray): N x 2 array of points (azimuth_bin, range_bin)
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
        downsample_rate (float): fft data may be downsampled along the range dimensions to speed up computation
    Returns:
        np.ndarray: N x 2 array of points (x, y) in metric
    """
    N = polar_points.shape[0]
    cart_points = np.zeros((N, 2))
    for i in range(0, N):
        azimuth = azimuths[int(polar_points[i, 0])]
        r = polar_points[i, 1] * radar_resolution * downsample_rate + radar_resolution / 2 + range_offset
        cart_points[i, 0] = r * np.cos(azimuth)
        cart_points[i, 1] = r * np.sin(azimuth)
    return cart_points

def convert_to_bev(cart_points: np.ndarray, cart_resolution: float, cart_pixel_width: int) -> np.ndarray:
    """Converts points from metric cartesian coordinates to pixel coordinates in the BEV image
    Args:
        cart_points (np.ndarray): N x 2 array of points (x, y) in metric
        cart_pixel_width (int): width and height of the output BEV image
    Returns:
        np.ndarray: N x 2 array of points (u, v) in pixels which can be plotted on the BEV image
    """
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    print(cart_min_range)
    pixels = []
    N = cart_points.shape[0]
    for i in range(0, N):
        u = (cart_min_range + cart_points[i, 1]) / cart_resolution
        v = (cart_min_range - cart_points[i, 0]) / cart_resolution
        if 0 < u and u < cart_pixel_width and 0 < v and v < cart_pixel_width:
            pixels.append((u, v))
    return np.asarray(pixels)

def modifiedCACFAR(
    raw_scan: np.ndarray,
    minr=2.0,
    maxr=80.0,
    res=0.04381,
    width=101,
    guard=5,
    threshold=1.0,
    threshold2=0.0,
    threshold3=0.09,
    peak_summary_method='max_intensity'):
    # peak_summary_method: median, geometric_mean, max_intensity, weighted_mean
    rows = raw_scan.shape[0]
    cols = raw_scan.shape[1]
    if width % 2 == 0: width += 1
    w2 = int(np.floor(width / 2))
    mincol = int(minr / res + w2 + guard + 1)
    if mincol > cols or mincol < 0: mincol = 0
    maxcol = int(maxr / res - w2 - guard)
    if maxcol > cols or maxcol < 0: maxcol = cols
    N = maxcol - mincol
    targets_polar_pixels = []
    for i in range(rows):
        mean = np.mean(raw_scan[i])
        peak_points = []
        peak_point_intensities = []
        for j in range(mincol, maxcol):
            left = 0
            right = 0
            for k in range(-w2 - guard, -guard):
                left += raw_scan[i, j + k]
            for k in range(guard + 1, w2 + guard):
                right += raw_scan[i, j + k]
            # (statistic) estimate of clutter power
            stat = max(left, right) / w2  # GO-CFAR
            thres = threshold * stat + threshold2 * mean + threshold3
            if raw_scan[i, j] > thres:
                peak_points.append(j)
                peak_point_intensities.append(raw_scan[i, j])
            elif len(peak_points) > 0:
                if peak_summary_method == 'median':
                    r = peak_points[len(peak_points) // 2]
                elif peak_summary_method == 'geometric_mean':
                    r = np.mean(peak_points)
                elif peak_summary_method == 'max_intensity':
                    r = peak_points[np.argmax(peak_point_intensities)]
                elif peak_summary_method == 'weighted_mean':
                    r = np.sum(np.array(peak_points) * np.array(peak_point_intensities) / np.sum(peak_point_intensities))
                else:
                    raise NotImplementedError("peak summary method: {} not supported".format(peak_summary_method))
                targets_polar_pixels.append((i, r))
                peak_points = []
                peak_point_intensities = []
    return np.asarray(targets_polar_pixels)



def get_radar_scan_images_and_timestamps(path):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    typestore.register(get_types_from_msg(FFT_MSG, 'nav_messages/msg/RadarFftDataMsg'))
    typestore.register(get_types_from_msg(SCAN_MSG,'navtech_msgs/msg/RadarBScanMsg'))

    # from rosbags.typesys.types import navtech_msgs__msg__RadarBScanMsg as RadarBScanMsg

    RadarBScanMsg = typestore.types['navtech_msgs/msg/RadarBScanMsg']
    scan_type = RadarBScanMsg.__msgtype__

    # intialize the arrays
    radar_times = []
    radar_images = []

    radar_timesteps = []
    radar_azimuths = []
    radar_fft_data = []

    lookup_tb = dict()
    print("Processing: Getting image_timestamp and radar image")
    with AnyReader([Path(path)]) as reader:
        connections = [x for x in reader.connections if x.topic == '/radar/b_scan_msg']
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            # print(f"Raw data size: {len(rawdata)}")
            msg = typestore.deserialize_cdr(rawdata, scan_type)
            # need to make sure everything is in secs
            radar_time_sec = msg.b_scan_img.header.stamp.sec
            radar_time_nano_sec = msg.b_scan_img.header.stamp.nanosec
            radar_time = radar_time_sec + radar_time_nano_sec/1e9
            # round to 4 significant digits 
            radar_time = round(radar_time,3)
            radar_times.append(radar_time)

            # now store the image
            bridge = cv_bridge.CvBridge()
            polar_img = bridge.imgmsg_to_cv2(msg.b_scan_img)
            azimuth_angles = msg.encoder_values
            azimuth_timestamp = msg.timestamps

            fft_data = msg.b_scan_img.data.reshape((msg.b_scan_img.height, msg.b_scan_img.width))
            
            radar_fft_data.append(polar_img)
            radar_azimuths.append(azimuth_angles)
            radar_timesteps.append(azimuth_timestamp)

            # print("fft_data",fft_data.shape)
            # plt.imshow(polar_img,cmap='gray', vmin=0, vmax=255)
            # plt.show()

            # print("polar image",polar_img.shape)
            azimuths = msg.encoder_values/5595*2*np.pi
            # print("azimuths",azimuths.shape)
            resolution = 0.040308
            cart_resolution = 0.275

            # convert the radar image to cartesian
            radar_image = radar_polar_to_cartesian(polar_img,azimuths, resolution, cart_resolution, 512)

            radar_images.append(radar_image)


    return radar_fft_data, radar_azimuths, radar_timesteps, radar_times



def radar_polar_to_cartesian(fft_data, azimuths, radar_resolution, cart_resolution=0.2384, cart_pixel_width=640,
                             interpolate_crossover=False, fix_wobble=True):
    # TAKEN FROM PYBOREAS
    """Convert a polar radar scan to cartesian.
    Args:
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        fft_data (np.ndarray): Polar radar power readings
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
        cart_resolution (float): Cartesian resolution (metres per pixel)
        cart_pixel_width (int): Width and height of the returned square cartesian output (pixels)
        interpolate_crossover (bool, optional): If true interpolates between the end and start  azimuth of the scan. In
            practice a scan before / after should be used but this prevents nan regions in the return cartesian form.

    Returns:
        np.ndarray: Cartesian radar power readings
    """
    # Compute the range (m) captured by pixels in cartesian scan
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    
    # Compute the value of each cartesian pixel, centered at 0
    coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)

    Y, X = np.meshgrid(coords, -1 * coords)
    sample_range = np.sqrt(Y * Y + X * X)
    sample_angle = np.arctan2(Y, X)
    sample_angle += (sample_angle < 0).astype(np.float32) * 2. * np.pi

    # Interpolate Radar Data Coordinates
    azimuth_step = (azimuths[-1] - azimuths[0]) / (azimuths.shape[0] - 1)
    sample_u = (sample_range - radar_resolution / 2) / radar_resolution
    sample_v = (sample_angle - azimuths[0]) / azimuth_step
    # This fixes the wobble in the old CIR204 data from Boreas
    M = azimuths.shape[0]
    azms = azimuths.squeeze()
    if fix_wobble:
        c3 = np.searchsorted(azms, sample_angle.squeeze())
        c3[c3 == M] -= 1
        c2 = c3 - 1
        c2[c2 < 0] += 1
        a3 = azms[c3]
        diff = sample_angle.squeeze() - a3
        a2 = azms[c2]
        delta = diff * (diff < 0) * (c3 > 0) / (a3 - a2 + 1e-14)
        sample_v = (c3 + delta).astype(np.float32)

    # We clip the sample points to the minimum sensor reading range so that we
    # do not have undefined results in the centre of the image. In practice
    # this region is simply undefined.
    sample_u[sample_u < 0] = 0

    if interpolate_crossover:
        fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
        sample_v = sample_v + 1

    polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
    return cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR)



# for lidar 

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from sensor_msgs.msg import PointField

def convert_to_ros2_pointcloud2(msg):
    """Convert a rosbags PointCloud2 object to ROS2's sensor_msgs.msg.PointCloud2."""
    ros2_msg = PointCloud2()
    ros2_msg.header = Header()
    ros2_msg.header.stamp.sec = msg.header.stamp.sec
    ros2_msg.header.stamp.nanosec = msg.header.stamp.nanosec
    ros2_msg.header.frame_id = msg.header.frame_id

    ros2_msg.height = msg.height
    ros2_msg.width = msg.width

    # Convert fields to a list of PointField
    ros2_msg.fields = [
        PointField(
            name=field.name,
            offset=field.offset,
            datatype=field.datatype,
            count=field.count,
        ) for field in msg.fields
    ]

    ros2_msg.is_bigendian = msg.is_bigendian
    ros2_msg.point_step = msg.point_step
    ros2_msg.row_step = msg.row_step
    ros2_msg.data = bytes(msg.data)  # Ensure data is a bytes object
    ros2_msg.is_dense = msg.is_dense

    return ros2_msg


def get_lidar_radar_pair(rosbag_path,radar_times,radar_fft_data,radar_azimuths,radar_timesteps):

    # I want 5 messages for lidar
    global index
    index = 0
    with AnyReader([Path(rosbag_path)]) as reader:
        connections = [x for x in reader.connections if x.topic == '/lslidar128/points']
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            try:
                msg = deserialize_cdr(rawdata, connection.msgtype)
                print(f"Deserialized message type: {type(msg)}")
            except KeyError as e:
                print(f"Deserialization KeyError: {e}")
                continue
            except Exception as e:
                print(f"Deserialization error: {e}")
                continue


            print("message type is", connection.msgtype)
            
            ros2_msg = convert_to_ros2_pointcloud2(msg)
            cloud_points = point_cloud2.read_points(ros2_msg, skip_nans=True)
            sec = msg.header.stamp.sec
            nsec = msg.header.stamp.nanosec

            
            lidar_time = sec + nsec/1e9
            lidar_time = round(lidar_time,3)
            print("lidar time",lidar_time)

            # find the closest radar time
            radar_time = min(radar_times, key=lambda x:abs(x-lidar_time))
            print("radar time",radar_time)
            
            # find the index of the radar time
            index = radar_times.index(radar_time)
            img = radar_fft_data[index]
            print("Sam:! radar image shape",img.shape)
            azimuths = radar_azimuths[index]
            timesteps = radar_timesteps[index]
            
            # save radar img and azimuth and timesteps
            print("saving radar image and meta number",index)
            cv2.imwrite('radar_sam/{}.png'.format(index), img)
            np.savetxt('radar_sam/{}_azimuths.txt'.format(index), azimuths, delimiter=',')
            np.savetxt('radar_sam/{}_timesteps.txt'.format(index), timesteps, delimiter=',')
          
            points = np.empty((0,3))
            for point in cloud_points:
                point = list(point)
                # print("Sam: point",point)
                x, y, z = point[:3]
                # print(x, y, z)
                points = np.append(points, [x, y, z])
            points = points.reshape(-1, 3)
            print("saving lidar points number",index)
            np.savetxt('lidar_sam/{}.txt'.format(index), points, delimiter=',')

            index += 1
            break

        
    
    return True




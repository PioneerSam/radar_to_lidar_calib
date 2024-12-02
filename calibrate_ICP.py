import os
import os.path as osp
import numpy as np
from pyboreas.utils.utils import load_lidar
from pyboreas.data.sensors import Lidar, Radar
# import open3d as o3d
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm
from pyboreas import BoreasDataset
from pyboreas.utils.utils import get_inverse_tf
import pylgmath.so3.operations as so3op


# use ICP to calibrate multiple scans at once
# we need to get 

root = '/workspace/Documents/boreas_data_collection/2023_12_08_calib/'
radar_resolution = 0.04381
cart_resolution = radar_resolution * 5
cart_pixel_width = int(200.0 / cart_resolution)
# initial guess for T_radar_lidar:
T_radar_lidar = np.array(
[[ 0.68148404,  0.73183297,  0.,          0.        ],
 [ 0.73183297, -0.68148404,  0.,          0.        ],
 [ 0.,          0.,         -1.,          0.365     ],
 [ 0.,          0.,          0.,          1.        ]])

# root = '/workspace/nas/ASRL/2021-Boreas/'
# # old radar:
# # bd = BoreasDataset(root, split=[['boreas-2020-11-26-13-58'],['boreas-2020-12-01-13-26']])
# # new radar:
# bd = BoreasDataset(root, split=[['boreas-2021-10-15-12-35'],['boreas-2021-10-22-11-36']])

# for seq in bd.sequences:
#     seq.synchronize_frames(ref='radar')

# T_radar_lidar = bd.sequences[0].calib.T_radar_lidar
# print("T_radar_lidar:")
# print(T_radar_lidar)
# rad = bd.sequences[0].get_radar(0)
# lid = bd.sequences[0].get_lidar(0)
# cart_resolution = rad.resolution * 5
# cart_pixel_width = int(200.0 / cart_resolution)
# out = rad.load_data()
# print('radar resolution:')
# print(rad.resolution)

# Lidar to Radar Spherical 2D Projection
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

# Radar Target Extraction
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


# Get radar-lidar pairs
rads_to_keep = sorted([f for f in os.listdir(osp.join(root, 'radar')) if f.endswith('.png')])
rads_to_keep = [Radar(osp.join(root, 'radar', fname)) for fname in rads_to_keep]
lids_to_keep = sorted([f for f in os.listdir(osp.join(root, 'lidar')) if f.endswith('.bin')])
lids_to_keep = [Lidar(osp.join(root, 'lidar', fname)) for fname in lids_to_keep]
print('radar frames to do calibration with:')
for rad, lid in zip(rads_to_keep, lids_to_keep):
    print('frame: {}'.format(rad.frame))

# LiDAR Extraction Parameters
crop_range = 60.0
voxel_size = 0.3
polar_r_scale = 2.0
vertical_angle_res = 0.0017453292519943296
r_scale = 4.0
h_scale = 2.0
num_sample1 = 10000
min_norm_score1 = 0.95
max_pair_d2 = 5.0**2

# Extract Lidar Radar Points
import cpp.build.extract_normals as extract_normals
import cpp.build.get_nearest_neighbors as get_nearest_neighbors
rads = []
lids = []

# for idx in rads_to_keep:
for rad, lid in zip(rads_to_keep, lids_to_keep):
    rad.load_data()
    lid.load_data()
    print(rad.frame)
    # crop range, voxel downsample, filter based on normal scores
    points_in = lid.points[:, :3]
    points_out = np.zeros(points_in.shape)
    normals = np.zeros(points_in.shape)
    normal_scores = np.zeros(points_in.shape[0])
    num_points = np.array([0])
    extract_normals.extract_normals(
        points_in,
        crop_range,
        voxel_size,
        polar_r_scale,
        vertical_angle_res,
        r_scale,
        h_scale,
        num_sample1,
        min_norm_score1,
        points_out,
        normals,
        normal_scores,
        num_points,
    )
    points_out = points_out[:num_points[0], :]
    normal_scores = normal_scores[:num_points[0]]
    normals = normals[:num_points[0], :]
    lid.points = np.copy(points_out)
    # transform lidar points into expected radar frame
    lid.transform(T_radar_lidar)
    # project lidar points into 2D
    lid.points = project_lidar_onto_radar(lid.points)
    lids.append(np.copy(lid.points))
    
    # extract radar targets
    if rad.resolution < 0.05:
        polar_targets = modifiedCACFAR(
            rad.polar,
            threshold=0.7,
            threshold3=0.23,
            width=137,
            guard=7,
            maxr=60.0,
            peak_summary_method="max_intensity",
        )
    else:
        polar_targets = modifiedCACFAR(
            rad.polar,
            threshold=1.0,
            threshold3=0.09,
            width=101,
            guard=5,
            maxr=60.0,
            peak_summary_method="max_intensity",
        )
    cart_targets = polar_to_cartesian_points(rad.azimuths, polar_targets, rad.resolution, range_offset=-0.31)
    radar_points = np.zeros((cart_targets.shape[0], 3))
    radar_points[:, :2] = cart_targets
    rads.append(radar_points)


# Radar-Lidar Extrinsic Calibration (ICP across multiple scans at once)
# G-N iterations:
max_steps = 50
init_steps = max_steps - 5
robust_k = 0.35

theta = np.array([0, 0, 0]).reshape(3, 1)  # C_lidar_radar

costs = []
thetas = []

matches = [None] * len(rads_to_keep)

for step in range(max_steps):
    # transform lidar data using current estimate
    Cop = so3op.vec2rot(theta)
    
    cost = 0
    # Build (A,b) terms
    A = np.zeros((3, 3))
    b = np.zeros((3, 1))
    
    for idx in range(len(rads_to_keep)):
        radar_points = rads[idx] @ Cop.T

        # estimate correspondences (for initial phase)
        if step < init_steps:
            correspondences = np.zeros((radar_points.shape[0], 2))
            get_nearest_neighbors.get_nearest_neighbors(
                lids[idx],
                radar_points,
                max_pair_d2,
                correspondences,
                num_points,
            )
            matches[idx] = np.copy(correspondences[:num_points[0]])

        for i in range(matches[idx].shape[0]):
            xl = lids[idx][int(matches[idx][i, 1]), :3].reshape(3, 1)
            xr = radar_points[int(matches[idx][i, 0]), :3].reshape(3, 1)
            ebar = xl - xr
            
            u = np.linalg.norm(xr - xl)
            weight = 1 / (1 + (u / robust_k)**2)
            cost += np.linalg.norm(ebar)
#             weight = 1.0
            jac = so3op.hat(xr).T
            A += jac @ jac.T * weight
            b += -1 * jac @ ebar * weight
        cost += np.sqrt(max_pair_d2) * (radar_points.shape[0] - matches[idx].shape[0])
    dtheta = np.linalg.solve(A, b)
    theta = theta + dtheta
    N = 0
    for match in matches:
        N += match.shape[0]
    costs.append(cost)
    thetas.append(theta[2, 0])
    print('step: {:3d} cost: {:.2f} dtheta: {:0.4f} theta: {:0.4f} N: {}'.format(step, cost, dtheta[2, 0], theta[2, 0], N))
print('calibrated update to T_radar_lidar: {:.4f}deg'.format(theta[2, 0] * 180 / np.pi))
print('T_radar_lidar (old):')
print(T_radar_lidar)
T_lidar_radar_update = np.eye(4)
T_lidar_radar_update[:3, :3] = so3op.vec2rot(theta)
T_radar_lidar_new = get_inverse_tf(T_lidar_radar_update) @ T_radar_lidar
print('T_radar_lidar (new):')
print(T_radar_lidar_new)

# SAVE
# np.savetxt('/workspace/Documents/T_radar_lidar.txt', T_radar_lidar_new)
np.savetxt('/workspace/Documents/T_radar_lidar_2023_12_08.txt', T_radar_lidar_new)

x = list(range(len(costs)))
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
axs[0].plot(x, costs, label='cost')
axs[0].legend()
axs[1].plot(x, thetas, label='theta')
axs[1].legend()
plt.show()

# Visualize Radar-Lidar Alignment
idx = 3
rad = rads_to_keep[idx]
ax = rad.visualize(cart_resolution = cart_resolution, cart_pixel_width=cart_pixel_width, show=False)
cart_targets = rads[idx][:, :2]
rad_pixels = convert_to_bev(cart_targets, cart_resolution, cart_pixel_width)
lid_pixels = convert_to_bev(lids[idx][:, :2], cart_resolution, cart_pixel_width)
ax.scatter(rad_pixels[:, 0], rad_pixels[:, 1], color='r', s=1)
ax.scatter(lid_pixels[:, 0], lid_pixels[:, 1], color='b', s=1)
plt.show()








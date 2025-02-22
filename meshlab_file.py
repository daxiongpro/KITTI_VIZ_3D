import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import yaml
import pandas as pd
from kitti_util import *
from matplotlib.lines import Line2D
import cv2


# ==============================================================================
#                                                         POINT_CLOUD_2_BIRDSEYE
# ==============================================================================
def point_cloud_2_top(points,
                      res=0.1,
                      zres=0.3,
                      side_range=(-20., 20 - 0.05),  # left-most to right-most
                      fwd_range=(0., 40. - 0.05),  # back-most to forward-most
                      height_range=(-2., 0.),  # bottom-most to upper-most
                      ):
    """ Creates an birds eye view representation of the point cloud data for MV3D.
    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        zres:        (float)
                    Desired resolution on Z-axis in metres to use.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        numpy array encoding height features , density and intensity.
    """
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    reflectance = points[:, 3]

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    z_max = int((height_range[1] - height_range[0]) / zres)
    top = np.zeros([y_max + 1, x_max + 1, z_max + 1], dtype=np.float32)

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and(
        (x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and(
        (y_points > -side_range[1]), (y_points < -side_range[0]))
    filt = np.logical_and(f_filt, s_filt)

    for i, height in enumerate(np.arange(height_range[0], height_range[1], zres)):
        z_filt = np.logical_and((z_points >= height),
                                (z_points < height + zres))
        zfilter = np.logical_and(filt, z_filt)
        indices = np.argwhere(zfilter).flatten()

        # KEEPERS
        xi_points = x_points[indices]
        yi_points = y_points[indices]
        zi_points = z_points[indices]
        ref_i = reflectance[indices]

        # CONVERT TO PIXEL POSITION VALUES - Based on resolution
        x_img = (-yi_points / res).astype(np.int32)  # x axis is -y in LIDAR
        y_img = (-xi_points / res).astype(np.int32)  # y axis is -x in LIDAR

        # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
        # floor & ceil used to prevent anything being rounded to below 0 after
        # shift
        x_img -= int(np.floor(side_range[0] / res))
        y_img += int(np.floor(fwd_range[1] / res))

        # CLIP HEIGHT VALUES - to between min and max heights
        pixel_values = zi_points - height_range[0]
        # pixel_values = zi_points

        # FILL PIXEL VALUES IN IMAGE ARRAY
        top[y_img, x_img, i] = pixel_values

        # max_intensity = np.max(prs[idx])
        top[y_img, x_img, z_max] = ref_i

    top = (top / np.max(top) * 255).astype(np.uint8)
    return top


def transform_to_img(xmin, xmax, ymin, ymax,
                     res=0.1,
                     side_range=(-20., 20 - 0.05),  # left-most to right-most
                     fwd_range=(0., 40. - 0.05),  # back-most to forward-most
                     ):
    xmin_img = -ymax / res - side_range[0] / res
    xmax_img = -ymin / res - side_range[0] / res
    ymin_img = -xmax / res + fwd_range[1] / res
    ymax_img = -xmin / res + fwd_range[1] / res

    return xmin_img, xmax_img, ymin_img, ymax_img


def draw_point_cloud(ax, points, axes=[0, 1, 2], point_size=0.1, xlim3d=None, ylim3d=None, zlim3d=None):
    """
    Convenient method for drawing various point cloud projections as a part of frame statistics.
    """
    axes_limits = [
        [-20, 80],  # X axis range
        [-40, 40],  # Y axis range
        [-3, 3]  # Z axis range
    ]
    axes_str = ['X', 'Y', 'Z']
    ax.grid(False)

    ax.scatter(*np.transpose(points[:, axes]), s=point_size, c=points[:, 3], cmap='gray')
    ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
    ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
    if len(axes) > 2:
        ax.set_xlim3d(*axes_limits[axes[0]])
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
    else:
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])
    # User specified limits
    if xlim3d != None:
        ax.set_xlim3d(xlim3d)
    if ylim3d != None:
        ax.set_ylim3d(ylim3d)
    if zlim3d != None:
        ax.set_zlim3d(zlim3d)


def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    """
    Return : 3xn in cam2 coordinate
    """
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2


def draw_box(ax, vertices, axes=[0, 1, 2], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.
    
    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for connection in connections:
        ax.plot(*vertices[:, connection], c=color, lw=0.5)


def read_detection(path):
    df = pd.read_csv(path, header=None, sep=' ')
    df.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                  'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y', 'score']
    #     df.loc[df.type.isin(['Truck', 'Van', 'Tram']), 'type'] = 'Car'
    #     df = df[df.type.isin(['Car', 'Pedestrian', 'Cyclist'])]
    df = df[df['type'] == 'Car']
    df.reset_index(drop=True, inplace=True)
    return df


txt_path = r'D:\code\KITTI_VIZ_3D\txt'
data_path = r'D:\code\KITTI_VIZ_3D\data\KITTI\object\testing'
img_id = 1

calib = Calibration(os.path.join(data_path, 'calib/%06d.txt' % img_id))

path = os.path.join(data_path, 'velodyne/%06d.bin' % img_id)

path_img = os.path.join(data_path, '/image_2/%06d.png' % img_id)

points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)

# df = read_detection('/home2/yang_ye/wei3/%06d.txt'%img_id)
df = read_detection(os.path.join(txt_path, '%06d.txt' % img_id))

image_shape = cv2.imread(path_img).shape

rect = calib.R0
rect_ = np.zeros((4, 4), dtype=np.float)
rect_[0:3, 0:3] = rect
rect_[3, 3] = 1

P2 = calib.P
P2_ = np.zeros((4, 4), dtype=np.float)
P2_[0:3, :] = P2
P2_[3, 3] = 1

Trv2c = calib.V2C
Trv2c_ = np.zeros((4, 4), dtype=np.float)
Trv2c_[0:3, :] = Trv2c
Trv2c_[3, 3] = 1

#######remove outside points############
points = remove_outside_points(points, rect_, Trv2c_, P2_, image_shape)
#######remove outside points############

df.head()

print(len(df))

##############plot 3D box#####################
save_ = np.zeros((7, 0), dtype=float)
for o in range(len(df)):
    corners_3d_cam2 = compute_3d_box_cam2(*df.loc[o, ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
    corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)

    rot_y = np.rad2deg(*df.loc[o, ['rot_y']])
    print(rot_y)

    x1, x2, x3, x4 = corners_3d_cam2[0, 0:4]
    y1, y2, y3, y4 = corners_3d_cam2[1, 0:4]
    z1, z2, z3, z4 = corners_3d_cam2[2, 0:4]

    x11, x22, x33, x44 = corners_3d_cam2[0, 4:8]
    y11, y22, y33, y44 = corners_3d_cam2[1, 4:8]
    z11, z22, z33, z44 = corners_3d_cam2[2, 4:8]

    save_points1 = compute_3D_line(x1, x2, y1, y2, z1, z2)
    save_points2 = compute_3D_line(x2, x3, y2, y3, z2, z3)
    save_points3 = compute_3D_line(x3, x4, y3, y4, z3, z4)
    save_points4 = compute_3D_line(x4, x1, y4, y1, z4, z1)

    save_points = np.concatenate((save_points1, save_points2, save_points3, save_points4), axis=1)

    save_points11 = compute_3D_line(x11, x22, y11, y22, z11, z22)
    save_points22 = compute_3D_line(x22, x33, y22, y33, z22, z33)
    save_points33 = compute_3D_line(x33, x44, y33, y44, z33, z44)
    save_points44 = compute_3D_line(x44, x11, y44, y11, z44, z11)

    save_points_ = np.concatenate((save_points11, save_points22, save_points33, save_points44), axis=1)

    save_points111 = compute_3D_line(x1, x11, y1, y11, z1, z11)
    save_points222 = compute_3D_line(x2, x22, y2, y22, z2, z22)
    save_points333 = compute_3D_line(x3, x33, y3, y33, z3, z33)
    save_points444 = compute_3D_line(x4, x44, y4, y44, z4, z44)
    save_points__ = np.concatenate((save_points111, save_points222, save_points333, save_points444), axis=1)

    n1, n2, m1, m2, l1, l2 = (x11 + x22) / 2, (x33 + x44) / 2, (y11 + y22) / 2, (y33 + y44) / 2, (z11 + z22) / 2, (
            z33 + z44) / 2
    #################
    save_points_center = compute_3D_line_(n1, n2, m1, m2, l1, l2)
    #################

    # if rot_y >= 0:
    save_points_center_ = save_points_center[:, 0:150]
    # else:
    #    save_points_center_ = save_points_center[:,150:300]    
    save_ = np.concatenate((save_, save_points, save_points_, save_points__, save_points_center_), axis=1)

    # print(save_points.shape)

pad = np.zeros((points.shape[0], 3), dtype=np.float)

save_points_velo = corners_3d_velo = calib.project_rect_to_velo(save_[0:3, :].T)

save_[0:3, :] = save_points_velo.T
points = np.concatenate((points, pad), axis=1)
points = np.concatenate((points, save_.T), axis=0)
print(points.shape)
##############plot 3D box#####################  

# save points to txt file
np.savetxt(str(img_id) + '_test_file_Lidar.txt', points)

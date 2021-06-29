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

path_img = os.path.join(data_path, '/image_2/%06d.png' % img_id)

df = read_detection(os.path.join(txt_path, '%06d.txt' % img_id))

image = cv2.imread(path_img)
df.head()

print(len(df))

##############plot 3D box#####################
for o in range(len(df)):
    corners_3d_cam2 = compute_3d_box_cam2(*df.loc[o, ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
    pts_2d = calib.project_rect_to_image(corners_3d_cam2.T)
    image = draw_projected_box3d(image, pts_2d, color=(255, 0, 255), thickness=1)

cv2.imwrite(str(img_id) + '_test_file_img.png', image)

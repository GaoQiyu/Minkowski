import matplotlib.pyplot as plt
import os
import numpy as np
from numpy.linalg import matrix_rank, inv
from plyfile import PlyData, PlyElement
import pandas as pd
CLASS_LABELS = ('wall', 'floor', 'stairs', 'beam', 'chair', 'sofa', 'table', 'door', 'window', 'bookcase', 'column', 'clutter', 'ceiling', 'board')
COLOR_MAP_RGB = {
    0: (0., 0., 0.),
    1: (227., 209., 212.),
    2: (143., 100., 21.),
    3: (242., 171., 39.),
    4: (100., 143., 156.),
    5: (3., 3., 3.),
    6: (255., 255., 0.),
    7: (255., 0., 17.),
    8: (45., 204., 193.),
    9: (204., 45., 191.),
    10: (85., 45., 204.),
    11: (45., 204., 85.),
    12: (81., 94., 51.),
    13: (0., 9., 255.),
    14: (122., 126., 240.)
}
IGNORE_COLOR = (0, 0, 0)


def color2label():
    fig = plt.figure()
    fig.suptitle('color2label', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)

    for key in COLOR_MAP_RGB:
        COLOR_MAP_RGB[key] = (
        COLOR_MAP_RGB[key][0] / 256, COLOR_MAP_RGB[key][1] / 256, COLOR_MAP_RGB[key][2] / 256)

    font = 20
    ax.text(0, 0.9, CLASS_LABELS[0], verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,
            color=COLOR_MAP_RGB[1], fontsize=font)
    ax.text(0.25, 0.9, CLASS_LABELS[1], verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,
            color=COLOR_MAP_RGB[2], fontsize=font)
    ax.text(0.5, 0.9, CLASS_LABELS[2], verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,
            color=COLOR_MAP_RGB[3], fontsize=font)
    ax.text(0.75, 0.9, CLASS_LABELS[3], verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,
            color=COLOR_MAP_RGB[4], fontsize=font)
    ax.text(0, 0.7, CLASS_LABELS[4], verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,
            color=COLOR_MAP_RGB[5], fontsize=font)
    ax.text(0.25, 0.7, CLASS_LABELS[5], verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,
            color=COLOR_MAP_RGB[6], fontsize=font)
    ax.text(0.5, 0.7, CLASS_LABELS[6], verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,
            color=COLOR_MAP_RGB[7], fontsize=font)
    ax.text(0.75, 0.7, CLASS_LABELS[7], verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,
            color=COLOR_MAP_RGB[8], fontsize=font)
    ax.text(0, 0.5, CLASS_LABELS[8], verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,
            color=COLOR_MAP_RGB[9], fontsize=font)
    ax.text(0.25, 0.5, CLASS_LABELS[9], verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,
            color=COLOR_MAP_RGB[10], fontsize=font)
    ax.text(0.5, 0.5, CLASS_LABELS[10], verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,
            color=COLOR_MAP_RGB[11], fontsize=font)
    ax.text(0.75, 0.5, CLASS_LABELS[11], verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,
            color=COLOR_MAP_RGB[12], fontsize=font)
    ax.text(0, 0.3, CLASS_LABELS[12], verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,
            color=COLOR_MAP_RGB[13], fontsize=font)
    ax.text(0.25, 0.3, CLASS_LABELS[13], verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,
            color=COLOR_MAP_RGB[14], fontsize=font)

    plt.axis('off')
    plt.savefig("./color2label.jpg")
    plt.show()


def read_plyfile(filepath):
  """Read ply file and return it as numpy array. Returns None if emtpy."""
  with open(filepath, 'rb') as f:
    plydata = PlyData.read(f)
  if plydata.elements:
    return pd.DataFrame(plydata.elements[0].data).values


def save_point_cloud(points_3d, filename, binary=True, with_label=False, verbose=True):
  """Save an RGB point cloud as a PLY file.

  Args:
    points_3d: Nx6 matrix where points_3d[:, :3] are the XYZ coordinates and points_3d[:, 4:] are
        the RGB values. If Nx3 matrix, save all points with [128, 128, 128] (gray) color.
  """
  assert points_3d.ndim == 2
  if with_label:
    assert points_3d.shape[1] == 7
    python_types = (float, float, float, int, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                 ('blue', 'u1'), ('label', 'u1')]
  else:
    if points_3d.shape[1] == 3:
      gray_concat = np.tile(np.array([128], dtype=np.uint8), (points_3d.shape[0], 3))
      points_3d = np.hstack((points_3d, gray_concat))
    assert points_3d.shape[1] == 6
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                 ('blue', 'u1')]
  if binary is True:
    # Format into NumPy structured array
    vertices = []
    for row_idx in range(points_3d.shape[0]):
      cur_point = points_3d[row_idx]
      vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
    vertices_array = np.array(vertices, dtype=npy_types)
    el = PlyElement.describe(vertices_array, 'vertex')

    # Write
    PlyData([el]).write(filename)
  else:
    # PlyData([el], text=True).write(filename)
    with open(filename, 'w') as f:
      f.write('ply\n'
              'format ascii 1.0\n'
              'element vertex %d\n'
              'property float x\n'
              'property float y\n'
              'property float z\n'
              'property uchar red\n'
              'property uchar green\n'
              'property uchar blue\n'
              'property uchar alpha\n'
              'end_header\n' % points_3d.shape[0])
      for row_idx in range(points_3d.shape[0]):
        X, Y, Z, R, G, B = points_3d[row_idx]
        f.write('%f %f %f %d %d %d 0\n' % (X, Y, Z, R, G, B))
  if verbose is True:
    print('Saved point cloud to: %s' % filename)


class Camera(object):

  def __init__(self, intrinsics):
    self._intrinsics = intrinsics
    self._camera_matrix = self.build_camera_matrix(self.intrinsics)
    self._K_inv = inv(self.camera_matrix)

  @staticmethod
  def build_camera_matrix(intrinsics):
    """Build the 3x3 camera matrix K using the given intrinsics.

    Equation 6.10 from HZ.
    """
    f = intrinsics['focal_length']
    pp_x = intrinsics['pp_x']
    pp_y = intrinsics['pp_y']

    K = np.array([[f, 0, pp_x], [0, f, pp_y], [0, 0, 1]], dtype=np.float32)
    # K[:, 0] *= -1.  # Step 1 of Kyle
    assert matrix_rank(K) == 3
    return K

  @staticmethod
  def extrinsics2RT(extrinsics):
    """Convert extrinsics matrix to separate rotation matrix R and translation vector T.
    """
    assert extrinsics.shape == (4, 4)
    R = extrinsics[:3, :3]
    T = extrinsics[3, :3]
    R = np.copy(R)
    T = np.copy(T)
    T = T.reshape(3, 1)
    R[0, :] *= -1.  # Step 1 of Kyle
    T *= 100.  # Convert from m to cm
    return R, T

  def project(self, points_3d, extrinsics=None):
    """Project a 3D point in camera coordinates into the camera/image plane.

    Args:
      point_3d:
    """
    if extrinsics is not None:  # Map points to camera coordinates
      points_3d = self.world2camera(extrinsics, points_3d)

    # TODO: Make sure to handle homogeneous AND non-homogeneous coordinate points
    # TODO: Consider handling a set of points
    raise NotImplementedError

  def backproject(self,
                  depth_map,
                  labels=None,
                  max_depth=None,
                  max_height=None,
                  min_height=None,
                  rgb_img=None,
                  extrinsics=None,
                  prune=True):
    """Backproject a depth map into 3D points (camera coordinate system). Attach color if RGB image
    is provided, otherwise use gray [128 128 128] color.

    Does not show points at Z = 0 or maximum Z = 65535 depth.

    Args:
      labels: Tensor with the same shape as depth map (but can be 1-channel or 3-channel).
      max_depth: Maximum depth in cm. All pts with depth greater than max_depth will be ignored.
      max_height: Maximum height in cm. All pts with height greater than max_height will be ignored.

    Returns:
      points_3d: Numpy array of size Nx3 (XYZ) or Nx6 (XYZRGB).
    """
    if labels is not None:
      assert depth_map.shape[:2] == labels.shape[:2]
      if (labels.ndim == 2) or ((labels.ndim == 3) and (labels.shape[2] == 1)):
        n_label_channels = 1
      elif (labels.ndim == 3) and (labels.shape[2] == 3):
        n_label_channels = 3

    if rgb_img is not None:
      assert depth_map.shape[:2] == rgb_img.shape[:2]
    else:
      rgb_img = np.ones_like(depth_map, dtype=np.uint8) * 255

    # Convert from 1-channel to 3-channel
    if (rgb_img.ndim == 3) and (rgb_img.shape[2] == 1):
      rgb_img = np.tile(rgb_img, [1, 1, 3])

    # Convert depth map to single channel if it is multichannel
    if (depth_map.ndim == 3) and depth_map.shape[2] == 3:
      depth_map = np.squeeze(depth_map[:, :, 0])
    depth_map = depth_map.astype(np.float32)

    # Get image dimensions
    H, W = depth_map.shape

    # Create meshgrid (pixel coordinates)
    Z = depth_map
    A, B = np.meshgrid(range(W), range(H))
    ones = np.ones_like(A)
    grid = np.concatenate((A[:, :, np.newaxis], B[:, :, np.newaxis], ones[:, :, np.newaxis]),
                          axis=2)
    grid = grid.astype(np.float32) * Z[:, :, np.newaxis]
    # Nx3 where each row is (a*Z, b*Z, Z)
    grid_flattened = grid.reshape((-1, 3))
    grid_flattened = grid_flattened.T  # 3xN where each col is (a*Z, b*Z, Z)
    prod = np.dot(self.K_inv, grid_flattened)
    XYZ = np.concatenate((prod[:2, :].T, Z.flatten()[:, np.newaxis]), axis=1)  # Nx3
    XYZRGB = np.hstack((XYZ, rgb_img.reshape((-1, 3))))
    points_3d = XYZRGB

    if labels is not None:
      labels_reshaped = labels.reshape((-1, n_label_channels))

    # Prune points
    if prune is True:
      valid = []
      for idx in range(points_3d.shape[0]):
        cur_y = points_3d[idx, 1]
        cur_z = points_3d[idx, 2]
        if (cur_z == 0) or (cur_z == 65535):  # Don't show things at 0 distance or max distance
          continue
        elif (max_depth is not None) and (cur_z > max_depth):
          continue
        elif (max_height is not None) and (cur_y > max_height):
          continue
        elif (min_height is not None) and (cur_y < min_height):
          continue
        else:
          valid.append(idx)
      points_3d = points_3d[np.asarray(valid)]
      if labels is not None:
        labels_reshaped = labels_reshaped[np.asarray(valid)]

    if extrinsics is not None:
      points_3d = self.camera2world(extrinsics, points_3d)

    if labels is not None:
      points_3d_labels = np.hstack((points_3d[:, :3], labels_reshaped))
      return points_3d, points_3d_labels
    else:
      return points_3d

  @staticmethod
  def _camera2world_transform(no_rgb_points_3d, R, T):
    points_3d_world = (np.dot(R.T, no_rgb_points_3d.T) - T).T  # Nx3
    return points_3d_world

  @staticmethod
  def _world2camera_transform(no_rgb_points_3d, R, T):
    points_3d_world = (np.dot(R, no_rgb_points_3d.T + T)).T  # Nx3
    return points_3d_world

  def _transform_points(self, points_3d, extrinsics, transform):
    """Base/wrapper method for transforming points using R and T.
    """
    assert points_3d.ndim == 2
    orig_points_3d = points_3d
    points_3d = np.copy(orig_points_3d)
    if points_3d.shape[1] == 6:  # XYZRGB
      points_3d = points_3d[:, :3]
    elif points_3d.shape[1] == 3:  # XYZ
      points_3d = points_3d
    else:
      raise ValueError('3D points need to be XYZ or XYZRGB.')

    R, T = self.extrinsics2RT(extrinsics)
    points_3d_world = transform(points_3d, R, T)

    # Add color again (if appropriate)
    if orig_points_3d.shape[1] == 6:  # XYZRGB
      points_3d_world = np.hstack((points_3d_world, orig_points_3d[:, -3:]))
    return points_3d_world

  def camera2world(self, extrinsics, points_3d):
    """Transform from camera coordinates (3D) to world coordinates (3D).

    Args:
      points_3d: Nx3 or Nx6 matrix of N points with XYZ or XYZRGB values.
    """
    return self._transform_points(points_3d, extrinsics, self._camera2world_transform)

  def world2camera(self, extrinsics, points_3d):
    """Transform from world coordinates (3D) to camera coordinates (3D).
    """
    return self._transform_points(points_3d, extrinsics, self._world2camera_transform)

  @property
  def intrinsics(self):
    return self._intrinsics

  @property
  def camera_matrix(self):
    return self._camera_matrix

  @property
  def K_inv(self):
    return self._K_inv


def colorize_pointcloud(xyz, label, ignore_label=255):
  assert label[label != ignore_label].max() < len(COLOR_MAP_RGB), 'Not enough colors.'
  label_rgb = np.array([COLOR_MAP_RGB[i] if i != ignore_label else IGNORE_COLOR for i in label])
  return np.hstack((xyz, label_rgb))


class PlyWriter(object):

  POINTCLOUD_DTYPE = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                      ('blue', 'u1')]

  @classmethod
  def read_txt(cls, txtfile):
    # Read txt file and parse its content.
    with open(txtfile) as f:
      pointcloud = [l.split() for l in f]
    # Load point cloud to named numpy array.
    pointcloud = np.array(pointcloud).astype(np.float32)
    assert pointcloud.shape[1] == 6
    xyz = pointcloud[:, :3].astype(np.float32)
    rgb = pointcloud[:, 3:].astype(np.uint8)
    return xyz, rgb

  @staticmethod
  def write_ply(array, filepath):
    ply_el = PlyElement.describe(array, 'vertex')
    target_path, _ = os.path.split(filepath)
    if target_path != '' and not os.path.exists(target_path):
      os.makedirs(target_path)
    PlyData([ply_el]).write(filepath)

  @classmethod
  def write_vertex_only_ply(cls, vertices, filepath):
    # assume that points are N x 3 np array for vertex locations
    color = 255 * np.ones((len(vertices), 3))
    pc_points = np.array([tuple(p) for p in np.concatenate((vertices, color), axis=1)],
                         dtype=cls.POINTCLOUD_DTYPE)
    cls.write_ply(pc_points, filepath)

  @classmethod
  def write_ply_vert_color(cls, vertices, colors, filepath):
    # assume that points are N x 3 np array for vertex locations
    pc_points = np.array([tuple(p) for p in np.concatenate((vertices, colors), axis=1)],
                         dtype=cls.POINTCLOUD_DTYPE)
    cls.write_ply(pc_points, filepath)

  @classmethod
  def concat_label(cls, target, xyz, label):
    subpointcloud = np.concatenate([xyz, label], axis=1)
    subpointcloud = np.array([tuple(l) for l in subpointcloud], dtype=cls.POINTCLOUD_DTYPE)
    return np.concatenate([target, subpointcloud], axis=0)

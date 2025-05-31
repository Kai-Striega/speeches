import numpy as np
from scipy.interpolate import RegularGridInterpolator

n_points_in_dim = 2 ** 14  # 16,384 points
rng = np.random.default_rng(42)
x = np.arange(n_points_in_dim)
y = np.arange(n_points_in_dim)
z = rng.uniform(size=(n_points_in_dim, n_points_in_dim))

xg, yg = np.meshgrid(x, y, indexing='ij', sparse=True)
rgi = RegularGridInterpolator((x, y), z.T, method='linear')
rgi((xg, yg))

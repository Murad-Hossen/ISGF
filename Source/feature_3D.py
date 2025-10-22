import numpy as np
import pandas as pd
from skimage import draw
import porespy as ps
from scipy.spatial import ConvexHull
import ot

def read_swc(file_path, remove_axon=False, remove_soma=False):
    """Read an SWC file and return a DataFrame with neuron morphology data."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                fields = line.split()
                if len(fields) >= 7:
                    data.append([int(fields[0]), int(fields[1]), float(fields[2]),
                                 float(fields[3]), float(fields[4]), float(fields[5]),
                                 int(fields[6])])
    columns = ['SampleID', 'TypeID', 'x', 'y', 'z', 'radius', 'parent']
    df = pd.DataFrame(data, columns=columns)
    if remove_axon:
        df = df[df['TypeID'] != 2].reset_index(drop=True)
    if remove_soma:
        df = df[df['TypeID'] != 1].reset_index(drop=True)
    valid_parents = set(df['SampleID'])
    df.loc[~df['parent'].isin(valid_parents) & (df['parent'] != -1), 'parent'] = -1
    return df

def add_ball(im, cy, cx, cz, radius):
    """Add a 3D ball to a volume at given coordinates with specified radius."""
    if radius <= 0:
        return
    r = int(np.ceil(radius))
    y_min = max(0, cy - r)
    y_max = min(im.shape[0], cy + r + 1)
    x_min = max(0, cx - r)
    x_max = min(im.shape[1], cx + r + 1)
    z_min = max(0, cz - r)
    z_max = min(im.shape[2], cz + r + 1)
    yy, xx, zz = np.meshgrid(np.arange(y_min, y_max),
                             np.arange(x_min, x_max),
                             np.arange(z_min, z_max),
                             indexing='ij')
    mask = (yy - cy)**2 + (xx - cx)**2 + (zz - cz)**2 <= radius**2
    im[yy[mask], xx[mask], zz[mask]] = True

def swc_to_binary_volume(swc_data, volume_size=700, radius_scale=2.0, padding_factor=1.1):
    """Convert SWC data to a 3D binary volume."""
    x_min, x_max = swc_data['x'].min(), swc_data['x'].max()
    y_min, y_max = swc_data['y'].min(), swc_data['y'].max()
    z_min, z_max = swc_data['z'].min(), swc_data['z'].max()
    delta_x = x_max - x_min
    delta_y = y_max - y_min
    delta_z = z_max - z_min
    L = max(delta_x, delta_y, delta_z)
    voxel_size = L * padding_factor / volume_size
    extent_x = delta_x * padding_factor
    extent_y = delta_y * padding_factor
    extent_z = delta_z * padding_factor
    nx = int(np.ceil(extent_x / voxel_size))
    ny = int(np.ceil(extent_y / voxel_size))
    nz = int(np.ceil(extent_z / voxel_size))
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    start_x = x_center - extent_x/2
    start_y = y_center - extent_y/2
    start_z = z_center - extent_z/2
    def scale_coord(coord, start, size):
        idx = int(np.floor((coord - start) / voxel_size))
        return max(0, min(idx, size - 1))
    im = np.zeros((ny, nx, nz), dtype=bool)
    for _, row in swc_data.iterrows():
        if row['TypeID'] == 1:
            cx = scale_coord(row['x'], start_x, nx)
            cy = scale_coord(row['y'], start_y, ny)
            cz = scale_coord(row['z'], start_z, nz)
            radius = max(1, row['radius'] / voxel_size * radius_scale)
            add_ball(im, cy, cx, cz, radius)
        elif row['parent'] != -1:
            parent_row = swc_data[swc_data['SampleID'] == row['parent']]
            if not parent_row.empty:
                cx1 = scale_coord(row['x'], start_x, nx)
                cy1 = scale_coord(row['y'], start_y, ny)
                cz1 = scale_coord(row['z'], start_z, nz)
                cx2 = scale_coord(parent_row['x'].iloc[0], start_x, nx)
                cy2 = scale_coord(parent_row['y'].iloc[0], start_y, ny)
                cz2 = scale_coord(parent_row['z'].iloc[0], start_z, nz)
                avg_radius = (row['radius'] + parent_row['radius'].iloc[0]) / 2
                radius = max(1, avg_radius / voxel_size * radius_scale)
                yy, xx, zz = draw.line_nd((cy1, cx1, cz1), (cy2, cx2, cz2))
                for y, x, z in zip(yy, xx, zz):
                    add_ball(im, y, x, z, radius)
    return im

def compute_fractal_dimension(swc_file, folder):
    """Compute fractal dimension for neuron SWC data using 3D box-counting."""
    try:
        remove_axon = (folder == 'Retina')
        swc_data = read_swc(swc_file, remove_axon=remove_axon, remove_soma=True)
        if swc_data.empty:
            return 0.0
        vol = swc_to_binary_volume(swc_data, volume_size=700, radius_scale=2.0, padding_factor=1.1)
        box_sizes = np.logspace(np.log10(5), np.log10(50), num=5, dtype=int)
        boxcount = ps.metrics.boxcount(vol, bins=box_sizes)
        stable_region = (boxcount.size >= 5) & (boxcount.size <= 50)
        if np.any(stable_region):
            return np.mean(boxcount.slope[stable_region])
        else:
            return 0.0
    except Exception as e:
        print(f"Error computing FD for {swc_file}: {e}")
        return 0.0

def compute_ot_distance_1d(data1, data2):
    """Compute 1D optimal transport distance between two distributions."""
    n_bins = max(len(data1), len(data2))
    bins = np.linspace(min(min(data1), min(data2)), max(max(data1), max(data2)), n_bins)
    hist1, _ = np.histogram(data1, bins=bins, density=True)
    hist2, _ = np.histogram(data2, bins=bins, density=True)
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    M = ot.dist(bin_centers.reshape(-1, 1), bin_centers.reshape(-1, 1), metric='euclidean')
    return ot.emd2(hist1, hist2, M)

def upper_triangular(df):
    """Extract upper triangular part of a DataFrame."""
    mask = np.triu(np.ones(df.shape), k=1).astype(bool)
    df_upper = df.where(mask)
    return df_upper

def average_bending_energy_curve(c, closed=False, eps=1e-12):
    """
    Compute the line integral of the squared curvature of a polyline:
        ∫ κ(s)^2 ds / len(c)
    using a discrete arclength-based scheme on edges/tangents.
    """
    c = np.asarray(c, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] < 2:
        raise ValueError("c must be an (N, d) array with d >= 2.")
    N, d = c.shape
    if N < 3:
        raise ValueError("Need at least 3 points to estimate curvature.")
    if closed:
        e = np.roll(c, -1, axis=0) - c
        ℓ = np.linalg.norm(e, axis=1)
        length = np.sum(ℓ)
        ℓ = np.maximum(ℓ, eps)
        t = e / ℓ[:, None]
        Δt = t - np.roll(t, 1, axis=0)
        avg_ℓ = 0.5 * (ℓ + np.roll(ℓ, 1))
        avg_ℓ = np.maximum(avg_ℓ, eps)
        energy = np.sum(np.einsum('ij,ij->i', Δt, Δt) / avg_ℓ) / length
        return float(energy)
    else:
        e = c[1:] - c[:-1]
        ℓ = np.linalg.norm(e, axis=1)
        length = np.sum(ℓ)
        ℓ = np.maximum(ℓ, eps)
        t = e / ℓ[:, None]
        Δt = t[1:] - t[:-1]
        avg_ℓ = 0.5 * (ℓ[1:] + ℓ[:-1])
        avg_ℓ = np.maximum(avg_ℓ, eps)
        energy = np.sum(np.einsum('ij,ij->i', Δt, Δt) / avg_ℓ) / length
        return float(energy)
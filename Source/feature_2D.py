import numpy as np
from skimage import draw
import porespy as ps

# === Road Network Functions ===

def graph_to_binary_image(G, image_size=700):
    """Convert road network graph to a 2D binary image with enhanced clarity."""
    coords = np.array([(data['x'], data['y']) for node, data in G.nodes(data=True)])
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    extent = max(x_max - x_min, y_max - y_min) * 1.1
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    im = np.zeros((image_size, image_size), dtype=bool)

    def scale_coord(coord, center, extent):
        return int(((coord - center) / extent + 0.5) * (image_size - 1))
    
    for u, v, data in G.edges(data=True):
        if 'geometry' in data:
            xs, ys = data['geometry'].xy
            coords = np.array(list(zip(xs, ys)))
        else:
            x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
            x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
            coords = np.array([[x1, y1], [x2, y2]])
  
        for i in range(len(coords) - 1):
            x0, y0 = scale_coord(coords[i, 0], x_center, extent), scale_coord(coords[i, 1], y_center, extent)
            x1, y1 = scale_coord(coords[i + 1, 0], x_center, extent), scale_coord(coords[i + 1, 1], y_center, extent)
            rr, cc = draw.line(y0, x0, y1, x1)
            for r, c in zip(rr, cc):
                if 0 <= r < image_size and 0 <= c < image_size:
                    rr_disk, cc_disk = draw.disk((r, c), radius=2, shape=(image_size, image_size))
                    im[rr_disk, cc_disk] = True
    return im

def get_edge_coords(G, u, v):
    """Extract coordinates of an edge in the graph."""
    data = G.get_edge_data(u, v)
    if not data:
        return np.array([])
    key = list(data.keys())[0]
    data = data[key]
    if 'geometry' in data:
        xs, ys = data['geometry'].xy
        coords = np.array(list(zip(xs, ys)))
    else:
        coords = np.array([[G.nodes[u]['x'], G.nodes[u]['y']], [G.nodes[v]['x'], G.nodes[v]['y']]])
    u_pos = np.array([G.nodes[u]['x'], G.nodes[u]['y']])
    dist_start = np.linalg.norm(coords[0] - u_pos)
    dist_end = np.linalg.norm(coords[-1] - u_pos)
    if dist_start > dist_end:
        coords = coords[::-1]
    return coords

def compute_branches(G):
    """Compute lengths and polylines of branches in the road network."""
    adj = {n: list(G.neighbors(n)) for n in G}
    degrees = {n: len(adj[n]) for n in adj}
    key_nodes = set(n for n in degrees if degrees[n] != 2)
    branch_lengths = []
    branch_polylines = []
    visited = set()
    for kn in key_nodes:
        for nb in adj[kn]:
            edge = frozenset([kn, nb])
            if edge in visited:
                continue
            visited.add(edge)
            coords = get_edge_coords(G, kn, nb)
            if len(coords) == 0:
                continue
            poly = coords.tolist()
            edge_data = G.get_edge_data(kn, nb)
            key = list(edge_data.keys())[0]
            seg_len = edge_data[key]['length']
            added_length = seg_len
            current = nb
            prev = kn
            while True:
                neighbors = [nxt for nxt in adj[current] if nxt != prev]
                if len(neighbors) != 1:
                    break
                nxt = neighbors[0]
                edge = frozenset([current, nxt])
                if edge in visited:
                    break
                visited.add(edge)
                edge_data = G.get_edge_data(current, nxt)
                if not edge_data:
                    break
                key = list(edge_data.keys())[0]
                edge_len = edge_data[key]['length']
                added_length += edge_len
                new_coords = get_edge_coords(G, current, nxt)
                if len(new_coords) == 0:
                    break
                if np.allclose(poly[-1], new_coords[0], atol=1e-6):
                    poly.extend(new_coords[1:].tolist())
                else:
                    poly.extend(new_coords.tolist())
                prev = current
                current = nxt
            branch_len = added_length
            if branch_len > 0:
                branch_lengths.append(branch_len)
                branch_polylines.append(np.array(poly))
    return np.array(branch_lengths), branch_polylines

def compute_fractal_dimension(G):
    """Compute fractal dimension using box-counting on a 2D binary image in specified range."""
    try:
        binary_image = graph_to_binary_image(G, image_size=700)
        box_sizes = np.logspace(np.log10(5), np.log10(50), num=5, dtype=int)
        boxcount = ps.metrics.boxcount(binary_image, bins=box_sizes)
        fds = boxcount.slope
        return np.mean(fds) if len(fds) > 0 else 0.0
    except Exception as e:
        print(f"Fractal dimension calculation error: {e}")
        return 0.0

# === Astrocyte-Specific Functions ===

def graph_to_binary_image_astro(graph, image_size=700, radius_scale=2.0):
    """Convert astrocyte graph to a 2D binary image."""
    all_x = []
    all_y = []
    for s, e in graph.edges():
        pts = graph[s][e]['pts']
        if pts.ndim != 2 or pts.shape[1] < 2:
            continue
        all_x.extend(pts[:,1])
        all_y.extend(pts[:,0])
    if not all_x:
        return np.zeros((image_size, image_size), dtype=bool)
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    extent = max(x_max - x_min, y_max - y_min) * 1.1
    if extent == 0:
        return np.zeros((image_size, image_size), dtype=bool)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    im = np.zeros((image_size, image_size), dtype=bool)
    def scale_coord(coord, center, extent):
        return int(((coord - center) / extent + 0.5) * (image_size - 1))
    radius = max(1, int(radius_scale))
    for s, e in graph.edges():
        pts = graph[s][e]['pts']
        if len(pts) < 2:
            continue
        xs = pts[:,1]
        ys = pts[:,0]
        for i in range(len(pts) - 1):
            x1 = scale_coord(xs[i], x_center, extent)
            y1 = scale_coord(ys[i], y_center, extent)
            x2 = scale_coord(xs[i+1], x_center, extent)
            y2 = scale_coord(ys[i+1], y_center, extent)
            rr, cc = draw.line(y1, x1, y2, x2)
            valid = (0 <= rr) & (rr < image_size) & (0 <= cc) & (cc < image_size)
            rr, cc = rr[valid], cc[valid]
            for j in range(len(rr)):
                r, c = rr[j], cc[j]
                rr_disk, cc_disk = draw.disk((r, c), radius, shape=im.shape)
                im[rr_disk, cc_disk] = True
    return im

def compute_branch_polylines_astro(graph):
    """Compute polylines for branches in the astrocyte graph."""
    branch_polylines = []
    visited_edges = set()
    for node in list(graph.nodes()):
        if graph.degree[node] != 2:
            for neighbor in list(graph.neighbors(node)):
                current = node
                next_node = neighbor
                edge = tuple(sorted((current, next_node)))
                if edge in visited_edges:
                    continue
                visited_edges.add(edge)
                pts = graph[current][next_node]['pts']
                o_current = graph.nodes[current]['o']
                dist_start = np.linalg.norm(pts[0] - o_current)
                dist_end = np.linalg.norm(pts[-1] - o_current)
                if dist_start > dist_end:
                    pts = pts[::-1].copy()
                poly = pts.tolist()
                current = next_node
                while True:
                    neighbors = list(graph.neighbors(current))
                    next_cand = [n for n in neighbors if tuple(sorted((current, n))) not in visited_edges]
                    if len(next_cand) != 1:
                        break
                    next_node = next_cand[0]
                    edge = tuple(sorted((current, next_node)))
                    visited_edges.add(edge)
                    pts = graph[current][next_node]['pts']
                    o_current = graph.nodes[current]['o']
                    dist_start = np.linalg.norm(pts[0] - o_current)
                    dist_end = np.linalg.norm(pts[-1] - o_current)
                    if dist_start > dist_end:
                        pts = pts[::-1].copy()
                    if np.allclose(poly[-1], pts[0]):
                        poly.extend(pts[1:].tolist())
                    else:
                        poly.extend(pts.tolist())
                    current = next_node
                poly_array = np.array(poly)
                if poly_array.shape[0] >= 3:
                    branch_polylines.append(poly_array)
    return branch_polylines

def compute_fractal_dimension_astro(graph):
    """Compute fractal dimension for astrocyte graph using box-counting."""
    try:
        im = graph_to_binary_image_astro(graph, image_size=700, radius_scale=2.0)
        box_sizes = np.logspace(np.log10(5), np.log10(50), num=5, dtype=int)
        boxcount = ps.metrics.boxcount(im, bins=box_sizes)
        stable_region = (boxcount.size >= 5) & (boxcount.size <= 50)
        fd = np.mean(boxcount.slope[stable_region]) if np.any(stable_region) else 0.0
        return fd
    except Exception as e:
        print(f"Fractal dimension error: {e}")
        return 0.0

# === Common Function (2D/3D Compatible) ===

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
def average_bending_energy_curve(c, closed=False, eps=1e-12):
    """
    Compute the line integral of the squared curvature of a polyline:
        ∫ κ(s)^2 ds / len(c)
    using a discrete arclength-based scheme on edges/tangents.
    Parameters
    ----------
    c : (N, d) array_like
        Vertices of the curve (d=2 or d=3).
    closed : bool, default True
        If True, treat the curve as closed (cyclic indices).
        If False, treat as open and use only interior vertices.
    eps : float, default 1e-12
        Small clamp to avoid division by zero for tiny edges.
    Returns
    -------
    float
        Discrete approximation of ∫ κ^2 ds.
    Notes
    -----
    Let e_i = c_{i+1} - c_i, ℓ_i = ||e_i||, and t_{i+1/2} = e_i / ℓ_i.
    For a closed curve, the per-vertex contribution at vertex i is:
        ||t_{i+1/2} - t_{i-1/2}||^2 / ((ℓ_i + ℓ_{i-1})/2).
    For an open curve, the sum is taken over interior vertices only.
    """
    c = np.asarray(c, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] < 2:
        raise ValueError("c must be an (N, d) array with d >= 2.")
    N, d = c.shape
    if N < 3:
        raise ValueError("Need at least 3 points to estimate curvature.")
    if closed:
        # Edges, lengths, and unit tangents on edges
        e = np.roll(c, -1, axis=0) - c # (N, d), edges i -> i+1
        ℓ = np.linalg.norm(e, axis=1)
        length = np.sum(ℓ)
        ℓ = np.maximum(ℓ, eps)
        t = e / ℓ[:, None] # unit tangents on edges
        # Δt at vertices (i): t_{i+1/2} - t_{i-1/2}
        Δt = t - np.roll(t, 1, axis=0) # (N, d)
        # Dual (Voronoi) cell length around vertex i
       
        avg_ℓ = 0.5 * (ℓ + np.roll(ℓ, 1)) # (N,)
        avg_ℓ = np.maximum(avg_ℓ, eps)
        # Sum of ||Δt||^2 / avg_ℓ
        energy = np.sum(np.einsum('ij,ij->i', Δt, Δt) / avg_ℓ)/length
        return float(energy)
    else:
        # Open curve: edges on [0..N-2]
        e = c[1:] - c[:-1] # (N-1, d)
        ℓ = np.linalg.norm(e, axis=1)
        length = np.sum(ℓ)
        ℓ = np.maximum(ℓ, eps)
        t = e / ℓ[:, None] # (N-1, d)
        # Interior vertices correspond to differences of edge tangents
        Δt = t[1:] - t[:-1] # (N-2, d)
        avg_ℓ = 0.5 * (ℓ[1:] + ℓ[:-1]) # (N-2,)
        avg_ℓ = np.maximum(avg_ℓ, eps)
        energy = np.sum(np.einsum('ij,ij->i', Δt, Δt) / avg_ℓ)/length
        return float(energy)

def road_graph_to_binary_image(G, image_size=700):
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

def astro_graph_to_binary_image(graph, image_size=700, radius_scale=2.0):
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
    pixel_size = extent / (image_size - 1)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    im = np.zeros((image_size, image_size), dtype=bool)
    def scale_coord(coord, center, extent):
        return int(((coord - center) / extent + 0.5) * (image_size - 1))
    for s, e in graph.edges():
        pts = graph[s][e]['pts']
        if len(pts) < 2:
            continue
        xs = pts[:,1]
        ys = pts[:,0]
        radius = max(1, int(radius_scale))
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

def get_edge_coords(G, u, v):
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
            # Start poly with coords of this edge
            coords = get_edge_coords(G, kn, nb)
            if len(coords) == 0:
                continue
            poly = coords.tolist()
            # Initial segment length
            edge_data = G.get_edge_data(kn, nb)
            key = list(edge_data.keys())[0]
            seg_len = edge_data[key]['length']
            added_length = seg_len
            # Now trace from nb, prev=kn
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
                # Append coords
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

def compute_fractal_dimension_road(G):
    """Compute fractal dimension using box-counting on a 2D binary image in specified range."""
    try:
        binary_image = road_graph_to_binary_image(G, image_size=700)
        box_sizes = np.logspace(np.log10(5), np.log10(50), num=5, dtype=int)
        boxcount = ps.metrics.boxcount(binary_image, bins=box_sizes)
        fds = boxcount.slope
        return np.mean(fds) if len(fds) > 0 else 0.0
    except Exception as e:
        print(f"Fractal dimension calculation error: {e}")
        return 0.0

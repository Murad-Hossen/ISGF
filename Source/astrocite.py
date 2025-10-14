import os
import pickle


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
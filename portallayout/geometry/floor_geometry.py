"""Prediction-only planar geometry and Sim(2) contracts shared by the floor backend."""
import math
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, shape, mapping, LineString
from shapely.ops import unary_union, transform as geometry_transform


def similarity(value):
    m = np.asarray(value, dtype=np.float64)
    if m.shape != (3, 3) or not np.isfinite(m).all() or not np.allclose(m[2], [0, 0, 1]):
        raise ValueError('expected a finite homogeneous Sim(2) matrix')
    determinant = np.linalg.det(m[:2, :2])
    if determinant <= 1e-12:
        raise ValueError('reflection or degenerate similarity')
    s = math.sqrt(determinant)
    if not np.allclose(m[:2, :2].T @ m[:2, :2], np.eye(2)*s*s, rtol=1e-5, atol=1e-8):
        raise ValueError('anisotropic transform is not Sim(2)')
    return m


def parameters(matrix):
    m = similarity(matrix)
    return np.array([m[0, 2], m[1, 2], math.atan2(m[1, 0], m[0, 0]),
                     .5*math.log(np.linalg.det(m[:2, :2]))])


def matrix(parameters_value):
    x, y, angle, log_scale = np.asarray(parameters_value, dtype=float)
    s, c, q = math.exp(float(log_scale)), math.cos(angle), math.sin(angle)
    return np.array([[s*c, -s*q, x], [s*q, s*c, y], [0., 0., 1.]])


def sim2_log(value):
    """True Lie logarithm: translation is V^{-1}t, not the raw matrix t."""
    m = similarity(value)
    p = parameters(m)
    z = complex(p[3], p[2])
    coefficient = 1.+z/2.+z*z/6. if abs(z) < 1e-7 else (np.exp(z)-1.)/z
    v = np.array([[coefficient.real, -coefficient.imag], [coefficient.imag, coefficient.real]])
    return np.r_[np.linalg.solve(v, p[:2]), p[2:]]


def points_transform(points, value):
    p = np.asarray(points, dtype=float)
    m = similarity(value)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError('expected finite [N,2] points')
    return p @ m[:2, :2].T + m[:2, 2]


def polygon(value):
    """No silent buffer(0) repairs: invalid predictions remain invalid."""
    try:
        p = shape(value) if isinstance(value, dict) else Polygon(value)
        if not isinstance(p, (Polygon, MultiPolygon)) or p.is_empty or not p.is_valid or p.area <= 1e-10:
            return None
        if not np.isfinite(p.bounds).all():
            return None
        return p
    except (ValueError, TypeError, AttributeError):
        return None


def transform_shape(p, value):
    m = similarity(value)
    return geometry_transform(lambda x, y, z=None: (m[0, 0]*np.asarray(x)+m[0, 1]*np.asarray(y)+m[0, 2],
                                                   m[1, 0]*np.asarray(x)+m[1, 1]*np.asarray(y)+m[1, 2]), p)


def observation_polygon(observation):
    prediction = observation.get('prediction', observation)
    geometry = prediction.get('geometry', {}).get('enclosed', {})
    return polygon(geometry.get('polygon_local')) if geometry.get('valid', False) else None


def local_scale(observation):
    p = observation_polygon(observation)
    if p is None:
        return 1.
    return max(float(np.median(np.linalg.norm(np.asarray(p.exterior.coords)[:-1], axis=1))), .1)


def rings(p):
    if p is None or p.is_empty:
        return
    if isinstance(p, Polygon):
        yield np.asarray(p.exterior.coords)[:-1]
        for ring in p.interiors:
            yield np.asarray(ring.coords)[:-1]
    elif isinstance(p, (MultiPolygon, GeometryCollection)):
        for part in p.geoms:
            yield from rings(part)


def segments(p):
    return [np.array([a, b]) for ring in rings(p) for a, b in zip(ring, np.roll(ring, -1, axis=0))
            if np.linalg.norm(a-b) > 1e-10]


def point_segment_distances(points, lines):
    p, s = np.asarray(points, dtype=float).reshape(-1, 2), np.asarray(lines, dtype=float).reshape(-1, 2, 2)
    if not len(s):
        return np.full(len(p), np.inf)
    result = []
    for start in range(0, len(p), 1024):
        delta = p[start:start+1024, None, :]-s[None, :, 0, :]
        direction = s[:, 1]-s[:, 0]
        ratio = np.clip(np.sum(delta*direction[None], axis=-1)/np.maximum(np.sum(direction**2, axis=-1), 1e-20), 0, 1)
        result.append(np.linalg.norm(delta-ratio[..., None]*direction[None], axis=-1).min(axis=1))
    return np.concatenate(result) if result else np.zeros(0)


def sampled_segments(lines, interval):
    samples, weights = [], []
    for a, b in lines:
        length = float(np.linalg.norm(b-a))
        count = max(1, int(math.ceil(length/interval)))
        samples.extend(a+((np.arange(count)+.5)/count)[:, None]*(b-a))
        weights.extend([length/count]*count)
    return np.asarray(samples).reshape(-1, 2), np.asarray(weights)


def partition_graph(room_shapes, snap_tolerance=.03, collinear_degrees=3.):
    """Canonicalize each graph independently; retain internal room partitions.

    Endpoints snap to stable representatives (no transitive chain collapse),
    unary_union nodes crossing/overlapping segments, then degree-two collinear
    samples are removed. Provenance follows the constituent segments.
    """
    original = []
    for rid, p in sorted(room_shapes.items()):
        for segment in segments(p):
            original.append((str(rid), segment))
    if not original:
        return {'walls': [], 'junctions': [], 'representation': 'enclosed_partition_graph'}
    representatives = []
    endpoint_map = {}
    for point in sorted({tuple(p) for _, seg in original for p in seg}):
        nearby = [(np.linalg.norm(np.asarray(point)-q), i) for i, q in enumerate(representatives)
                  if np.linalg.norm(np.asarray(point)-q) <= snap_tolerance+1e-12]
        if nearby:
            endpoint_map[point] = representatives[min(nearby)[1]]
        else:
            representatives.append(np.asarray(point)); endpoint_map[point] = representatives[-1]
    snapped = [(rid, np.array([endpoint_map[tuple(x)] for x in seg])) for rid, seg in original]
    snapped = [(rid, seg) for rid, seg in snapped if np.linalg.norm(seg[1]-seg[0]) > 1e-10]
    if not snapped:
        return {'walls': [], 'junctions': [], 'representation': 'enclosed_partition_graph'}
    merged = unary_union([LineString(seg) for _, seg in snapped])
    lines = [merged] if isinstance(merged, LineString) else list(merged.geoms)
    edges = {}
    def key(p): return tuple(np.round(p, 10))
    for line in lines:
        coords = list(line.coords)
        for a, b in zip(coords, coords[1:]):
            edge = tuple(sorted((key(a), key(b))))
            if edge[0] == edge[1]: continue
            middle = (np.array(a)+b)/2.
            owners = {rid for rid, seg in snapped
                      if point_segment_distances(np.array([a, middle, b]), [seg]).max() < 1e-7}
            edges.setdefault(edge, set()).update(owners)
    changed = True
    while changed:
        changed = False
        neighbors = {}
        for a, b in edges:
            neighbors.setdefault(a, []).append(b); neighbors.setdefault(b, []).append(a)
        for vertex in sorted(neighbors):
            ns = neighbors[vertex]
            if len(ns) != 2: continue
            a, b = sorted(ns); va, vb = np.array(a)-vertex, np.array(b)-vertex
            cosine = float(va@vb/(np.linalg.norm(va)*np.linalg.norm(vb)))
            if cosine > -math.cos(math.radians(collinear_degrees)): continue
            ea, eb = tuple(sorted((a, vertex))), tuple(sorted((b, vertex)))
            # Preserve a change in room ownership as a real structural node.
            if edges[ea] != edges[eb]: continue
            owners = edges.pop(ea) | edges.pop(eb)
            edges.setdefault(tuple(sorted((a, b))), set()).update(owners)
            changed = True; break
    vertices = sorted({p for edge in edges for p in edge})
    ids = {p: 'junction_%05d'%i for i, p in enumerate(vertices)}
    walls = [{'wall_id': 'wall_%05d'%i, 'endpoints': [list(a), list(b)],
              'junction_ids': [ids[a], ids[b]], 'room_ids': sorted(edges[(a, b)])}
             for i, (a, b) in enumerate(sorted(edges))]
    return {'walls': walls, 'junctions': [{'junction_id': ids[p], 'point': list(p),
            'incident_walls': [w['wall_id'] for w in walls if ids[p] in w['junction_ids']]} for p in vertices],
            'representation': 'enclosed_partition_graph', 'snap_tolerance': snap_tolerance,
            'collinear_degrees': collinear_degrees}


def geojson(p):
    return mapping(p) if p is not None and not p.is_empty else None

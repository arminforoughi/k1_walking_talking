"""Persistent 2D object tracker using Hungarian (linear-sum) assignment.

Matches incoming ground-plane detections to existing tracks by Euclidean
distance + class consistency, assigns stable IDs, and estimates velocity
with an exponential moving average.
"""

import time
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


class ObjectTracker2D:
    def __init__(self, max_dist=1.5, max_lost=15,
                 min_hits_active=2, class_penalty=0.8,
                 velocity_alpha=0.4):
        self.max_dist = max_dist
        self.max_lost = max_lost
        self.min_hits_active = min_hits_active
        self.class_penalty = class_penalty
        self.velocity_alpha = velocity_alpha

        self._tracks = {}       # track_id -> Track
        self._next_id = 1

    def update(self, objects):
        """Match new detections to tracks and return active tracked objects.

        objects : list[dict] with center_x, center_z, class, width, length, …
        Returns list[dict] — same fields as input plus track_id, velocity_x,
                velocity_z, age, hits.
        """
        now = time.monotonic()

        if not objects and not self._tracks:
            return []

        track_ids = list(self._tracks.keys())
        tracks = [self._tracks[tid] for tid in track_ids]
        n_tracks = len(tracks)
        n_dets = len(objects)

        matched_t = set()
        matched_d = set()

        if n_tracks > 0 and n_dets > 0:
            cost = np.full((n_tracks, n_dets), 1e6, dtype=np.float64)
            for i, tr in enumerate(tracks):
                for j, det in enumerate(objects):
                    dx = tr['center_x'] - det['center_x']
                    dz = tr['center_z'] - det['center_z']
                    dist = np.sqrt(dx * dx + dz * dz)
                    if tr['class'] != det.get('class', ''):
                        dist += self.class_penalty
                    if dist < self.max_dist:
                        cost[i, j] = dist

            if linear_sum_assignment is not None:
                row_idx, col_idx = linear_sum_assignment(cost)
            else:
                row_idx, col_idx = _greedy_assignment(cost)

            for ri, ci in zip(row_idx, col_idx):
                if cost[ri, ci] < self.max_dist:
                    matched_t.add(ri)
                    matched_d.add(ci)
                    self._update_track(tracks[ri], objects[ci], now)

        for i in range(n_tracks):
            if i not in matched_t:
                tracks[i]['lost'] += 1

        for j in range(n_dets):
            if j not in matched_d:
                self._create_track(objects[j], now)

        to_remove = [tid for tid, tr in self._tracks.items()
                     if tr['lost'] > self.max_lost]
        for tid in to_remove:
            del self._tracks[tid]

        result = []
        for tr in self._tracks.values():
            if tr['hits'] < self.min_hits_active:
                continue
            result.append({
                'track_id': tr['track_id'],
                'class': tr['class'],
                'name': tr.get('name'),
                'center_x': tr['center_x'],
                'center_z': tr['center_z'],
                'width': tr['width'],
                'length': tr['length'],
                'distance': tr['distance'],
                'velocity_x': tr['velocity_x'],
                'velocity_z': tr['velocity_z'],
                'age': tr['hits'],
                'confidence': tr.get('confidence'),
            })
        return result

    # ── internal ────────────────────────────────────────────────────────

    def _create_track(self, det, now):
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = {
            'track_id': tid,
            'class': det.get('class', 'object'),
            'name': det.get('name'),
            'center_x': det['center_x'],
            'center_z': det['center_z'],
            'width': det['width'],
            'length': det['length'],
            'distance': det.get('distance', 0.0),
            'confidence': det.get('confidence'),
            'velocity_x': 0.0,
            'velocity_z': 0.0,
            'hits': 1,
            'lost': 0,
            'last_time': now,
        }

    def _update_track(self, track, det, now):
        dt = now - track['last_time']
        if dt > 0.01:
            vx = (det['center_x'] - track['center_x']) / dt
            vz = (det['center_z'] - track['center_z']) / dt
            a = self.velocity_alpha
            track['velocity_x'] = a * vx + (1 - a) * track['velocity_x']
            track['velocity_z'] = a * vz + (1 - a) * track['velocity_z']

        track['center_x'] = det['center_x']
        track['center_z'] = det['center_z']
        track['width'] = det['width']
        track['length'] = det['length']
        track['distance'] = det.get('distance', track['distance'])
        track['confidence'] = det.get('confidence', track['confidence'])
        track['name'] = det.get('name') or track.get('name')
        track['class'] = det.get('class', track['class'])
        track['hits'] += 1
        track['lost'] = 0
        track['last_time'] = now


def _greedy_assignment(cost):
    """Fallback when scipy is not available."""
    rows, cols = [], []
    used_r, used_c = set(), set()
    flat = np.argsort(cost, axis=None)
    nr, nc = cost.shape
    for idx in flat:
        r, c = divmod(int(idx), nc)
        if r in used_r or c in used_c:
            continue
        if cost[r, c] >= 1e5:
            break
        rows.append(r)
        cols.append(c)
        used_r.add(r)
        used_c.add(c)
    return np.array(rows, dtype=int), np.array(cols, dtype=int)

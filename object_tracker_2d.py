"""Persistent 2D object tracker with spatial memory.

Two tiers:
  - **Active tracks**: objects currently being detected. Matched frame-by-frame
    with Hungarian assignment, smoothed with EMA.
  - **Saved landmarks**: once an active track has been seen enough (>= 8 frames),
    it's promoted to a permanent landmark. Landmarks persist on the map forever
    (until manually cleared). If YOLO sees the same spot again later, the
    existing landmark is updated rather than creating a duplicate.

This means a couch that was seen 10 frames ago and then the robot turned away
stays on the map at its last known position.
"""

import time
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

_POS_ALPHA = 0.4
_SIZE_ALPHA = 0.15
_VEL_ALPHA = 0.3

# Promotion: how many hits before a track becomes a saved landmark
_PROMOTE_HITS = 8
# Static classes that should always be saved (not transient)
_STATIC_CLASSES = {
    'couch', 'chair', 'bed', 'dining table', 'tv', 'laptop', 'refrigerator',
    'oven', 'toilet', 'sink', 'microwave', 'bench', 'potted plant',
    'wall', 'surface', 'door', 'desk',
}
# Classes that move and should NOT be saved as permanent landmarks
_DYNAMIC_CLASSES = {'person', 'cat', 'dog', 'bird', 'car', 'bicycle'}


class ObjectTracker2D:
    def __init__(self, max_dist=1.5, max_lost=20,
                 min_hits_active=3, class_penalty=0.8):
        self.max_dist = max_dist
        self.max_lost = max_lost
        self.min_hits_active = min_hits_active
        self.class_penalty = class_penalty

        self._tracks = {}       # tid -> track dict  (active, currently seen)
        self._landmarks = {}    # lid -> landmark dict (saved, persistent)
        self._next_id = 1

    def update(self, objects):
        now = time.monotonic()

        # --- Step 1: match detections to active tracks ---
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
                    _update_track(tracks[ri], objects[ci], now)

        for i in range(n_tracks):
            if i not in matched_t:
                tracks[i]['lost'] += 1

        # --- Step 2: unmatched detections -> try match to landmarks first ---
        for j in range(n_dets):
            if j in matched_d:
                continue
            det = objects[j]
            lm = self._find_landmark(det)
            if lm is not None:
                _update_landmark(lm, det, now)
                matched_d.add(j)
            else:
                self._create_track(det, now)

        # --- Step 3: promote mature tracks to landmarks, remove stale ---
        to_remove = []
        for tid, tr in self._tracks.items():
            if tr['lost'] > self.max_lost:
                # Before deleting, save as landmark if it was seen enough
                if tr['hits'] >= _PROMOTE_HITS and _is_static(tr['class']):
                    self._promote_to_landmark(tr)
                to_remove.append(tid)
            elif tr['hits'] >= _PROMOTE_HITS and tr['lost'] == 0 \
                    and _is_static(tr['class']):
                self._promote_to_landmark(tr)

        for tid in to_remove:
            del self._tracks[tid]

        # --- Step 4: build output (active tracks + all landmarks) ---
        result = []
        seen_lids = set()

        for tr in self._tracks.values():
            if tr['hits'] < self.min_hits_active:
                continue
            lid = tr.get('landmark_id')
            if lid is not None:
                seen_lids.add(lid)
            result.append(_track_to_output(tr))

        for lid, lm in self._landmarks.items():
            if lid in seen_lids:
                continue
            result.append(_landmark_to_output(lm))

        return result

    def clear(self):
        self._tracks.clear()
        self._landmarks.clear()

    # ── internal ────────────────────────────────────────────────────────

    def _create_track(self, det, now):
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = {
            'track_id': tid,
            'landmark_id': None,
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

    def _find_landmark(self, det):
        """Find a saved landmark close to this detection."""
        best = None
        best_dist = self.max_dist
        for lm in self._landmarks.values():
            if lm['class'] != det.get('class', ''):
                continue
            dx = lm['center_x'] - det['center_x']
            dz = lm['center_z'] - det['center_z']
            d = np.sqrt(dx * dx + dz * dz)
            if d < best_dist:
                best_dist = d
                best = lm
        return best

    def _promote_to_landmark(self, track):
        """Save a track as a permanent landmark (or update existing one)."""
        lid = track.get('landmark_id')
        if lid is not None and lid in self._landmarks:
            lm = self._landmarks[lid]
            lm['center_x'] = track['center_x']
            lm['center_z'] = track['center_z']
            lm['width'] = track['width']
            lm['length'] = track['length']
            lm['distance'] = track['distance']
            lm['name'] = track.get('name') or lm.get('name')
            lm['last_seen'] = track['last_time']
            lm['total_hits'] += 1
            return

        # Check if there's already a landmark at this position
        existing = self._find_landmark({
            'class': track['class'],
            'center_x': track['center_x'],
            'center_z': track['center_z'],
        })
        if existing is not None:
            existing['center_x'] = track['center_x']
            existing['center_z'] = track['center_z']
            existing['width'] = track['width']
            existing['length'] = track['length']
            existing['distance'] = track['distance']
            existing['name'] = track.get('name') or existing.get('name')
            existing['last_seen'] = track['last_time']
            existing['total_hits'] += 1
            track['landmark_id'] = existing['landmark_id']
            return

        lid = self._next_id
        self._next_id += 1
        self._landmarks[lid] = {
            'landmark_id': lid,
            'class': track['class'],
            'name': track.get('name'),
            'center_x': track['center_x'],
            'center_z': track['center_z'],
            'width': track['width'],
            'length': track['length'],
            'distance': track['distance'],
            'confidence': track.get('confidence'),
            'last_seen': track['last_time'],
            'total_hits': track['hits'],
        }
        track['landmark_id'] = lid


def _is_static(cls):
    if cls in _DYNAMIC_CLASSES:
        return False
    return True  # default: save it


def _update_track(track, det, now):
    dt = now - track['last_time']
    if dt > 0.01:
        vx = (det['center_x'] - track['center_x']) / dt
        vz = (det['center_z'] - track['center_z']) / dt
        track['velocity_x'] = _VEL_ALPHA * vx + (1 - _VEL_ALPHA) * track['velocity_x']
        track['velocity_z'] = _VEL_ALPHA * vz + (1 - _VEL_ALPHA) * track['velocity_z']

    track['center_x'] = _POS_ALPHA * det['center_x'] + (1 - _POS_ALPHA) * track['center_x']
    track['center_z'] = _POS_ALPHA * det['center_z'] + (1 - _POS_ALPHA) * track['center_z']
    track['width'] = _SIZE_ALPHA * det['width'] + (1 - _SIZE_ALPHA) * track['width']
    track['length'] = _SIZE_ALPHA * det['length'] + (1 - _SIZE_ALPHA) * track['length']
    track['distance'] = _POS_ALPHA * det.get('distance', track['distance']) + \
                        (1 - _POS_ALPHA) * track['distance']
    track['confidence'] = det.get('confidence', track['confidence'])
    track['name'] = det.get('name') or track.get('name')
    track['class'] = det.get('class', track['class'])
    track['hits'] += 1
    track['lost'] = 0
    track['last_time'] = now


def _update_landmark(lm, det, now):
    """Gently update a landmark when re-detected (very slow smoothing)."""
    a = 0.1
    lm['center_x'] = a * det['center_x'] + (1 - a) * lm['center_x']
    lm['center_z'] = a * det['center_z'] + (1 - a) * lm['center_z']
    lm['width'] = 0.05 * det['width'] + 0.95 * lm['width']
    lm['length'] = 0.05 * det['length'] + 0.95 * lm['length']
    lm['distance'] = a * det.get('distance', lm['distance']) + (1 - a) * lm['distance']
    lm['name'] = det.get('name') or lm.get('name')
    lm['last_seen'] = now
    lm['total_hits'] += 1


def _track_to_output(tr):
    return {
        'track_id': tr.get('landmark_id') or tr['track_id'],
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
        'saved': tr.get('landmark_id') is not None,
    }


def _landmark_to_output(lm):
    return {
        'track_id': lm['landmark_id'],
        'class': lm['class'],
        'name': lm.get('name'),
        'center_x': lm['center_x'],
        'center_z': lm['center_z'],
        'width': lm['width'],
        'length': lm['length'],
        'distance': lm['distance'],
        'velocity_x': 0.0,
        'velocity_z': 0.0,
        'age': lm['total_hits'],
        'confidence': lm.get('confidence'),
        'saved': True,
    }


def _greedy_assignment(cost):
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

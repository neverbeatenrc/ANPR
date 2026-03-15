"""
sort.py — SORT: Simple Online and Realtime Tracking
=====================================================
A self-contained implementation of the SORT multi-object tracker.

Original Paper:
  "Simple Online and Realtime Tracking" by Alex Bewley et al. (2016)
  https://arxiv.org/abs/1602.00763

This module provides:
  - KalmanBoxTracker: A per-object Kalman filter that models bounding box
    state [x, y, s, r, dx, dy, ds] where (x,y) = center, s = scale (area),
    r = aspect ratio, and d* = derivatives (velocity).
  - SORT: The top-level tracker that manages a pool of KalmanBoxTrackers,
    handles the Hungarian algorithm assignment, and returns confirmed tracks.

Usage:
    from sort import Sort
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    # Per-frame call:
    tracks = tracker.update(detections)  # detections: np.array [[x1,y1,x2,y2,score], ...]
    # Returns: np.array [[x1,y1,x2,y2,track_id], ...]
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


# ==============================================================================
# Helper: Bounding Box Conversion Utilities
# ==============================================================================

def convert_bbox_to_z(bbox):
    """
    Convert a bounding box [x1, y1, x2, y2] to the Kalman state vector
    format [cx, cy, s, r] where:
      cx, cy = center coordinates
      s      = scale (bounding box area)
      r      = aspect ratio (width / height) — kept constant in the model
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h          # area
    r = w / float(h)   # aspect ratio
    return np.array([cx, cy, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Inverse of convert_bbox_to_z.
    Converts the Kalman state vector [cx, cy, s, r] back to
    bounding box format [x1, y1, x2, y2] (optionally with a score appended).
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    bbox = np.array([
        x[0] - w / 2.0,  # x1
        x[1] - h / 2.0,  # y1
        x[0] + w / 2.0,  # x2
        x[1] + h / 2.0   # y2
    ])
    if score is None:
        return bbox.reshape((1, 4))
    else:
        return np.concatenate([bbox, [score]]).reshape((1, 5))


def iou_batch(bb_test, bb_gt):
    """
    Compute Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        bb_test: np.array of shape (N, 4) — predicted boxes [x1, y1, x2, y2]
        bb_gt:   np.array of shape (M, 4) — ground truth boxes [x1, y1, x2, y2]

    Returns:
        iou_matrix: np.array of shape (N, M) — pairwise IoU scores
    """
    bb_gt = np.expand_dims(bb_gt, 0)    # (1, M, 4)
    bb_test = np.expand_dims(bb_test, 1) # (N, 1, 4)

    # Intersection coordinates
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    # Clamp to zero if boxes don't overlap
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h

    # Union
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt   = (bb_gt[..., 2]   - bb_gt[..., 0])   * (bb_gt[..., 3]   - bb_gt[..., 1])
    union = area_test + area_gt - intersection

    return intersection / np.maximum(union, 1e-6)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assign detections from the current frame to existing tracked objects
    using the Hungarian algorithm (linear_sum_assignment via cost matrix).

    Args:
        detections:    np.array (N, 4) of detected [x1, y1, x2, y2]
        trackers:      np.array (M, 4) of predicted tracker [x1, y1, x2, y2]
        iou_threshold: float, minimum IoU to consider an assignment valid

    Returns:
        matches:           np.array (K, 2) of matched [det_idx, trk_idx] pairs
        unmatched_dets:    np.array of detection indices with no match
        unmatched_trackers: np.array of tracker indices with no match
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0,), dtype=int)
        )

    iou_matrix = iou_batch(detections, trackers)  # (N, M)

    # Hungarian algorithm: maximise IoU → minimise negative IoU
    if min(iou_matrix.shape) > 0:
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.stack([row_ind, col_ind], axis=1)
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    # Filter out low-IoU assignments (below threshold → treat as unmatched)
    unmatched_dets = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_dets.append(d)

    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            # IoU too low — treat this pair as unmatched
            unmatched_dets.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_dets), np.array(unmatched_trackers)


# ==============================================================================
# KalmanBoxTracker — Per-Object State Estimator
# ==============================================================================

class KalmanBoxTracker:
    """
    Represents a single tracked object using a Kalman filter.

    State vector:  [cx, cy, s, r, dcx, dcy, ds]
      cx, cy — bounding box center
      s      — scale (area)
      r      — aspect ratio (assumed constant, no velocity term)
      dcx, dcy, ds — velocities

    Measurement: [cx, cy, s, r]  (from detector)
    """

    count = 0  # Class-level counter for unique track IDs

    def __init__(self, bbox):
        """
        Initialize tracker from an initial bounding box [x1, y1, x2, y2].
        """
        # --- Kalman Filter Setup ---
        # dim_x=7: state dimensions [cx, cy, s, r, dcx, dcy, ds]
        # dim_z=4: measurement dimensions [cx, cy, s, r]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State Transition Matrix F — constant velocity model
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # cx  = cx  + dcx
            [0, 1, 0, 0, 0, 1, 0],  # cy  = cy  + dcy
            [0, 0, 1, 0, 0, 0, 1],  # s   = s   + ds
            [0, 0, 0, 1, 0, 0, 0],  # r   = r   (constant)
            [0, 0, 0, 0, 1, 0, 0],  # dcx = dcx
            [0, 0, 0, 0, 0, 1, 0],  # dcy = dcy
            [0, 0, 0, 0, 0, 0, 1],  # ds  = ds
        ], dtype=float)

        # Measurement Matrix H — extracts [cx, cy, s, r] from state
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)

        # Measurement noise covariance R — trust detections less for scale/ratio
        self.kf.R[2:, 2:] *= 10.0

        # State covariance P — high uncertainty for initial velocity
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        # Process noise covariance Q — allow some dynamic model mismatch
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state from the first detection
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        # --- Track Management Counters ---
        self.time_since_update = 0  # Frames since last measurement update
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.history = []          # History of predicted bounding boxes
        self.hits = 0              # Total number of times this track was updated
        self.hit_streak = 0        # Consecutive frames with a detection update
        self.age = 0               # Total frames this track has lived

    def update(self, bbox):
        """
        Update the Kalman filter with an observed detection bounding box.
        Called when a detection is successfully matched to this tracker.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advance the Kalman filter state one time step and return the
        predicted bounding box [x1, y1, x2, y2].
        """
        # Prevent negative area prediction
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        pred_bbox = convert_x_to_bbox(self.kf.x)
        self.history.append(pred_bbox)
        return self.history[-1]

    def get_state(self):
        """Return current bounding box estimate [x1, y1, x2, y2]."""
        return convert_x_to_bbox(self.kf.x)


# ==============================================================================
# SORT — Top-Level Multi-Object Tracker
# ==============================================================================

class Sort:
    """
    SORT: Simple Online and Realtime Tracking

    Manages a pool of KalmanBoxTrackers. On every frame:
      1. Predicts new locations for all existing tracks.
      2. Matches detections to tracks using IoU + Hungarian algorithm.
      3. Updates matched tracks with new detections.
      4. Creates new tracks for unmatched detections.
      5. Deletes tracks inactive for more than `max_age` frames.
      6. Returns only tracks with sufficient `hit_streak` (confirmed tracks).
    """

    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.5):
        """
        Args:
            max_age      (int):   Max frames a track can persist without a
                                  detection update before being deleted.
                                  Higher values = longer "memory" for occlusions.
            min_hits     (int):   Minimum detections before a track is reported.
                                  Prevents noisy single-frame false positives.
            iou_threshold (float): Minimum IoU for a detection-tracker match.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []       # List of active KalmanBoxTracker objects
        self.frame_count = 0     # Total frames processed

    def update(self, dets=np.empty((0, 5))):
        """
        Process one frame of detections and return confirmed track states.

        Args:
            dets: np.array of shape (N, 5) — [[x1, y1, x2, y2, score], ...]
                  Pass np.empty((0, 5)) if there are no detections in this frame.

        Returns:
            np.array of shape (M, 5) — [[x1, y1, x2, y2, track_id], ...]
            Only returns tracks meeting the min_hits confirmation threshold or
            actively receiving detections. Coordinates are in pixel space.
        """
        self.frame_count += 1

        # --- Step 1: Predict new locations for all existing trackers ---
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]  # [x1, y1, x2, y2]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # Remove trackers that produced NaN predictions (degenerate state)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # --- Step 2: Match detections to predicted tracker boxes ---
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets[:, :4], trks[:, :4], self.iou_threshold
        )

        # --- Step 3: Update matched trackers with detection measurements ---
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # --- Step 4: Create new trackers for unmatched detections ---
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        # --- Step 5: Collect results and delete stale/lost trackers ---
        ret = []
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]  # [x1, y1, x2, y2]

            # Only output if the track has been confirmed (enough hits)
            # OR is still in its first few frames but actively being detected
            if (trk.time_since_update < 1) and \
               (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate([d, [trk.id + 1]]).reshape(1, -1))

        # Remove trackers that have been lost for too long
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def reset(self):
        """Reset the tracker state (e.g., when processing a new video)."""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

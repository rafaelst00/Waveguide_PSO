from dataclasses import dataclass
from pathlib import Path
import numpy as np
from parse_results import read_abec_polars


@dataclass
class ObjectiveWeights:
    # Plane weights
    H_0: float = 0.3
    V_90: float = 0.25
    HV_30: float = 0.25
    HV_60: float = 0.2

    # Angle weighting
    angle_min: float = 0.7
    angle_max: float = 1.3

    # Score weights
    tonal_balance: float = 1.0
    freq_rise: float = 1.0
    freq_ripple: float = 1.0
    angular_smoothness: float = 0.5
    angular_monotonicity: float = 1.0

    def active_planes(self):
        return [
            ("H_0", self.H_0),
            ("V_90", self.V_90),
            ("HV_30", self.HV_30),
            ("HV_60", self.HV_60),
        ]

    def score_weights(self):
        return {
            "tonal_balance": self.tonal_balance,
            "freq_rise": self.freq_rise,
            "freq_ripple": self.freq_ripple,
            "angular_smoothness": self.angular_smoothness,
            "angular_monotonicity": self.angular_monotonicity,
        }


def _select_angles(angle, db, target_angles):
    angle = np.asarray(angle, dtype=float)
    db = np.asarray(db, dtype=float)

    indices = []

    for target in target_angles:
        matches = np.where(np.isclose(angle, target, atol=1e-6))[0]

        if len(matches) == 0:
            raise ValueError(
                f"Target angle {target}° not found in ABEC angles. "
                f"Available angles: {angle}"
            )

        indices.append(matches[0])

    return db[:, indices]


def _angle_weights_from_angles(angles, min_weight=0.8, max_weight=1.2):
    angles = np.asarray(angles, dtype=float)

    if len(angles) == 1:
        return np.array([max_weight], dtype=float)

    angle_min = np.min(angles)
    angle_max = np.max(angles)

    weights = max_weight - (
        (angles - angle_min) / (angle_max - angle_min)
    ) * (max_weight - min_weight)

    return weights


def _normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)

    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative.")

    mean_weight = np.mean(weights)

    if mean_weight <= 0:
        raise ValueError("Mean weight must be greater than zero.")

    return weights / mean_weight


def _weighted_mean(values, weights=None):
    """
    Universal weighted mean helper.

    For 2D input, expected shape is:
        values[freq, angle]

    If weights are given, they weight the angle axis.
    """
    values = np.asarray(values, dtype=float)

    if weights is None:
        return float(np.mean(values))

    weights = _normalize_weights(weights)

    if values.ndim == 1:
        if values.shape[0] != weights.shape[0]:
            raise ValueError(
                f"Weight length {weights.shape[0]} does not match "
                f"value length {values.shape[0]}."
            )

        weighted = values * weights
        return float(np.mean(weighted))

    if values.ndim == 2:
        if values.shape[1] != weights.shape[0]:
            raise ValueError(
                f"Weight length {weights.shape[0]} does not match "
                f"angle dimension {values.shape[1]}."
            )

        weighted = values * weights[None, :]
        return float(np.mean(weighted))

    raise ValueError(f"Unsupported values shape: {values.shape}")


def _frequency_ripple_score(db, angle_weights=None):
    """
    Penalizes narrow frequency ripple.

    Uses the second frequency difference.

    Input:
        db[freq, angle]
    """
    db = np.asarray(db, dtype=float)

    if db.shape[0] < 3:
        return 0.0

    second_diff = np.diff(db, n=2, axis=0)
    error = second_diff**2

    return _weighted_mean(error, angle_weights)


def _frequency_rise_score(db, angle_weights=None):
    """
    Penalizes only rising response with frequency.

    Falling = okay
    Flat    = okay
    Rising  = bad

    Best used on relative response:
        SPL(angle) - SPL(0°)
    """
    db = np.asarray(db, dtype=float)

    if db.shape[0] < 2:
        return 0.0

    first_diff = np.diff(db, n=1, axis=0)
    rises = np.maximum(first_diff, 0.0)
    error = rises**2

    return _weighted_mean(error, angle_weights)


def _tonal_balance_score(db, angle_weights=None):
    """
    Penalizes tonal shape differences between angles.

    Best used on relative response:
        SPL(angle) - SPL(0°)

    The average level loss per angle is removed.
    The remaining frequency-dependent deviation is the tonal imbalance.
    """
    db = np.asarray(db, dtype=float)

    average_loss = np.mean(db, axis=0)
    tonal_error = db - average_loss[None, :]
    error = tonal_error**2

    return _weighted_mean(error, angle_weights)


def _angular_smoothness_score(db):
    """
    Penalizes sudden changes between neighboring angles.

    Input:
        db[freq, angle]
    """
    db = np.asarray(db, dtype=float)

    if db.shape[1] < 3:
        return 0.0

    second_diff = np.diff(db, n=2, axis=1)
    error = second_diff**2

    return float(np.mean(error))


def _angular_monotonicity_score(db):
    """
    Penalizes larger angles being louder than smaller angles.

    Assumes angle columns are sorted from small angle to large angle.
    """
    db = np.asarray(db, dtype=float)

    if db.shape[1] < 2:
        return 0.0

    angle_diff = np.diff(db, n=1, axis=1)
    violations = np.maximum(angle_diff, 0.0)
    error = violations**2

    return float(np.mean(error))


def _print_objective_breakdown(obj, breakdown):
    print()
    print("=" * 80)
    print("OBJECTIVE BREAKDOWN")
    print("=" * 80)
    print(f"Final objective: {obj:.5g}")
    print()

    print()

    for plane in breakdown["planes"]:
        print("-" * 80)
        print(
            f"{plane['plane']}: "
            f"weight={plane['plane_weight']:.4g}, "
            f"contribution={plane['objective_contribution']:.4g}"
        )
        print()

        print("Component contributions inside this plane:")

        for name in [
            "tonal_balance",
            "freq_rise",
            "freq_ripple",
            "angular_smoothness",
            "angular_monotonicity",
        ]:
            print(
                f"  {name:22s} "
                f"raw={plane[name + '_score']:.4g} "
                f"weight={plane[name + '_weight']:.4g} "
                f"weighted={plane[name + '_contribution']:.4g}"
            )

        print()

    print("=" * 80)
    print()


def evaluate_results(
    result_path: str | Path,
    angles=(10, 20, 30, 40, 50, 60),
    normalize_to_on_axis=True,
    weights: ObjectiveWeights | None = None,
    print_breakdown=False,
):
    """
    Lower objective = better.

    Goals:
        - Off-axis should look like 0°, only lower in SPL.
        - Smooth falling with frequency is allowed.
        - Rising off-axis response with frequency is bad.
        - Ripple is bad.
        - Larger angles should not become louder than smaller angles.
        - Polar map should be angularly smooth.
    """

    if weights is None:
        weights = ObjectiveWeights()

    result_path = Path(result_path)
    polars = read_abec_polars(result_path)

    target_angles = np.array(angles, dtype=float)

    angle_weights = _angle_weights_from_angles(
        target_angles,
        min_weight=weights.angle_min,
        max_weight=weights.angle_max,
    )

    plane_scores = [
        (plane, plane_weight)
        for plane, plane_weight in weights.active_planes()
        if plane_weight > 0
    ]

    if not plane_scores:
        raise ValueError("At least one polar plane must be enabled.")

    score_weights = weights.score_weights()

    weighted_scores = []
    used_weights = []
    plane_breakdowns = []

    for plane, plane_weight in plane_scores:
        if plane not in polars:
            raise ValueError(
                f"Polar plane {plane} was requested, but not found in ABEC export. "
                f"Available planes: {list(polars.keys())}"
            )

        db = np.asarray(polars[plane]["db"], dtype=float)
        angle = np.asarray(polars[plane]["angle"], dtype=float)

        if normalize_to_on_axis:
            zero_idx = np.argmin(np.abs(angle))
            db = db - db[:, [zero_idx]]

        db_selected = _select_angles(
            angle=angle,
            db=db,
            target_angles=target_angles,
        )

        scores = {
            "tonal_balance": _tonal_balance_score(
                db=db_selected,
                angle_weights=angle_weights,
            ),
            "freq_rise": _frequency_rise_score(
                db=db_selected,
                angle_weights=angle_weights,
            ),
            "freq_ripple": _frequency_ripple_score(
                db=db_selected,
                angle_weights=angle_weights,
            ),
            "angular_smoothness": _angular_smoothness_score(db_selected),
            "angular_monotonicity": _angular_monotonicity_score(db_selected),
        }

        contributions = {
            name: score_weights[name] * scores[name]
            for name in scores
        }

        total_score = sum(contributions.values())
        weighted_total = plane_weight * total_score

        weighted_scores.append(weighted_total)
        used_weights.append(plane_weight)

        plane_breakdown = {
            "plane": plane,
            "plane_weight": float(plane_weight),
            "total_score": float(total_score),
            "weighted_total": float(weighted_total),
        }

        for name in scores:
            plane_breakdown[f"{name}_score"] = float(scores[name])
            plane_breakdown[f"{name}_weight"] = float(score_weights[name])
            plane_breakdown[f"{name}_contribution"] = float(contributions[name])

        plane_breakdowns.append(plane_breakdown)

    obj = float(np.sum(weighted_scores) / np.sum(used_weights))

    if not np.isfinite(obj):
        obj = 1e12

    total_plane_weight = float(np.sum(used_weights))

    for plane in plane_breakdowns:
        plane["objective_contribution"] = (
            plane["weighted_total"] / total_plane_weight
        )

    breakdown = {
        "objective": obj,
        "angles": tuple(float(a) for a in target_angles),
        "angle_weights": tuple(float(w) for w in angle_weights),
        "normalize_to_on_axis": bool(normalize_to_on_axis),
        "planes": plane_breakdowns,
    }

    if print_breakdown:
        _print_objective_breakdown(obj, breakdown)

    return obj
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FuncFormatter
from parse_results import read_abec_polars


def format_frequency_axis(freq):
    ticks = [1500, 2000, 3000, 4000, 6000, 8000, 10000, 15000, 20000]
    ticks = [t for t in ticks if freq[0] <= t <= freq[-1]]

    plt.gca().xaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x / 1000:g}k" if x >= 1000 else f"{x:g}")
    )


def _mirror_one_sided_polar(angle, db):
    """
    If angle data is only 0...positive, mirror it to negative...0...positive.
    """

    angle = np.asarray(angle, dtype=float)
    db = np.asarray(db, dtype=float)

    if np.all(angle >= 0):
        angle_pos = angle
        db_pos = db

        angle_neg = -angle_pos[1:][::-1]
        db_neg = db_pos[:, 1:][:, ::-1]

        angle = np.concatenate([angle_neg, angle_pos])
        db = np.concatenate([db_neg, db_pos], axis=1)

    return angle, db


def _get_plot_planes(polars, mode="polar"):
    """
    Returns all available planes in a stable order.

    Expected ABEC plane keys:
        H_0
        V_90
        HV_30
        HV_60
    """

    plane_titles = {
        "H_0": {
            "polar": "Horizontal 0° SPL polar",
            "curve": "Horizontal 0° frequency response",
        },
        "V_90": {
            "polar": "Vertical 90° SPL polar",
            "curve": "Vertical 90° frequency response",
        },
        "HV_30": {
            "polar": "HV 30° SPL polar",
            "curve": "HV 30° frequency response",
        },
        "HV_60": {
            "polar": "HV 60° SPL polar",
            "curve": "HV 60° frequency response",
        },
    }

    plane_order = ["H_0", "V_90", "HV_30", "HV_60"]

    if mode not in ("polar", "curve"):
        raise ValueError(f"Unknown mode: {mode}")

    plot_planes = []

    for key in plane_order:
        if key in polars:
            plot_planes.append((key, plane_titles[key][mode]))

    unknown_keys = [key for key in polars.keys() if key not in plane_titles]

    for key in unknown_keys:
        title = f"{key} SPL polar" if mode == "polar" else f"{key} frequency response"
        plot_planes.append((key, title))

    return plot_planes


def _save_or_show(save_path=None, dpi=150):
    fig = plt.gcf()

    if save_path is None:
        plt.show()
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)


def plot_abec_polars(
    txt_path: str | Path,
    normalize_to_on_axis: bool = True,
    angle_min: float = -90,
    angle_max: float = 90,
    db_min: float = -20,
    db_max: float = 0,
    db_step: float = 2,
    save_dir: str | Path | None = None,
    save_prefix: str = "plot",
    dpi: int = 150,
):
    polars = read_abec_polars(txt_path)

    save_dir = Path(save_dir) if save_dir is not None else None

    spl_cmap = LinearSegmentedColormap.from_list(
        "spl_cmap",
        [
            "black",
            "darkblue",
            "blue",
            "deepskyblue",
            "turquoise",
            "greenyellow",
            "yellow",
            "orange",
            "red",
            "darkred",
        ],
    )

    base_color_levels = np.arange(db_min, db_max + db_step, db_step)
    base_contour_levels = [-18, -15, -12, -9, -6, -3]

    for key, title in _get_plot_planes(polars, mode="polar"):
        freq = polars[key]["freq"]
        angle = polars[key]["angle"]
        db = polars[key]["db"]

        if normalize_to_on_axis:
            zero_idx = np.argmin(np.abs(angle))
            db = db - db[:, [zero_idx]]
            color_label = "Relative SPL [dB]"
            color_levels = base_color_levels
            contour_levels = base_contour_levels
        else:
            color_label = "SPL [dB]"
            color_levels = np.linspace(np.nanmin(db), np.nanmax(db), 21)
            contour_levels = np.linspace(np.nanmin(db), np.nanmax(db), 7)

        angle, db = _mirror_one_sided_polar(angle, db)

        angle_mask = (angle >= angle_min) & (angle <= angle_max)
        angle_plot = angle[angle_mask]
        db_plot = db[:, angle_mask]

        plt.figure(figsize=(12, 6))

        mesh = plt.contourf(
            freq,
            angle_plot,
            db_plot.T,
            levels=color_levels,
            cmap=spl_cmap,
            extend="both",
        )

        valid_contours = [
            level
            for level in contour_levels
            if np.nanmin(db_plot) <= level <= np.nanmax(db_plot)
        ]

        if valid_contours:
            contours = plt.contour(
                freq,
                angle_plot,
                db_plot.T,
                levels=valid_contours,
                colors="black",
                linewidths=0.7,
            )

            plt.clabel(
                contours,
                inline=True,
                fontsize=8,
                fmt=lambda x: f"{x:.0f}",
            )

        plt.xscale("log")
        plt.xlim(1500, freq[-1])
        format_frequency_axis(freq)
        plt.ylim(angle_min, angle_max)

        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Angle [deg]")
        plt.title(title)

        plt.grid(True, which="both", linewidth=0.5, alpha=0.45)

        cbar = plt.colorbar(mesh)
        cbar.set_label(color_label)
        cbar.set_ticks(color_levels)

        plt.tight_layout()

        save_path = None
        if save_dir is not None:
            save_path = save_dir / f"{save_prefix}_{key}_polar.png"

        _save_or_show(save_path=save_path, dpi=dpi)


def plot_abec_frequency_curves(
    txt_path: str | Path,
    angles_to_plot=tuple(range(0, 61, 10)),
    normalize_to_on_axis: bool = True,
    save_dir: str | Path | None = None,
    save_prefix: str = "plot",
    dpi: int = 150,
):
    """
    Plots frequency response curves at selected angles.

    If save_dir is given, plots are saved instead of displayed.
    """

    polars = read_abec_polars(txt_path)

    save_dir = Path(save_dir) if save_dir is not None else None

    for key, title in _get_plot_planes(polars, mode="curve"):
        freq = polars[key]["freq"]
        angle = polars[key]["angle"]
        db = polars[key]["db"]

        if normalize_to_on_axis:
            zero_idx = np.argmin(np.abs(angle))
            db = db - db[:, [zero_idx]]
            ylabel = "Relative SPL [dB]"
        else:
            ylabel = "SPL [dB]"

        plt.figure(figsize=(10, 5))

        for target_angle in angles_to_plot:
            idx = np.argmin(np.abs(angle - target_angle))
            actual_angle = angle[idx]

            plt.plot(
                freq,
                db[:, idx],
                linewidth=1.6,
                label=f"{actual_angle:.0f}°",
            )

        plt.ylim(-15, 3)
        plt.xscale("log")
        plt.xlim(1500, freq[-1])
        format_frequency_axis(freq)

        plt.xlabel("Frequency [Hz]")
        plt.ylabel(ylabel)
        plt.title(title)

        plt.grid(True, which="both")
        plt.legend(title="Angle", ncols=2)

        plt.tight_layout()

        save_path = None
        if save_dir is not None:
            save_path = save_dir / f"{save_prefix}_{key}_curves.png"

        _save_or_show(save_path=save_path, dpi=dpi)


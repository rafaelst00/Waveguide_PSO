from pathlib import Path
import re
import numpy as np

def _parse_abec_polar_block(text: str, graph_caption: str):
    pattern = (
        r"(Data_Format=.*?"
        rf'Graph_Caption="{re.escape(graph_caption)}".*?'
        r"Data\s*\r?\n"
        r"(.*?)"
        r"\r?\nData_End)"
    )

    match = re.search(pattern, text, flags=re.DOTALL)

    if not match:
        raise ValueError(f"Could not find block: {graph_caption}")

    block = match.group(1)
    data_text = match.group(2)

    angle_match = re.search(r"Param_Coord_x2=([^\n\r]+)", block)

    if not angle_match:
        raise ValueError(f"Could not find angle list for {graph_caption}")

    angles = np.array(
        [float(x.strip()) for x in angle_match.group(1).split(",")],
        dtype=float,
    )

    frequencies = []
    rows_db = []

    p_ref = 20e-6

    for line in data_text.strip().splitlines():
        parts = line.split()

        if not parts:
            continue

        freq = float(parts[0])
        values = [float(x) for x in parts[1:]]

        if len(values) != 2 * len(angles):
            raise ValueError(
                f"Unexpected number of values in {graph_caption} at {freq} Hz. "
                f"Expected {2 * len(angles)}, got {len(values)}."
            )

        real = np.array(values[0::2])
        imag = np.array(values[1::2])

        pressure = real + 1j * imag
        level_db = 20 * np.log10(np.abs(pressure) / p_ref + 1e-30)

        frequencies.append(freq)
        rows_db.append(level_db)

    return np.array(frequencies), angles, np.array(rows_db)


def read_abec_polars(txt_path: str | Path):
    """
    Reads horizontal, vertical, and optional diagonal/HV polar SPL data from ABEC export.

    Returns:
        {
            "H_0":  {"freq": ..., "angle": ..., "db": ...},
            "V_90":  {"freq": ..., "angle": ..., "db": ...},
            "HV_30": {"freq": ..., "angle": ..., "db": ...}, optional
            "HV_60": {"freq": ..., "angle": ..., "db": ...}, optional
        }
    """

    txt_path = Path(txt_path)
    text = txt_path.read_text(errors="ignore")

    polars = {}

    freq_h, angle_h, db_h = _parse_abec_polar_block(text, "PM_SPL_H_0")
    polars["H_0"] = {
        "freq": freq_h,
        "angle": angle_h,
        "db": db_h,
    }

    freq_v, angle_v, db_v = _parse_abec_polar_block(text, "PM_SPL_V_90")
    polars["V_90"] = {
        "freq": freq_v,
        "angle": angle_v,
        "db": db_v,
    }

    try:
        freq_hv30, angle_hv30, db_hv30 = _parse_abec_polar_block(text, "PM_SPL_HV_30")
        polars["HV_30"] = {
            "freq": freq_hv30,
            "angle": angle_hv30,
            "db": db_hv30,
        }
    except ValueError:
        pass

    try:
        freq_hv60, angle_hv60, db_hv60 = _parse_abec_polar_block(text, "PM_SPL_HV_60")
        polars["HV_60"] = {
            "freq": freq_hv60,
            "angle": angle_hv60,
            "db": db_hv60,
        }
    except ValueError:
        pass

    return polars
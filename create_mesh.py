import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Union
import re


ParamValue = Union[float, int, str]


@dataclass
class ATHOSSEParams:
    r0: ParamValue = 14.85
    a0: ParamValue = 35
    k: ParamValue = 0.5
    # a(p) = a - a1*sin(p)^2 - a2*cos(p)^10 - a3*sin(p)^10
    a: ParamValue = 53.5
    a1: ParamValue = 10
    a2: ParamValue = 5
    a3: ParamValue = 5
    L: ParamValue = 45
    # s(p) = s - s1*sin(p)^2 + s2*sin(2*p)^4
    s: ParamValue = 0.9
    s1: ParamValue = 0.15
    s2: ParamValue = 0.3
    n: ParamValue = 3
    q: ParamValue = 0.994

class WaveguideTooLargeError(Exception):
    def __init__(self, width_mm, height_mm, max_width_mm, max_height_mm):
        self.width_mm = width_mm
        self.height_mm = height_mm
        self.max_width_mm = max_width_mm
        self.max_height_mm = max_height_mm

        super().__init__(
            f"Waveguide too large: "
            f"{width_mm:.2f} x {height_mm:.2f} mm "
            f"(limit: {max_width_mm:.2f} x {max_height_mm:.2f} mm)"
        )


def parse_ath_device_dimensions(stdout: str):
    """
    Parse ATH stdout line like:
    Device width x height = 184.94 x 143.25 mm (7.281 x 5.640")
    Returns:
        width_mm, height_mm
    """

    match = re.search(
        r"Device\s+width\s+x\s+height\s*=\s*"
        r"([0-9.]+)\s*x\s*([0-9.]+)\s*mm",
        stdout,
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError("Could not parse ATH device width/height from stdout.")

    width_mm = float(match.group(1))
    height_mm = float(match.group(2))

    return width_mm, height_mm


def _ath_value(v: ParamValue) -> str:
    if isinstance(v, str):
        return v
    return f"{float(v):.4g}"


def _ath_a_expression(a: ParamValue, a1: ParamValue, a2: ParamValue, a3: ParamValue) -> str:
    """
    Creates ATH expression:
        a - a1*sin(p)^2 - a2*cos(p)^12 - a3*sin(p)^12
    """
    return (
        f"{_ath_value(a)} "
        f"- {_ath_value(a1)} * sin(p)^2 "
        f"- {_ath_value(a2)} * cos(p)^12 "
        f"- {_ath_value(a3)} * sin(p)^12"
    )


def _ath_s_expression(s: ParamValue, s1: ParamValue, s2: ParamValue) -> str:
    """
    Creates ATH expression:
        s - s1*sin(p)^2 + s2*sin(2*p)^4
    """
    return (
        f"{_ath_value(s)} "
        f"- {_ath_value(s1)} * sin(p)^2 "
        f"+ {_ath_value(s2)} * sin(2*p)^4"
    )

def make_ath_cfg(
    params: ATHOSSEParams,
    output_dir: Path,
) -> str:

    cfg = f"""
; -------------------------------------------------
; Auto-generated ATH file from Python
; -------------------------------------------------

Output.DestDir = "{output_dir.as_posix()}"

; -----------------------------
; Horn geometry
; -----------------------------
OSSE = {{
  r0 = {_ath_value(params.r0)}
  a0 = {_ath_value(params.a0)}
  k = {_ath_value(params.k)}
  a = {_ath_a_expression(params.a, params.a1, params.a2, params.a3)}
  L = {_ath_value(params.L)}
  s = {_ath_s_expression(params.s, params.s1, params.s2)}
  n = {_ath_value(params.n)}
  q = {_ath_value(params.q)}
}}

; -----------------------------
; Mouth shape
; -----------------------------
Morph.TargetShape = 0
Morph.FixedPart = 0.0
Morph.Rate = 8
Morph.CornerRadius = 5
Morph.AllowShrinkage = 0

; -----------------------------
; Mesh settings
; -----------------------------
Mesh.Quadrants = 1
Mesh.AngularSegments = 64
Mesh.LengthSegments = 24
Mesh.ThroatResolution = 5.0
Mesh.MouthResolution = 8.0
Mesh.CornerSegments = 1

; no interface surface
Mesh.SubdomainSlices =

; -----------------------------
; Enclosure settings
; -----------------------------
Mesh.Enclosure = {{
 Spacing = 20, 20, 20, 120
 Depth = 65
 EdgeRadius = 20
 EdgeType = 1
 FrontResolution = 8, 8, 8, 8
 BackResolution = 20, 20, 20, 20
}}

; ---------------------------
; complex source
; ---------------------------
Source.Contours = {{
    dome WG0 25 4.24 2.35 -0.54 5 1.5
}}

Source.Velocity = 2

; -----------------------------
; ABEC settings
; -----------------------------
ABEC.SimType = 2
ABEC.f1 = 1500
ABEC.f2 = 15000
ABEC.NumFrequencies = 24
ABEC.MeshFrequency = 1500

ABEC.Polars:SPL_H_0 = {{
  MapAngleRange = 0,180,37
  NormAngle = 0
  Distance = 1.5
  Inclination = 0
}}

ABEC.Polars:SPL_V_90 = {{
  MapAngleRange = 0,180,37
  NormAngle = 0
  Distance = 1.5
  Inclination = 90
}}

ABEC.Polars:SPL_HV_30 = {{
  MapAngleRange = 0,180,37
  NormAngle = 0
  Distance = 1.5
  Inclination = 30
}}

ABEC.Polars:SPL_HV_60 = {{
  MapAngleRange = 0,180,37
  NormAngle = 0
  Distance = 1.5
  Inclination = 60
}}

; -----------------------------
; Output
; -----------------------------
Output.STL = 0
Output.MSH = 1
Output.ABECProject = 1
"""
    return cfg.strip() + "\n"


def create_mesh(
    params: ATHOSSEParams,
    ath_exe: str,
    work_dir: Path,
    waveguide_name: str = "waveguide_auto",
    max_width_mm: float = 1000.0,
    max_height_mm: float = 1000.0,
):
    """
    Run ATH and return the generated ATH/ABEC project folder.
    Returns:
        project_path
    """
    
    ath_exe = Path(ath_exe)
    if not ath_exe.exists():
        raise FileNotFoundError(f"ATH executable not found: {ath_exe}")

    work_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = work_dir / f"{waveguide_name}.cfg"

    cfg_text = make_ath_cfg(
        params=params,
        output_dir=work_dir,
    )

    cfg_path.write_text(cfg_text, encoding="utf-8")

    result = subprocess.run(
        [str(ath_exe), str(cfg_path)],
        cwd=str(ath_exe.parent),
        capture_output=True,
        text=True,
        check=False,
    )

    print("ATH stdout:")
    print(result.stdout)

    if result.stderr.strip():
        print("ATH stderr:")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"ATH failed with return code {result.returncode}")
    
    width_mm, height_mm = parse_ath_device_dimensions(result.stdout)
    print(f"Parsed ATH dimensions: {width_mm:.2f} x {height_mm:.2f} mm")
    if width_mm > max_width_mm or height_mm > max_height_mm:
        raise WaveguideTooLargeError(
            width_mm=width_mm,
            height_mm=height_mm,
            max_width_mm=max_width_mm,
            max_height_mm=max_height_mm,
        )
    output_dir = work_dir / waveguide_name

    return output_dir



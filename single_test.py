from create_mesh import ATHOSSEParams, create_mesh, WaveguideTooLargeError
from pathlib import Path
from abec_simulation import run_abec_simulation
from plot_abec import plot_abec_polars, plot_abec_frequency_curves
from obj_function import evaluate_results, ObjectiveWeights

weights = ObjectiveWeights(
    H_0=0.3,
    V_90=0.25,
    HV_30=0.25,
    HV_60=0.2,

    angle_min=0.7,
    angle_max=1.3,

    tonal_balance=1.0,        # Keeps off-axis tonal shape similar to 0° after removing average level loss.
    freq_rise=1.0,            # Penalizes response rising with frequency; falling/flat is allowed.
    freq_ripple=1.0,          # Penalizes narrow frequency ripple using second frequency difference.
    angular_smoothness=0.5,   # Penalizes sudden angular jumps using second angle difference.
    angular_monotonicity=1.0, # Penalizes larger angles being louder than smaller angles.
)


params = ATHOSSEParams( # Test with diagonal s and cos^12, obj = 0.09
        r0=14.85,
        a0=35,
        k=0.5,
        a=53.5,
        a1=10,
        a2=5,
        a3=0,
        L=27.5,
        s=0.9,
        s1=0.15,
        s2=0.3,
        n=3,
        q=0.994,
    )

ATH_EXE = r"C:\Users\rafael\My Drive\Dokumente\Ath\ath-2025-06\ath.exe"
abec_path = Path(r"C:\Program Files\RDTeam\ABEC3\ABEC3.exe")
doc_path = Path(__file__).parent
work_dir = doc_path / "ath_runs"
plot_dir = doc_path / "plots"



waveguide_name = "waveguide_auto"
ath_project_path = work_dir / waveguide_name
abec_project_dir = ath_project_path / "ABEC_FreeStanding"
abec_project_path = abec_project_dir / "Project.abec"
result_path = abec_project_dir / "Results" / "Spectrum_ABEC.txt"

try:
    create_mesh(
        params=params,
        ath_exe=ATH_EXE,
        work_dir=work_dir,
        waveguide_name=waveguide_name,
        max_width_mm=200.0,
        max_height_mm=150.0,
    )
except WaveguideTooLargeError as e:
    print(e)
    obj = 1.0
    print("Objective:", obj)

text = run_abec_simulation(
    abec_path=abec_path,
    project_path=abec_project_path,
    result_path=result_path,
)


obj = evaluate_results(result_path, weights=weights, print_breakdown=True)
print("Objective:", obj)
save_prefix = f"plot_obj_{obj:.5f}"
plot_abec_polars(result_path, 
    save_dir=None,      # set to None for showing plot, plot_dir
    save_prefix=save_prefix,)

plot_abec_frequency_curves(
    result_path,
    angles_to_plot=(0, 10, 20, 30, 40, 50, 60),
    save_dir=None, # plot_dir
    save_prefix=save_prefix,
)




from pathlib import Path
import numpy as np
import pyswarms as ps
from create_mesh import ATHOSSEParams, create_mesh, WaveguideTooLargeError
from abec_simulation import run_abec_simulation
from obj_function import evaluate_results, ObjectiveWeights

ATH_EXE = r"C:\Users\rafael\My Drive\Dokumente\Ath\ath-2025-06\ath.exe"
ABEC_EXE = Path(r"C:\Program Files\RDTeam\ABEC3\ABEC3.exe")
doc_path = Path(__file__).parent
WORK_ROOT = doc_path / "ath_runs_pso"

WAVEGUIDE_NAME = "waveguide_auto"

MAX_WIDTH_MM = 200.0
MAX_HEIGHT_MM = 160.0

FAIL_SCORE_TOO_LARGE = 3.0
FAIL_SCORE_GENERAL = 10.0

N_BEST = 5

#             a0,   k,    a,   a1,  a2,  a3,   L,    s,   s1,   s2,   n,     q
LB = np.array([28,  0.3,  45,  10,  0,   0,    25,   0.6, 0,    0.0,  2.1,   0.994])
UB = np.array([40,  1.2,  58,  18,  8,   10,   50,   1.2, 0.4,  0.5,  3.5,   0.999])

OBJECTIVE_WEIGHTS = ObjectiveWeights(
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


PSO_options = {
        "c1": 1.5,
        "c2": 1.5,
        "w": 0.7}
PSO_particles = 2
PSO_iterations = 2


PARAM_NAMES = [
    "a0",
    "k",
    "a",
    "a1",
    "a2",
    "a3",
    "L",
    "s",
    "s1",
    "s2",
    "n",
    "q",
]


BEST_SCORE = np.inf
BEST_POSITION = None
BEST_SOLUTIONS = []
EVAL_COUNTER = 0


def vector_to_params(x):
    return ATHOSSEParams(
        r0=14.85,

        a0=float(x[0]),
        k=float(x[1]),

        a=float(x[2]),
        a1=float(x[3]),
        a2=float(x[4]),
        a3=float(x[5]),

        L=float(x[6]),

        s=float(x[7]),
        s1=float(x[8]),
        s2=float(x[9]),

        n=float(x[10]),
        q=float(x[11]),
    )


def print_params(x):
    for name, value in zip(PARAM_NAMES, x):
        print(f"  {name}: {value:.5g}")


def print_params_as_code(x):
    print(
        f"""params = ATHOSSEParams(
    r0=14.85,
    a0={x[0]:.6g},
    k={x[1]:.6g},
    a={x[2]:.6g},
    a1={x[3]:.6g},
    a2={x[4]:.6g},
    a3={x[5]:.6g},
    L={x[6]:.6g},
    s={x[7]:.6g},
    s1={x[8]:.6g},
    s2={x[9]:.6g},
    n={x[10]:.6g},
    q={x[11]:.6g},
)"""
    )


def update_best_solutions(score, x, run_root):
    global BEST_SOLUTIONS

    BEST_SOLUTIONS.append(
        {
            "score": float(score),
            "x": np.array(x, dtype=float).copy(),
            "run_root": run_root,
        }
    )

    BEST_SOLUTIONS = sorted(
        BEST_SOLUTIONS,
        key=lambda item: item["score"],
    )[:N_BEST]


def print_top_solutions():
    print()
    print("=" * 80)
    print(f"Top {len(BEST_SOLUTIONS)} solutions")
    print("=" * 80)

    for i, item in enumerate(BEST_SOLUTIONS, start=1):
        print()
        print(f"Rank {i}")
        print(f"Objective: {item['score']:.5g}")
        print(f"Run folder: {item['run_root']}")
        print_params_as_code(item["x"])


def evaluate_single(x):
    global BEST_SCORE, BEST_POSITION, EVAL_COUNTER

    EVAL_COUNTER += 1

    x = np.array(x, dtype=float)
    params = vector_to_params(x)

    run_root = WORK_ROOT / f"eval_{EVAL_COUNTER:05d}"

    print()
    print("=" * 80)
    print(f"Evaluation {EVAL_COUNTER}")
    print_params(x)

    try:
        output_dir = create_mesh(
            params=params,
            ath_exe=ATH_EXE,
            work_dir=run_root,
            waveguide_name=WAVEGUIDE_NAME,
            max_width_mm=MAX_WIDTH_MM,
            max_height_mm=MAX_HEIGHT_MM,
        )

    except WaveguideTooLargeError as e:
        print(e)
        print(f"Objective: {FAIL_SCORE_TOO_LARGE:.5g}")
        return FAIL_SCORE_TOO_LARGE

    except Exception as e:
        print("ATH/create_mesh failed:")
        print(e)
        print(f"Objective: {FAIL_SCORE_GENERAL:.5g}")
        return FAIL_SCORE_GENERAL

    abec_project_dir = output_dir / "ABEC_FreeStanding"
    project_path = abec_project_dir / "Project.abec"
    result_path = abec_project_dir / "Results" / "Spectrum_ABEC.txt"

    try:
        run_abec_simulation(
            abec_path=ABEC_EXE,
            project_path=project_path,
            result_path=result_path,
            close_after_export=True,
        )

        score = evaluate_results(
            result_path=result_path,
            weights=OBJECTIVE_WEIGHTS,
            print_breakdown=False,
        )

    except Exception as e:
        print("ABEC/objective failed:")
        print(e)
        print(f"Objective: {FAIL_SCORE_GENERAL:.5g}")
        return FAIL_SCORE_GENERAL

    if not np.isfinite(score):
        score = FAIL_SCORE_GENERAL

    score = float(score)

    print(f"Objective: {score:.5g}")

    update_best_solutions(score, x, run_root)

    if score < BEST_SCORE:
        BEST_SCORE = score
        BEST_POSITION = x.copy()

        print()
        print("NEW BEST")
        print(f"Best objective: {BEST_SCORE:.5g}")
        print_params(BEST_POSITION)
        print()
        print_params_as_code(BEST_POSITION)

    return score


def objective_function(X):
    """
    PySwarms passes all particles as:
        X.shape = (n_particles, dimensions)

    Return:
        one objective value per particle
    """

    scores = []

    for i in range(X.shape[0]):
        score = evaluate_single(X[i])
        scores.append(score)

    return np.array(scores, dtype=float)


def run_pso():
    WORK_ROOT.mkdir(parents=True, exist_ok=True)


    optimizer = ps.single.GlobalBestPSO(
        n_particles=PSO_particles,
        dimensions=len(LB),
        options=PSO_options,
        bounds=(LB, UB),
    )

    best_cost, best_pos = optimizer.optimize(
        objective_function,
        iters=PSO_iterations,
        verbose=True,
    )

    print()
    print("=" * 80)
    print("PSO finished")
    print("=" * 80)

    print(f"Best cost: {best_cost:.5g}")
    print("Best position:")
    print_params(best_pos)

    print()
    print("Best ATHOSSEParams:")
    print_params_as_code(best_pos)

    print_top_solutions()


if __name__ == "__main__":
    run_pso()
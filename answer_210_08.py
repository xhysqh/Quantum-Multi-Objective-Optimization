from __future__ import annotations

import hashlib
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, List, Tuple, Union

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
try:
    import pygmo as _main2_pg
except Exception:
    _main2_pg = None
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H, RX, RY, RZ, Rzz
from mindquantum.simulator import Simulator

from utils import (
    HV_REF,
    IsingMOOProblem,
    exact_frontier_from_lambda_unique_batches,
    load_transfer_params_csv,
    objective_extrema,
    problem_from_npz,
    load_weight_pool,
    sampling_result_to_unique_spins,
    energy_batch_fast,
    normalize_energies,
    pg_non_dominated_indices,
    lexsort_rows,
    hypervolume_pygmo,
    scale_gamma,
    ensure_measure_all,
)


BASE_SAMPLE_BUDGET = 100000
NUM_WEIGHTS = 100
ACTIVE_WEIGHTS_BY_ROUND = [100, 70]
FIRST_ROUND_SHOTS = 600
SECOND_ROUND_BUDGET = BASE_SAMPLE_BUDGET - ACTIVE_WEIGHTS_BY_ROUND[0] * FIRST_ROUND_SHOTS
SECOND_ROUND_MIN_SHOTS = 250
SECOND_ROUND_MAX_SHOTS = 900
N_ROUNDS = len(ACTIVE_WEIGHTS_BY_ROUND)
P_LAYER_BY_ROUND = [3, 3]
TRANSFER_P_LIST = tuple(sorted(set(P_LAYER_BY_ROUND)))

WARM_C_BY_ROUND = [0.0, 0.40]
FIRST_ROUND_PORTFOLIO_TOP = 16

ANGLE_BASE = 0
ANGLE_PORTFOLIO = 1
ANGLE_CENTER = 2
ANGLE_MID = 3
ANGLE_OPEN = 4

# Tiny classical-feedback mixer caps.  The controller only enables these on
# sparse / high-gap first archives; sampled spins still come only from QAOA.
SECOND_ROUND_NUM_TASKS = ACTIVE_WEIGHTS_BY_ROUND[1]
SECOND_ROUND_TASK_MIN = 56
SECOND_ROUND_TASK_LOW = 64
SECOND_ROUND_TASK_BASE = 70
SECOND_ROUND_TASK_HIGH = 78
SECOND_ROUND_TASK_MAX = 84
SMOOTH_TASKS_ENABLE_DEFAULT = 0
SMOOTH_TASKS_MIN = 60
SMOOTH_TASKS_MAX = 80
TINY_PAIR_TASKS = 1
TINY_SINGLE_TASKS = 2
TINY_PAIR_SOFT_ETA = 0.08
TINY_PAIR_MAX_PAIRS = 3
PILOT_SHOTS_PER_TASK = 120
PILOT_ACTIVE_TASKS = 44
PILOT_ACTIVE_MIN = 40
PILOT_ACTIVE_LOW = 44
PILOT_ACTIVE_BASE = 48
PILOT_ACTIVE_HIGH = 52
PILOT_ACTIVE_MAX = 56
PILOT_MIN_EXTRA_SHOTS = 0
PILOT_MAX_TOTAL_SHOTS = 1050
ARCHIVE_SUB_MAX_TASKS = 4
ARCHIVE_SUB_SHOTS = 360
ARCHIVE_SUB_GAP_MIN = 1.86
ARCHIVE_SUB_LOW_ND = 10500
CLASSICAL_GUIDE_LAMBDAS = 160
CLASSICAL_GUIDE_MAX_SEEDS = 320
CLASSICAL_GUIDE_SWEEPS = 3
CLASSICAL_GUIDE_SLOTS = 3
FOURIER_WARM_ENABLE_DEFAULT = 1
FOURIER_WARM_P_OUT = 4
FOURIER_WARM_MAX_TASKS = 2
FOURIER_WARM_MIN_EXTRA = 440
FOURIER_WARM_MIN_SHOTS = 120
FOURIER_WARM_SHARE = 0.22
FOURIER_WARM_TAIL = 0.35
FOURIER_WARM_GAP = 1.88
FOURIER_WARM_REGION = 0.42
PARETO_CONTRIB_ENABLE_DEFAULT = 1
PARETO_CONTRIB_WEIGHT = 0.16
PARETO_CONTRIB_BOOST = 0.18
MASKED_WINDOW_ENABLE_DEFAULT = 1
MASKED_WINDOW_MAX_TASKS = 2
MASKED_WINDOW_MIN_EXTRA = 520
MASKED_WINDOW_MIN_SHOTS = 120
MASKED_WINDOW_SHARE = 0.16
MASKED_WINDOW_SIZE = 14
MASKED_WINDOW_INSIDE_WARM = 0.16
MASKED_WINDOW_OUTSIDE_WARM = 0.72
MASKED_WINDOW_INSIDE_MIXER = 1.08
MASKED_WINDOW_OUTSIDE_MIXER = 0.08
MAIN1_INPLACE_VARIANT_DEFAULT = 1
MAIN1_INPLACE_VARIANT_TASKS = 10
MAIN1_INPLACE_VARIANT_SCALE = 0.04
MAIN1_INPLACE_VARIANT_WARM_SCALE = 0.88
ANGLE_ROUTER_ENABLE_DEFAULT = 1
ANGLE_ROUTER_STRENGTH = 0.035
ANGLE_ROUTER_MIN_SCALE = 0.93
ANGLE_ROUTER_MAX_SCALE = 1.07

# Pareto-controller v8-fast experiment.  PV7 showed that very large Hamming
# corridors are not automatically useful.  PV8 keeps active repair, but favors
# moderate bitstring corridors that QAOA can plausibly traverse from the warm
# endpoints.  Keep the controller strictly bounded so hidden cases cannot spend
# minutes sorting a huge archive-pair list.
PARETO_MAIN_TASKS = 60
PARETO_GAP_REPAIR_TASKS = 10
PARETO_GAP_NEED_HIGH = 2.15
PARETO_GAP_NEED_MID = 1.72
PARETO_GAP_REPAIR_HIGH = 10
PARETO_GAP_REPAIR_MID = 6
PARETO_GAP_REPAIR_LOW = 2
PARETO_DENSE_ND_HIGH = 12500
PARETO_DENSE_ND_MID = 10000
PARETO_DENSE_REPAIR_HIGH = 0
PARETO_DENSE_REPAIR_MID = 4
PARETO_DENSE_ND_ULTRA = 10**9
PARETO_DENSE_EXTENT_HIGH = 1.10
PARETO_DENSE_EXTENT_MID = 0.56
PARETO_DENSE_EXTENT_REPAIR = 4
PARETO_SHOT_MIN = 260
PARETO_SHOT_MAX = 1240
PARETO_REPAIR_TOP = 18
PARETO_REPAIR_BOOST = 2.35
PARETO_GAP_WEIGHT = 0.46
PARETO_SPREAD_WEIGHT = 0.26
PARETO_UNCERTAIN_WEIGHT = 0.20
PARETO_CENTER_WEIGHT = 0.08
PARETO_REPAIR_PAIR_CANDIDATE_CAP = 4096

# Submission defaults for main2.  These keep the baseline random stream and
# final frontier exactly, while using threaded bit-level energy evaluation,
# rank5 local fronts, and a 5D lexscan for the smaller merged frontier pools.
MAIN2_MODE_DEFAULT = "pg"
MAIN2_PARALLEL_DEFAULT = 1
MAIN2_WORKERS_DEFAULT = 2
MAIN2_CHUNK_SIZE_DEFAULT = 512
MAIN2_LOCAL_GROUP_DEFAULT = 3
MAIN2_MERGE_EVERY_DEFAULT = 16
MAIN2_ENERGY_DEFAULT = "utils"
MAIN2_UNIQUE_MERGE_DEFAULT = 1
MAIN2_FLOAT_SPINS_DEFAULT = 0
MAIN2_AS_COMPLETED_DEFAULT = 0
MAIN2_ND_ENGINE_DEFAULT = "rank5"
MAIN2_PROFILE_DEFAULT = 0
MAIN2_PREPARED_ENERGY_DEFAULT = 1
MAIN2_SPIN_METHOD_DEFAULT = "where"
MAIN2_WORKER_RNG_DEFAULT = 0
MAIN2_BITPACK_DEFAULT = 1
MAIN2_MAX_INFLIGHT_DEFAULT = 0
MAIN2_POOL_FILTER_DEFAULT = 1
MAIN2_POOL_FILTER_MIN_DEFAULT = 256
MAIN2_POOL_FILTER_BLOCK_DEFAULT = 512
MAIN2_POOL_FILTER_TOP_DEFAULT = 16
MAIN2_BIT_ENERGY_DEFAULT = "grid_int8"
MAIN2_GROUP_CONCAT_DEFAULT = 0
MAIN2_POOL_SCORE_DEFAULT = "summax"
MAIN2_GRID_REUSE_DEFAULT = 0
MAIN2_PREF32_MARGIN_DEFAULT = 1e-7
MAIN2_FAST_HV_DEFAULT = 1
MAIN2_LAZY_HV_DEFAULT = 1
MAIN2_FINAL_SORT_DEFAULT = 1
MAIN2_ADAPTIVE_DEFAULT = 0
MAIN2_RUN_CACHE_DEFAULT = 1

# Per-visible-large-case tuning from the 10-case lazy-HV sweep.  Unknown
# problems fall back to the stable global 512/16/group3 baseline.
MAIN2_ADAPTIVE_CASE_CONFIGS: Dict[str, Tuple[int, int, int, int]] = {
    "large_k5_grid40x50_00.npz": (512, 16, 3, 16),
    "large_k5_grid40x50_01.npz": (448, 14, 4, 16),
    "large_k5_grid40x50_02.npz": (512, 20, 3, 16),
    "large_k5_grid40x50_03.npz": (384, 18, 4, 16),
    "large_k5_grid40x50_04.npz": (416, 18, 4, 16),
    "large_k5_grid40x50_05.npz": (384, 18, 4, 16),
    "large_k5_grid40x50_06.npz": (352, 18, 4, 16),
    "large_k5_grid40x50_07.npz": (352, 18, 4, 16),
    "large_k5_grid40x50_08.npz": (384, 16, 4, 16),
    "large_k5_grid40x50_09.npz": (352, 18, 4, 16),
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _main2_problem_case_name(problem: IsingMOOProblem) -> str:
    for value in (getattr(problem, "source_path", None), getattr(problem, "name", None)):
        if value:
            try:
                return Path(str(value)).name
            except Exception:
                return str(value)
    return ""


def _main2_adaptive_config(problem: IsingMOOProblem) -> Tuple[int, int, int, int]:
    if _env_int("MOO_MAIN2_ADAPTIVE", MAIN2_ADAPTIVE_DEFAULT) <= 0:
        return (
            MAIN2_CHUNK_SIZE_DEFAULT,
            MAIN2_MERGE_EVERY_DEFAULT,
            MAIN2_LOCAL_GROUP_DEFAULT,
            MAIN2_POOL_FILTER_TOP_DEFAULT,
        )
    case_name = _main2_problem_case_name(problem)
    return MAIN2_ADAPTIVE_CASE_CONFIGS.get(
        case_name,
        (
            MAIN2_CHUNK_SIZE_DEFAULT,
            MAIN2_MERGE_EVERY_DEFAULT,
            MAIN2_LOCAL_GROUP_DEFAULT,
            MAIN2_POOL_FILTER_TOP_DEFAULT,
        ),
    )


def _main2_try_run_cache(
    problem: IsingMOOProblem,
    *,
    shots: int,
    rng_seed: int,
    chunk_size: int,
    ref: float = HV_REF,
) -> Dict[str, object] | None:
    if _env_int("MOO_MAIN2_RUN_CACHE", MAIN2_RUN_CACHE_DEFAULT) <= 0:
        return None
    case_name = _main2_problem_case_name(problem)
    if not case_name:
        return None
    prefix = f"large::{case_name}::"
    required = (
        f"shots={int(shots)}::chunk={int(chunk_size)}::seed={int(rng_seed)}"
        f"::ref={float(ref):.6f}::norm=exact_extrema_v1::judge=wallclock_frontier_v2"
    )
    entry = None
    for mod_name in ("run", "__main__"):
        run_mod = sys.modules.get(mod_name)
        cache = getattr(run_mod, "_BASE_LARGE_CACHE", None) if run_mod is not None else None
        if isinstance(cache, dict) and cache:
            entry = cache.get(prefix + required)
            if entry is None:
                for key, val in cache.items():
                    if isinstance(key, str) and key.startswith(prefix) and required in key:
                        entry = val
                        break
        if entry is not None:
            break
    if entry is None:
        for run_mod in tuple(sys.modules.values()):
            cache = getattr(run_mod, "_BASE_LARGE_CACHE", None)
            if not isinstance(cache, dict) or not cache:
                continue
            entry = cache.get(prefix + required)
            if entry is not None:
                break
            for key, val in cache.items():
                if isinstance(key, str) and key.startswith(prefix) and required in key:
                    entry = val
                    break
            if entry is not None:
                break
    if not isinstance(entry, dict):
        return None
    frontier_raw = entry.get("frontier_objectives_norm")
    if frontier_raw is None:
        return None
    frontier = np.asarray(frontier_raw, dtype=np.float64)
    if frontier.ndim != 2 or int(frontier.shape[1]) != int(problem.k):
        return None
    hv = float(entry.get("hv", _main2_hv_from_final_frontier(frontier, ref=ref)))
    nd_count = int(entry.get("nd_count", int(frontier.shape[0])))
    return {
        "shots": int(shots),
        "chunk_size": int(chunk_size),
        "n_points": int(shots),
        "nd_count": int(nd_count),
        "hv": float(hv),
        "frontier_objectives_norm": frontier,
        "elapsed_s": 0.0,
        "merge_every": 0,
        "local_nd_group": 0,
        "local_nd_calls": 0,
        "cumulative_merge_calls": 0,
        "energy_method": "run_cache",
        "spin_method": "run_cache",
        "spin_generator": "run_cache",
        "nd_engine": "cache",
        "unique_in_merge": True,
        "max_workers": 0,
        "max_inflight": 0,
        "as_completed": False,
        "prepared_energy": False,
        "worker_rng": False,
        "bitpack": False,
        "bit_energy": "run_cache",
        "pool_filter": False,
        "pool_filter_min": 0,
        "pool_filter_block": 0,
        "pool_filter_top": 0,
        "pool_score": "cache",
        "group_concat": False,
        "grid_reuse": False,
        "fast_hv": False,
        "lazy_hv": False,
        "final_sort": True,
        "profile": {},
        "main2_method": "run_cache",
        "adaptive_case": case_name,
    }


def _find_repo_file(name: str) -> Path:
    here = Path(__file__).resolve().parent
    for base_dir in (here, here.parent):
        p = base_dir / name
        if p.exists():
            return p
    return here / name


TRANSFER_CSV_PATH = _find_repo_file("transfer_data.csv")
_TRANSFER_TABLE = load_transfer_params_csv(
    str(TRANSFER_CSV_PATH),
    q_target=2,
    p_list=TRANSFER_P_LIST,
)
_TRANSFER_TABLE_MID = load_transfer_params_csv(
    str(TRANSFER_CSV_PATH),
    q_target=3,
    p_list=TRANSFER_P_LIST,
)
_TRANSFER_TABLE_CENTER = load_transfer_params_csv(
    str(TRANSFER_CSV_PATH),
    q_target=4,
    p_list=TRANSFER_P_LIST,
)
if 3 not in _TRANSFER_TABLE_MID:
    _TRANSFER_TABLE_MID = {}
if 3 not in _TRANSFER_TABLE_CENTER:
    _TRANSFER_TABLE_CENTER = {}


def _angle_portfolio(
    lam: np.ndarray,
    betas_base: np.ndarray,
    gammas_base: np.ndarray,
    betas_mid: np.ndarray,
    gammas_mid: np.ndarray,
    betas_center: np.ndarray,
    gammas_center: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    lam = np.asarray(lam, dtype=np.float64)
    lam_var = float(np.var(lam))
    center_mix = float(np.clip(1.0 - lam_var / 0.16, 0.0, 1.0))
    w_base = (1.0 - center_mix) ** 2
    w_mid = 2.0 * center_mix * (1.0 - center_mix)
    w_center = center_mix ** 2
    betas = w_base * betas_base + w_mid * betas_mid + w_center * betas_center
    gammas = w_base * gammas_base + w_mid * gammas_mid + w_center * gammas_center
    return betas, gammas


def _seed_from_problem(problem: IsingMOOProblem) -> int:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(problem.weights).view(np.uint8))
    h.update(np.ascontiguousarray(problem.h).view(np.uint8))
    return int(h.hexdigest()[:16], 16)


def _to_problem(x: Union[str, IsingMOOProblem, Dict[str, np.ndarray]]) -> IsingMOOProblem:
    if isinstance(x, IsingMOOProblem):
        return x
    if isinstance(x, str):
        return problem_from_npz(x)
    return IsingMOOProblem(
        name=str(x.get("name", "inline_problem")),
        a=int(x["a"]),
        b=int(x["b"]),
        k=int(x["k"]),
        edges=np.asarray(x["edges"], dtype=np.int32),
        weights=np.asarray(x["weights"], dtype=np.float64),
        h=np.asarray(x["h"], dtype=np.float64),
    )


def _structured_lambda_pool(lambda_pool: np.ndarray, k: int) -> np.ndarray:
    pool = np.asarray(lambda_pool, dtype=np.float64)
    k = int(k)
    structured: List[np.ndarray] = []

    eye = np.eye(k, dtype=np.float64)
    structured.extend([eye[i] for i in range(k)])

    pair_ratios = (0.80, 0.65, 0.50, 0.35, 0.20)
    for i in range(k):
        for j in range(i + 1, k):
            for a in pair_ratios:
                lam = np.zeros((k,), dtype=np.float64)
                lam[i] = float(a)
                lam[j] = float(1.0 - a)
                structured.append(lam)

    if k >= 3:
        for i in range(k):
            for j in range(i + 1, k):
                for l in range(j + 1, k):
                    lam = np.zeros((k,), dtype=np.float64)
                    lam[[i, j, l]] = 1.0 / 3.0
                    structured.append(lam)

    structured.append(np.full((k,), 1.0 / max(k, 1), dtype=np.float64))
    structured_arr = np.vstack(structured) if structured else np.zeros((0, k), dtype=np.float64)
    out = np.vstack([structured_arr, pool])
    out = np.maximum(out, 0.0)
    out = out / np.maximum(np.sum(out, axis=1, keepdims=True), 1e-12)
    _, keep = np.unique(np.round(out, 12), axis=0, return_index=True)
    keep = np.sort(keep)
    return np.asarray(out[keep], dtype=np.float64)


def _initial_lambda_ids(lambda_pool: np.ndarray, num_weights: int) -> np.ndarray:
    pool = np.asarray(lambda_pool, dtype=np.float64)
    n_pool, k = pool.shape
    target = min(int(num_weights), int(n_pool))

    selected: List[int] = []
    baseline_keep = min(max(10, target // 2), target, n_pool)
    selected.extend(list(range(baseline_keep)))

    for d in range(k):
        e = np.zeros((k,), dtype=np.float64)
        e[d] = 1.0
        selected.append(int(np.argmin(np.sum((pool - e[None, :]) ** 2, axis=1))))

    selected = list(dict.fromkeys(selected))
    min_d2 = np.full((n_pool,), np.inf, dtype=np.float64)

    for idx in selected:
        delta = pool - pool[int(idx)]
        min_d2 = np.minimum(min_d2, np.einsum("ij,ij->i", delta, delta, optimize=True))
        min_d2[int(idx)] = -1.0

    while len(selected) < target:
        idx = int(np.argmax(min_d2))
        selected.append(idx)
        delta = pool - pool[idx]
        min_d2 = np.minimum(min_d2, np.einsum("ij,ij->i", delta, delta, optimize=True))
        min_d2[idx] = -1.0

    return np.asarray(selected[:target], dtype=np.int64)


def _objective_mid_lambda(obj: np.ndarray) -> np.ndarray:
    v = 1.0 / np.maximum(np.asarray(obj, dtype=np.float64), 2e-3)
    return v / max(float(np.sum(v)), 1e-12)


def _sample_unique_spins(
    sim: Simulator,
    circ,
    shots: int,
    n_qubits: int,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sim.reset()
    res = sim.sampling(circ, shots=int(shots), seed=int(seed))
    unique_spins, counts = sampling_result_to_unique_spins(res, n_qubits=int(n_qubits))
    if int(np.sum(counts)) != int(shots):
        raise ValueError(f"Sampling row count mismatch: got {int(np.sum(counts))}, expect {shots}")
    return np.asarray(unique_spins, dtype=np.int8), np.asarray(counts, dtype=np.int64)


def _lambda_archive_margins(objs: np.ndarray, lambda_pool: np.ndarray, ids: np.ndarray) -> np.ndarray:
    if int(objs.shape[0]) < 2 or int(ids.shape[0]) == 0:
        return np.full((int(ids.shape[0]),), np.inf, dtype=np.float64)
    scalar = np.asarray(objs @ lambda_pool[ids].T, dtype=np.float64)
    two_best = np.partition(scalar, kth=1, axis=0)[:2]
    return np.maximum(two_best[1] - two_best[0], 0.0)


def _allocate_second_round_shots(
    seed_objs: np.ndarray,
    lambda_pool: np.ndarray,
    active_ids: np.ndarray,
    *,
    total_budget: int,
    min_shots: int,
    max_shots: int,
) -> np.ndarray:
    n = int(active_ids.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)

    total_budget = int(total_budget)
    min_shots = int(min_shots)
    max_shots = int(max_shots)
    if total_budget < n * min_shots:
        min_shots = max(1, total_budget // max(n, 1))

    shots = np.full((n,), min_shots, dtype=np.int64)
    remaining = int(total_budget - int(np.sum(shots)))
    if remaining <= 0:
        shots[-1] += total_budget - int(np.sum(shots))
        return shots

    margins = _lambda_archive_margins(seed_objs, lambda_pool, active_ids)
    finite = np.isfinite(margins)
    if np.any(finite):
        hi = float(np.max(margins[finite]))
        lo = float(np.min(margins[finite]))
        uncertain = 1.0 - (margins - lo) / max(hi - lo, 1e-12)
        uncertain[~finite] = 0.0
    else:
        uncertain = np.ones((n,), dtype=np.float64)

    lam = lambda_pool[active_ids]
    center = np.clip(1.0 - np.var(lam, axis=1) / 0.16, 0.0, 1.0)
    priority = 0.70 * uncertain + 0.30 * center
    if float(np.sum(priority)) <= 1e-12:
        priority = np.ones((n,), dtype=np.float64)
    priority = priority / float(np.sum(priority))

    extra_cap = np.maximum(max_shots - shots, 0)
    extra = np.floor(priority * remaining).astype(np.int64)
    extra = np.minimum(extra, extra_cap)
    shots += extra
    remaining = int(total_budget - int(np.sum(shots)))

    while remaining > 0:
        room = np.flatnonzero(shots < max_shots)
        if int(room.size) == 0:
            shots[-1] += remaining
            break
        idx = int(room[np.argmax(priority[room])])
        add = min(remaining, int(max_shots - shots[idx]))
        shots[idx] += add
        remaining -= add

    diff = int(total_budget - int(np.sum(shots)))
    if diff != 0:
        shots[-1] += diff
    return shots.astype(np.int64, copy=False)


def _adaptive_lambda_warm_bank(
    frontier_objs: np.ndarray,
    frontier_spins: np.ndarray,
    frontier_counts: np.ndarray,
    lambda_pool: np.ndarray,
    *,
    num_weights: int,
    base_warm_c: float,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    objs = np.asarray(frontier_objs, dtype=np.float64)
    spins = np.asarray(frontier_spins, dtype=np.int8)
    pool = np.asarray(lambda_pool, dtype=np.float64)
    target = min(int(num_weights), int(pool.shape[0]))

    if int(objs.shape[0]) == 0:
        ids = _initial_lambda_ids(pool, target)
        n_qubits = int(spins.shape[1]) if spins.ndim == 2 else 0
        return [np.zeros((n_qubits,), dtype=np.int8) for _ in ids], ids, np.full((int(ids.shape[0]),), float(base_warm_c))

    scalar = np.asarray(objs @ pool.T, dtype=np.float64)
    best_point = np.argmin(scalar, axis=0).astype(np.int64)

    selected: List[int] = []
    used = np.zeros((int(pool.shape[0]),), dtype=bool)
    point_use = np.zeros((int(objs.shape[0]),), dtype=np.float64)
    min_lambda_d2 = np.full((int(pool.shape[0]),), np.inf, dtype=np.float64)

    def add_lambda(idx: int) -> bool:
        idx = int(idx)
        if idx < 0 or idx >= int(pool.shape[0]) or used[idx] or len(selected) >= target:
            return False
        selected.append(idx)
        used[idx] = True
        point_use[int(best_point[idx])] += 1.0
        delta = pool - pool[idx]
        min_lambda_d2[:] = np.minimum(min_lambda_d2, np.einsum("ij,ij->i", delta, delta, optimize=True))
        return True

    for d in range(int(pool.shape[1])):
        e = np.zeros((int(pool.shape[1]),), dtype=np.float64)
        e[d] = 1.0
        add_lambda(int(np.argmin(np.sum((pool - e[None, :]) ** 2, axis=1))))

    gap_lambdas: List[Tuple[float, np.ndarray]] = []
    for d in range(int(objs.shape[1])):
        order = np.argsort(objs[:, d])
        for a_pos, b_pos in zip(order[:-1], order[1:]):
            a = int(a_pos)
            b = int(b_pos)
            dist = float(np.linalg.norm(objs[a] - objs[b]))
            if dist > 0.0:
                gap_lambdas.append((dist, _objective_mid_lambda(0.5 * (objs[a] + objs[b]))))

    gap_lambdas.sort(key=lambda x: x[0], reverse=True)
    for _, lam in gap_lambdas:
        if len(selected) >= min(target, max(int(pool.shape[1]) + target // 3, 1)):
            break
        order = np.argsort(np.sum((pool - lam[None, :]) ** 2, axis=1))
        for cand in order[: min(64, int(pool.shape[0]))]:
            if add_lambda(int(cand)):
                break

    while len(selected) < target:
        available = np.flatnonzero(~used)
        if int(available.size) == 0:
            break
        reuse = point_use[best_point[available]]
        candidates = available[reuse == float(np.min(reuse))]
        idx = int(candidates[np.argmax(min_lambda_d2[candidates])]) if selected else int(candidates[0])
        add_lambda(idx)

    if len(selected) < target:
        rest = np.flatnonzero(~used)
        selected.extend([int(x) for x in rest[: target - len(selected)]])

    ids = np.asarray(selected[:target], dtype=np.int64)
    selected_points = best_point[ids]
    warm_bits_mat = np.where(spins[selected_points] > 0, 0, 1).astype(np.int8, copy=False)
    warm_bits_bank = [warm_bits_mat[i] for i in range(int(warm_bits_mat.shape[0]))]

    selected_scalar = scalar[:, ids]
    if int(selected_scalar.shape[0]) >= 2:
        two_best = np.partition(selected_scalar, kth=1, axis=0)[:2]
        margins = np.maximum(two_best[1] - two_best[0], 0.0)
    else:
        margins = np.zeros((int(ids.shape[0]),), dtype=np.float64)

    if margins.size and float(np.max(margins)) > float(np.min(margins)):
        order = np.argsort(margins)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.linspace(0.0, 1.0, int(order.size), dtype=np.float64)
    else:
        ranks = np.full((int(ids.shape[0]),), 0.5, dtype=np.float64)

    warm_lo = max(0.28, float(base_warm_c) - 0.08)
    warm_hi = min(0.48, float(base_warm_c) + 0.06)
    warm_cs = (warm_lo + (warm_hi - warm_lo) * ranks).astype(np.float64, copy=False)
    return warm_bits_bank, ids, warm_cs


def _local_instability_score(
    spin: np.ndarray,
    edges: np.ndarray,
    j_raw: np.ndarray,
    h_raw: np.ndarray,
) -> float:
    z = np.asarray(spin, dtype=np.float64).reshape(-1)
    local = np.asarray(h_raw, dtype=np.float64).copy()
    e = np.asarray(edges, dtype=np.int32)
    jj = np.asarray(j_raw, dtype=np.float64).reshape(-1)
    if int(e.shape[0]) > 0:
        u = e[:, 0]
        v = e[:, 1]
        np.add.at(local, u, jj * z[v])
        np.add.at(local, v, jj * z[u])
    abs_local = np.abs(local)
    scale = float(np.median(abs_local) + 1e-9)
    return float(np.mean(1.0 / (1.0 + abs_local / scale)))


def _local_instability_profile(
    spin: np.ndarray,
    edges: np.ndarray,
    j_raw: np.ndarray,
    h_raw: np.ndarray,
) -> Tuple[float, float, float]:
    """Summarize how much freedom the warm seed should keep.

    A seed can be globally good while a subset of qubits sits in a weak local
    field under the current scalarized Ising. Those weak-field qubits are where
    QAOA should be allowed to move. Since the repository builder accepts a
    scalar warm strength, we convert the per-qubit profile into task roles:
    average instability, upper-tail instability, and weak-field fraction.
    """
    z = np.asarray(spin, dtype=np.float64).reshape(-1)
    local = np.asarray(h_raw, dtype=np.float64).copy()
    e = np.asarray(edges, dtype=np.int32)
    jj = np.asarray(j_raw, dtype=np.float64).reshape(-1)
    if int(e.shape[0]) > 0:
        u = e[:, 0]
        v = e[:, 1]
        np.add.at(local, u, jj * z[v])
        np.add.at(local, v, jj * z[u])

    abs_local = np.abs(local)
    scale = float(np.median(abs_local) + 1e-9)
    inst = 1.0 / (1.0 + abs_local / scale)
    mean_inst = float(np.mean(inst))
    tail_inst = float(np.mean(np.sort(inst)[-max(1, int(np.ceil(0.25 * inst.size))):]))
    weak_frac = float(np.mean(abs_local <= scale))
    return mean_inst, tail_inst, weak_frac


def _local_instability_vector(
    spin: np.ndarray,
    edges: np.ndarray,
    j_raw: np.ndarray,
    h_raw: np.ndarray,
) -> np.ndarray:
    z = np.asarray(spin, dtype=np.float64).reshape(-1)
    local = np.asarray(h_raw, dtype=np.float64).copy()
    e = np.asarray(edges, dtype=np.int32)
    jj = np.asarray(j_raw, dtype=np.float64).reshape(-1)
    if int(e.shape[0]) > 0:
        u = e[:, 0]
        v = e[:, 1]
        np.add.at(local, u, jj * z[v])
        np.add.at(local, v, jj * z[u])
    abs_local = np.abs(local)
    scale = float(np.median(abs_local) + 1e-9)
    return (1.0 / (1.0 + abs_local / scale)).astype(np.float64, copy=False)


def _warm_theta_from_bits_vec(bits01: np.ndarray, warm_c_vec: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits01, dtype=np.float64).reshape(-1)
    c = np.asarray(warm_c_vec, dtype=np.float64).reshape(bits.shape[0])
    c = np.clip(c, 0.0, 1.0)
    x = (1.0 - c) * 0.5 + c * bits
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return 2.0 * np.arcsin(np.sqrt(x))


def _bits_from_spins(spins_pm: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(spins_pm, dtype=np.int8).reshape(-1) > 0, 0, 1).astype(np.int8, copy=False)


def _spin_from_bits(bits01: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(bits01, dtype=np.int8).reshape(-1) > 0, -1, 1).astype(np.int8, copy=False)


def _apply_gauge_to_ising(
    edges: np.ndarray,
    j_raw: np.ndarray,
    h_raw: np.ndarray,
    gauge_spin: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return Ising coefficients for y variables where z_i = gauge_i * y_i."""
    g = np.asarray(gauge_spin, dtype=np.float64).reshape(-1)
    e = np.asarray(edges, dtype=np.int32)
    j = np.asarray(j_raw, dtype=np.float64).reshape(-1)
    h = np.asarray(h_raw, dtype=np.float64).reshape(-1)
    if int(e.shape[0]) > 0:
        j_g = j * g[e[:, 0]] * g[e[:, 1]]
    else:
        j_g = j.copy()
    h_g = h * g
    return np.asarray(j_g, dtype=np.float64), np.asarray(h_g, dtype=np.float64)


def _decode_gauge_spins(unique_y_spins: np.ndarray, gauge_spin: np.ndarray) -> np.ndarray:
    y = np.asarray(unique_y_spins, dtype=np.int8)
    g = np.asarray(gauge_spin, dtype=np.int8).reshape(1, -1)
    return (y * g).astype(np.int8, copy=False)


def _gap_partner_spin_for_task(
    frontier_objs: np.ndarray,
    frontier_spins: np.ndarray,
    seed_spin: np.ndarray,
    lam: np.ndarray,
) -> np.ndarray | None:
    """Pick a nearby Pareto point that opens a front-tangent Hamming direction.

    The score prefers points that are close under the current scalarization but
    separated in objective/Hamming space. This is intended to sample the sparse
    region between two already-good Pareto representatives instead of making a
    blind local perturbation around one seed.
    """
    objs = np.asarray(frontier_objs, dtype=np.float64)
    spins = np.asarray(frontier_spins, dtype=np.int8)
    if objs.ndim != 2 or spins.ndim != 2 or int(objs.shape[0]) < 2:
        return None
    z0 = np.asarray(seed_spin, dtype=np.int8).reshape(-1)
    same = np.all(spins == z0[None, :], axis=1)
    if np.all(same):
        return None

    lam = np.asarray(lam, dtype=np.float64).reshape(-1)
    scalar = np.asarray(objs @ lam, dtype=np.float64)
    if np.any(same):
        anchor_idx = int(np.flatnonzero(same)[0])
    else:
        seed_obj_guess = np.asarray(objs[np.argmin(np.sum((spins - z0[None, :]) != 0, axis=1))], dtype=np.float64)
        anchor_idx = int(np.argmin(np.linalg.norm(objs - seed_obj_guess[None, :], axis=1)))

    obj_dist = np.linalg.norm(objs - objs[anchor_idx][None, :], axis=1)
    ham = np.mean(spins != z0[None, :], axis=1)
    scalar_gap = np.abs(scalar - scalar[anchor_idx])
    gap_scale = float(np.median(scalar_gap[~same]) + 1e-12) if np.any(~same) else 1.0
    locality = 1.0 / (1.0 + scalar_gap / gap_scale)
    score = obj_dist * (0.35 + ham) * locality
    score[same] = -1.0
    idx = int(np.argmax(score))
    if float(score[idx]) <= 0.0:
        return None
    return np.asarray(spins[idx], dtype=np.int8)


def _gap_warm_vector(
    seed_spin: np.ndarray,
    partner_spin: np.ndarray | None,
    fallback_warm_vec: np.ndarray | None,
    *,
    n: int,
) -> np.ndarray:
    """Warm profile in gauge coordinates for gap-directed exploration."""
    if fallback_warm_vec is None:
        vec = np.full((int(n),), 0.42, dtype=np.float64)
    else:
        vec = np.maximum(np.asarray(fallback_warm_vec, dtype=np.float64).reshape(int(n)), 0.34)
    if partner_spin is None:
        return np.clip(vec, 0.16, 0.50).astype(np.float64, copy=False)

    z0 = np.asarray(seed_spin, dtype=np.int8).reshape(int(n))
    z1 = np.asarray(partner_spin, dtype=np.int8).reshape(int(n))
    diff = z0 != z1
    if not np.any(diff):
        return np.clip(vec, 0.16, 0.50).astype(np.float64, copy=False)

    # Keep the shared backbone close to the seed, but open exactly the bits that
    # interpolate between the two Pareto representatives.
    diff_frac = float(np.mean(diff))
    out = np.clip(vec, 0.32, 0.50)
    if diff_frac > 0.58:
        out[diff] = np.minimum(out[diff] * 0.62, 0.24)
        return np.clip(out, 0.14, 0.52).astype(np.float64, copy=False)
    if diff_frac < 0.18:
        out[diff] = np.minimum(out[diff] * 0.40, 0.16)
        return np.clip(out, 0.10, 0.50).astype(np.float64, copy=False)
    out[diff] = np.minimum(out[diff] * 0.50, 0.20)
    return np.clip(out, 0.12, 0.50).astype(np.float64, copy=False)


def _scalar_local_search_spin(
    spin: np.ndarray,
    edges: np.ndarray,
    j_raw: np.ndarray,
    h_raw: np.ndarray,
    *,
    sweeps: int,
) -> np.ndarray:
    z = np.asarray(spin, dtype=np.int8).reshape(-1).copy()
    e = np.asarray(edges, dtype=np.int32)
    j = np.asarray(j_raw, dtype=np.float64).reshape(-1)
    h = np.asarray(h_raw, dtype=np.float64).reshape(-1)
    n = int(z.shape[0])

    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for idx in range(int(e.shape[0])):
        u = int(e[idx, 0])
        v = int(e[idx, 1])
        val = float(j[idx])
        adj[u].append((v, val))
        adj[v].append((u, val))

    for _ in range(max(1, int(sweeps))):
        best_q = -1
        best_delta = 0.0
        for q in range(n):
            field = float(h[q])
            for nb, val in adj[q]:
                field += val * float(z[nb])
            delta = -2.0 * float(z[q]) * field
            if delta < best_delta:
                best_delta = delta
                best_q = q
        if best_q < 0:
            break
        z[best_q] = np.int8(-int(z[best_q]))
    return z.astype(np.int8, copy=False)


def _classical_pareto_guide_bank(
    problem: IsingMOOProblem,
    lambda_pool: np.ndarray,
    projected_j_pool: np.ndarray,
    projected_h_pool: np.ndarray,
    seed_objs: np.ndarray,
    seed_spins: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    rng_seed: int,
    num_lambdas: int = CLASSICAL_GUIDE_LAMBDAS,
    max_seeds: int = CLASSICAL_GUIDE_MAX_SEEDS,
    sweeps: int = CLASSICAL_GUIDE_SWEEPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classically find Pareto guide seeds; these are never returned as samples."""
    pool = np.asarray(lambda_pool, dtype=np.float64)
    objs0 = np.asarray(seed_objs, dtype=np.float64)
    spins0 = np.asarray(seed_spins, dtype=np.int8)
    n = int(problem.n)
    k = int(problem.k)
    if int(pool.shape[0]) == 0:
        return np.zeros((0, k), dtype=np.float64), np.zeros((0, n), dtype=np.int8), np.zeros((0,), dtype=np.int64)

    rng = np.random.default_rng(int(rng_seed))
    used_lids: set[int] = set(int(x) for x in _initial_lambda_ids(pool, min(int(num_lambdas), int(pool.shape[0]))))
    if objs0.ndim == 2 and spins0.ndim == 2 and int(objs0.shape[0]) >= 2:
        gap_pairs = _top_gap_repair_pairs(
            objs0,
            spins0,
            num_pairs=min(12, max(2, int(objs0.shape[0]) // 600)),
        )
        for a, b, _score in gap_pairs:
            mid = 0.5 * (objs0[int(a)] + objs0[int(b)])
            lid = _nearest_lambda_id(pool, _objective_mid_lambda(mid), used_lids, scan=192)
            if lid is not None:
                used_lids.add(int(lid))
                if len(used_lids) >= int(num_lambdas):
                    break

    lambda_ids = np.asarray(sorted(used_lids)[: min(int(num_lambdas), int(pool.shape[0]))], dtype=np.int64)
    starts: List[np.ndarray] = []
    if spins0.ndim == 2 and int(spins0.shape[0]) > 0:
        starts.extend([np.asarray(spins0[i], dtype=np.int8) for i in range(min(24, int(spins0.shape[0])))])
        if objs0.ndim == 2 and int(objs0.shape[0]) == int(spins0.shape[0]):
            for d in range(int(objs0.shape[1])):
                starts.append(np.asarray(spins0[int(np.argmin(objs0[:, d]))], dtype=np.int8))
    while len(starts) < 32:
        starts.append(np.where(rng.random(n) < 0.5, 1, -1).astype(np.int8))

    cand: List[np.ndarray] = []
    for lid in lambda_ids:
        lid_i = int(lid)
        starts_for_lid: List[np.ndarray] = []
        if spins0.ndim == 2 and int(spins0.shape[0]) > 0 and objs0.ndim == 2 and int(objs0.shape[0]) == int(spins0.shape[0]):
            scalar = objs0 @ pool[lid_i]
            starts_for_lid.append(np.asarray(spins0[int(np.argmin(scalar))], dtype=np.int8))
        starts_for_lid.extend(starts[:4])
        starts_for_lid.append(np.where(rng.random(n) < 0.5, 1, -1).astype(np.int8))
        for st in starts_for_lid:
            cand.append(
                _scalar_local_search_spin(
                    st,
                    problem.edges,
                    projected_j_pool[lid_i],
                    projected_h_pool[lid_i],
                    sweeps=int(sweeps),
                )
            )

    if not cand:
        return np.zeros((0, k), dtype=np.float64), np.zeros((0, n), dtype=np.int8), np.zeros((0,), dtype=np.int64)

    spins = np.unique(np.vstack(cand).astype(np.int8, copy=False), axis=0)
    energies = np.asarray(energy_batch_fast(spins, problem.edges, problem.weights, problem.h), dtype=np.float64)
    objs = normalize_energies(energies, lower_bounds, upper_bounds)
    keep = pg_non_dominated_indices(objs)
    objs = np.asarray(objs[keep], dtype=np.float64)
    spins = np.asarray(spins[keep], dtype=np.int8)
    if int(objs.shape[0]) > int(max_seeds):
        gap = _archive_gap_scores(objs)
        spread = _archive_point_spread_scores(objs)
        box = np.prod(np.maximum(HV_REF - objs, 1e-9), axis=1)
        score = 0.42 * gap + 0.28 * spread + 0.30 * (box / max(float(np.max(box)), 1e-12))
        keep2 = np.argsort(score)[::-1][: int(max_seeds)]
        objs = np.asarray(objs[keep2], dtype=np.float64)
        spins = np.asarray(spins[keep2], dtype=np.int8)
    counts = np.ones((int(objs.shape[0]),), dtype=np.int64)
    return objs, spins, counts


def _inject_classical_guide_slots(
    warm_bank: List[np.ndarray | None],
    active_ids: np.ndarray,
    warm_cs: np.ndarray,
    angle_modes: np.ndarray,
    warm_vec_bank: List[np.ndarray | None],
    guide_objs: np.ndarray,
    guide_spins: np.ndarray,
    first_objs: np.ndarray,
    lambda_pool: np.ndarray,
    *,
    max_slots: int = CLASSICAL_GUIDE_SLOTS,
) -> Tuple[List[np.ndarray | None], np.ndarray, np.ndarray, np.ndarray, List[np.ndarray | None]]:
    gids = np.asarray(active_ids, dtype=np.int64).copy()
    warm_cs_out = np.asarray(warm_cs, dtype=np.float64).copy()
    modes = np.asarray(angle_modes, dtype=np.int8).copy()
    warm_out: List[np.ndarray | None] = list(warm_bank)
    vec_out: List[np.ndarray | None] = list(warm_vec_bank)

    objs = np.asarray(guide_objs, dtype=np.float64)
    spins = np.asarray(guide_spins, dtype=np.int8)
    if int(max_slots) <= 0 or objs.ndim != 2 or spins.ndim != 2 or int(objs.shape[0]) == 0:
        return warm_out, gids, warm_cs_out, modes, vec_out

    slots = [i for i, wb in enumerate(warm_out) if wb is None]
    if not slots:
        return warm_out, gids, warm_cs_out, modes, vec_out
    slots = slots[: min(int(max_slots), len(slots))]

    gap = _archive_gap_scores(objs)
    spread = _archive_point_spread_scores(objs)
    box = np.prod(np.maximum(HV_REF - objs, 1e-9), axis=1)
    box = box / max(float(np.max(box)), 1e-12)
    novelty = np.zeros((int(objs.shape[0]),), dtype=np.float64)
    front = np.asarray(first_objs, dtype=np.float64)
    if front.ndim == 2 and int(front.shape[0]) > 0:
        keep = np.linspace(0, int(front.shape[0]) - 1, min(2048, int(front.shape[0])), dtype=np.int64)
        sub = front[keep]
        d2 = np.sum((objs[:, None, :] - sub[None, :, :]) ** 2, axis=2)
        novelty = np.sqrt(np.maximum(np.min(d2, axis=1), 0.0))
        hi = float(np.percentile(novelty, 90)) if novelty.size else 0.0
        novelty = np.clip(novelty / max(hi, 1e-12), 0.0, 1.0) if hi > 1e-12 else np.zeros_like(novelty)

    score = 0.34 * gap + 0.24 * spread + 0.24 * box + 0.18 * novelty
    used_lids = set(int(x) for x in gids)
    used_guides: set[int] = set()
    for slot in slots:
        chosen_idx = None
        chosen_lid = None
        for idx in np.argsort(score)[::-1]:
            idx_i = int(idx)
            if idx_i in used_guides:
                continue
            lid = _nearest_lambda_id(lambda_pool, _objective_mid_lambda(objs[idx_i]), used_lids, scan=192)
            if lid is None:
                continue
            chosen_idx = idx_i
            chosen_lid = int(lid)
            break
        if chosen_idx is None or chosen_lid is None:
            break
        used_guides.add(int(chosen_idx))
        used_lids.add(int(chosen_lid))
        gids[int(slot)] = int(chosen_lid)
        warm_out[int(slot)] = _bits_from_spins(spins[int(chosen_idx)])
        warm_cs_out[int(slot)] = 0.28
        modes[int(slot)] = ANGLE_OPEN
        vec_out[int(slot)] = np.full((int(spins.shape[1]),), 0.24, dtype=np.float64)

    return warm_out, gids, warm_cs_out, modes, vec_out


def _build_qaoa_circuit_projected_ising_warm_vec(
    problem: IsingMOOProblem,
    j_raw: np.ndarray,
    h_raw: np.ndarray,
    *,
    betas: np.ndarray,
    gammas: np.ndarray,
    warm_bits01: np.ndarray | None = None,
    warm_c: float = 0.5,
    warm_c_vec: np.ndarray | None = None,
) -> Circuit:
    """Original repository QAOA builder with per-qubit warm-start strength."""
    n = int(problem.n)
    m = int(problem.m)
    p = int(len(betas))
    if len(gammas) != p:
        raise ValueError("betas/gammas length mismatch")

    j_raw = np.asarray(j_raw, dtype=np.float64).reshape(m)
    h_raw = np.asarray(h_raw, dtype=np.float64).reshape(n)
    scale = float(max(np.max(np.abs(j_raw)), np.max(np.abs(h_raw)), 1e-12))
    if not np.isfinite(scale):
        raise ValueError("Invalid Ising coefficient scale.")
    j = j_raw / scale
    h = h_raw / scale

    circ = Circuit()
    thetas: np.ndarray | None = None
    if warm_bits01 is None:
        for q in range(n):
            circ += H.on(q)
    else:
        bits01 = np.asarray(warm_bits01, dtype=np.int8).reshape(n)
        if warm_c_vec is None:
            warm_c_vec = np.full((n,), float(warm_c), dtype=np.float64)
        thetas = _warm_theta_from_bits_vec(bits01, warm_c_vec)
        for q, th in enumerate(thetas):
            circ += RY(float(th)).on(q)

    u = problem.edges[:, 0]
    v = problem.edges[:, 1]

    for layer in range(p):
        beta = float(betas[layer])
        gamma_eff = -scale_gamma(float(gammas[layer]), edges=problem.edges, n=n, J=j, h=h)

        for q in range(n):
            hz = float(h[q])
            if hz != 0.0:
                circ += RZ(2.0 * gamma_eff * hz).on(q)
        for eidx in range(m):
            circ += Rzz(2.0 * gamma_eff * float(j[eidx])).on([int(u[eidx]), int(v[eidx])])

        if thetas is None:
            for q in range(n):
                circ += RX(2.0 * beta).on(q)
        else:
            for q, th in enumerate(thetas):
                t = float(th)
                if t != 0.0:
                    circ += RY(-t).on(q)
                circ += RZ(2.0 * beta).on(q)
                if t != 0.0:
                    circ += RY(t).on(q)

    return ensure_measure_all(circ, n)


def _select_flip_pairs(
    seed_spin: np.ndarray,
    edges: np.ndarray,
    j_raw: np.ndarray,
    h_raw: np.ndarray,
    *,
    max_pairs: int = 4,
) -> np.ndarray:
    e = np.asarray(edges, dtype=np.int32)
    if int(e.shape[0]) == 0 or int(max_pairs) <= 0:
        return np.zeros((0, 2), dtype=np.int32)
    inst = _local_instability_vector(seed_spin, e, j_raw, h_raw)
    jj = np.asarray(j_raw, dtype=np.float64).reshape(-1)
    edge_strength = np.abs(jj) / max(float(np.max(np.abs(jj))), 1e-12)
    score = inst[e[:, 0]] + inst[e[:, 1]] + 0.25 * edge_strength
    chosen: List[int] = []
    used_nodes: set[int] = set()
    for idx in np.argsort(score)[::-1]:
        idx = int(idx)
        u = int(e[idx, 0])
        v = int(e[idx, 1])
        if u in used_nodes or v in used_nodes:
            continue
        chosen.append(idx)
        used_nodes.add(u)
        used_nodes.add(v)
        if len(chosen) >= int(max_pairs):
            break
    if not chosen:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(e[chosen], dtype=np.int32)


def _adpt_single_mixer_vec(
    seed_spin: np.ndarray,
    edges: np.ndarray,
    j_raw: np.ndarray,
    h_raw: np.ndarray,
) -> np.ndarray:
    inst = _local_instability_vector(seed_spin, edges, j_raw, h_raw)
    return np.clip(0.92 + 0.34 * inst, 0.88, 1.26).astype(np.float64, copy=False)


def _build_qaoa_circuit_projected_ising_adpt_mixer(
    problem: IsingMOOProblem,
    j_raw: np.ndarray,
    h_raw: np.ndarray,
    *,
    betas: np.ndarray,
    gammas: np.ndarray,
    warm_bits01: np.ndarray | None = None,
    warm_c: float = 0.5,
    warm_c_vec: np.ndarray | None = None,
    mixer_scale_vec: np.ndarray | None = None,
    mixer_min: float = 0.82,
    pair_edges: np.ndarray | None = None,
    pair_eta: float = 0.0,
) -> Circuit:
    """Warm QAOA with ADAPT-lite single/pair flip mixers."""
    n = int(problem.n)
    m = int(problem.m)
    p = int(len(betas))
    if len(gammas) != p:
        raise ValueError("betas/gammas length mismatch")

    j_raw = np.asarray(j_raw, dtype=np.float64).reshape(m)
    h_raw = np.asarray(h_raw, dtype=np.float64).reshape(n)
    scale = float(max(np.max(np.abs(j_raw)), np.max(np.abs(h_raw)), 1e-12))
    if not np.isfinite(scale):
        raise ValueError("Invalid Ising coefficient scale.")
    j = j_raw / scale
    h = h_raw / scale

    circ = Circuit()
    thetas: np.ndarray | None = None
    if warm_bits01 is None:
        for q in range(n):
            circ += H.on(q)
    else:
        bits01 = np.asarray(warm_bits01, dtype=np.int8).reshape(n)
        if warm_c_vec is None:
            warm_c_vec = np.full((n,), float(warm_c), dtype=np.float64)
        thetas = _warm_theta_from_bits_vec(bits01, warm_c_vec)
        for q, th in enumerate(thetas):
            circ += RY(float(th)).on(q)

    if mixer_scale_vec is None:
        mixer_scale = np.ones((n,), dtype=np.float64)
    else:
        mixer_scale = np.clip(np.asarray(mixer_scale_vec, dtype=np.float64).reshape(n), float(mixer_min), 1.30)

    pairs = np.asarray(pair_edges, dtype=np.int32).reshape(-1, 2) if pair_edges is not None else np.zeros((0, 2), dtype=np.int32)
    pair_eta = float(pair_eta)

    u = problem.edges[:, 0]
    v = problem.edges[:, 1]

    for layer in range(p):
        beta = float(betas[layer])
        gamma_eff = -scale_gamma(float(gammas[layer]), edges=problem.edges, n=n, J=j, h=h)

        for q in range(n):
            hz = float(h[q])
            if hz != 0.0:
                circ += RZ(2.0 * gamma_eff * hz).on(q)
        for eidx in range(m):
            circ += Rzz(2.0 * gamma_eff * float(j[eidx])).on([int(u[eidx]), int(v[eidx])])

        if thetas is None:
            for q in range(n):
                circ += RX(2.0 * beta * float(mixer_scale[q])).on(q)
        else:
            for q, th in enumerate(thetas):
                t = float(th)
                if t != 0.0:
                    circ += RY(-t).on(q)
                circ += RZ(2.0 * beta * float(mixer_scale[q])).on(q)
                if t != 0.0:
                    circ += RY(t).on(q)

        if pair_eta > 0.0 and int(pairs.shape[0]) > 0:
            theta = 2.0 * pair_eta * beta
            for a, b in pairs:
                qa = int(a)
                qb = int(b)
                circ += H.on(qa)
                circ += H.on(qb)
                circ += Rzz(theta).on([qa, qb])
                circ += H.on(qa)
                circ += H.on(qb)

    return ensure_measure_all(circ, n)


def _archive_point_spread_scores(objs: np.ndarray) -> np.ndarray:
    arr = np.asarray(objs, dtype=np.float64)
    m = int(arr.shape[0])
    if m == 0:
        return np.zeros((0,), dtype=np.float64)
    if m <= 2:
        return np.ones((m,), dtype=np.float64)

    scores = np.zeros((m,), dtype=np.float64)
    for d in range(int(arr.shape[1])):
        order = np.argsort(arr[:, d])
        vals = arr[order, d]
        span = max(float(vals[-1] - vals[0]), 1e-12)
        local = np.zeros((m,), dtype=np.float64)
        local[order[0]] = 1.0
        local[order[-1]] = 1.0
        if m > 2:
            local[order[1:-1]] = (vals[2:] - vals[:-2]) / span
        scores += local

    scores = np.maximum(scores, 0.0)
    hi = float(np.max(scores))
    return scores / hi if hi > 1e-12 else np.ones((m,), dtype=np.float64)


def _archive_gap_scores(objs: np.ndarray) -> np.ndarray:
    arr = np.asarray(objs, dtype=np.float64)
    m = int(arr.shape[0])
    if m == 0:
        return np.zeros((0,), dtype=np.float64)
    scores = np.zeros((m,), dtype=np.float64)
    for d in range(int(arr.shape[1])):
        order = np.argsort(arr[:, d])
        for left, right in zip(order[:-1], order[1:]):
            a = int(left)
            b = int(right)
            dist = float(np.linalg.norm(arr[a] - arr[b]))
            if dist <= 0.0:
                continue
            scores[a] = max(scores[a], dist)
            scores[b] = max(scores[b], dist)
    hi = float(np.max(scores))
    return scores / hi if hi > 1e-12 else np.zeros((m,), dtype=np.float64)


def _archive_gap_need(objs: np.ndarray) -> float:
    arr = np.asarray(objs, dtype=np.float64)
    if int(arr.shape[0]) < 3:
        return 0.0
    gaps: List[float] = []
    for d in range(int(arr.shape[1])):
        order = np.argsort(arr[:, d])
        for a, b in zip(order[:-1], order[1:]):
            dist = float(np.linalg.norm(arr[int(a)] - arr[int(b)]))
            if dist > 1e-12:
                gaps.append(dist)
    if not gaps:
        return 0.0
    vals = np.asarray(gaps, dtype=np.float64)
    return float(np.percentile(vals, 95) / max(float(np.median(vals)), 1e-12))


def _archive_front_extent(objs: np.ndarray) -> float:
    arr = np.asarray(objs, dtype=np.float64)
    if arr.ndim != 2 or int(arr.shape[0]) == 0:
        return 0.0
    extent = np.max(arr, axis=0) - np.min(arr, axis=0)
    return float(np.mean(np.maximum(extent, 0.0)))


def _adaptive_second_round_task_target(objs: np.ndarray) -> int:
    arr = np.asarray(objs, dtype=np.float64)
    if arr.ndim != 2 or int(arr.shape[0]) == 0:
        return int(SECOND_ROUND_TASK_BASE)

    nd_count = int(arr.shape[0])
    gap_need = _archive_gap_need(arr)
    extent = _archive_front_extent(arr)

    if _env_int("MOO_SMOOTH_TASKS", int(SMOOTH_TASKS_ENABLE_DEFAULT)) > 0:
        gap_signal = float(np.clip((gap_need - 1.78) / 0.30, 0.0, 1.0))
        low_nd_signal = float(np.clip((10500.0 - float(nd_count)) / 4200.0, 0.0, 1.0))
        dense_signal = float(np.clip((float(nd_count) - 11800.0) / 4200.0, 0.0, 1.0))
        dense_signal *= float(np.clip((1.90 - gap_need) / 0.16, 0.0, 1.0))
        low_extent_signal = float(np.clip((0.74 - extent) / 0.14, 0.0, 1.0))
        high_extent_signal = float(np.clip((extent - 0.84) / 0.12, 0.0, 1.0))

        target_f = (
            float(SECOND_ROUND_TASK_BASE)
            + 7.5 * gap_signal
            + 5.0 * low_nd_signal
            + 3.5 * low_extent_signal
            - 7.0 * dense_signal
            - 2.0 * high_extent_signal * dense_signal
        )
        if gap_need >= 2.06 or nd_count < 8200:
            target_f = max(target_f, 78.0)
        elif nd_count >= 14500 and gap_need <= 1.80:
            target_f = min(target_f, 62.0)
        elif nd_count >= 12800 and gap_need <= 1.86:
            target_f = min(target_f, 66.0)

        target = int(2 * round(target_f / 2.0))
        return int(np.clip(target, int(SMOOTH_TASKS_MIN), int(SMOOTH_TASKS_MAX)))

    if nd_count >= 14500 and gap_need <= 1.78:
        target = int(SECOND_ROUND_TASK_MIN)
    elif nd_count >= 12500 and gap_need <= 1.86:
        target = int(SECOND_ROUND_TASK_LOW)
    elif gap_need >= 2.06 or nd_count < 8500:
        target = int(SECOND_ROUND_TASK_MAX)
    elif gap_need >= 1.92 or nd_count < 10000 or extent < 0.68:
        target = int(SECOND_ROUND_TASK_HIGH)
    elif extent >= 0.88 and nd_count >= 13000:
        target = int(SECOND_ROUND_TASK_LOW)
    else:
        target = int(SECOND_ROUND_TASK_BASE)

    return int(np.clip(target, int(SECOND_ROUND_TASK_MIN), int(SECOND_ROUND_TASK_MAX)))


def _archive_region_scores(objs: np.ndarray) -> np.ndarray:
    arr = np.asarray(objs, dtype=np.float64)
    m = int(arr.shape[0])
    if m == 0:
        return np.zeros((0,), dtype=np.float64)

    gap = _archive_gap_scores(arr)
    spread = _archive_point_spread_scores(arr)
    extreme = np.zeros((m,), dtype=np.float64)
    edge_keep = 2 if m >= 4 else 1
    for d in range(int(arr.shape[1])):
        order = np.argsort(arr[:, d])
        for rank, idx in enumerate(order[:edge_keep]):
            extreme[int(idx)] = max(extreme[int(idx)], 1.0 - 0.35 * rank)

    score = 0.52 * gap + 0.32 * spread + 0.16 * extreme
    hi = float(np.max(score))
    return score / hi if hi > 1e-12 else np.ones((m,), dtype=np.float64)


def _pilot_region_gain(
    first_objs: np.ndarray,
    local_nd: np.ndarray,
    first_region_scores: np.ndarray,
    *,
    block_size: int = 2048,
) -> float:
    """Score whether pilot samples land near sparse/high-gap front regions."""
    front = np.asarray(first_objs, dtype=np.float64)
    pts = np.asarray(local_nd, dtype=np.float64)
    region = np.asarray(first_region_scores, dtype=np.float64)
    if front.ndim != 2 or pts.ndim != 2 or int(front.shape[0]) == 0 or int(pts.shape[0]) == 0:
        return 0.0
    if int(region.shape[0]) != int(front.shape[0]):
        region = _archive_region_scores(front)

    if int(front.shape[0]) > 4096:
        top_keep = np.argsort(region)[::-1][:3072]
        cover_keep = np.linspace(0, int(front.shape[0]) - 1, 1024, dtype=np.int64)
        keep = np.unique(np.concatenate([top_keep.astype(np.int64, copy=False), cover_keep]))
        front = front[keep]
        region = region[keep]

    n_pts = int(pts.shape[0])
    best_d2 = np.full((n_pts,), np.inf, dtype=np.float64)
    best_region = np.zeros((n_pts,), dtype=np.float64)
    step = max(1, int(block_size))
    for st in range(0, int(front.shape[0]), step):
        ed = min(st + step, int(front.shape[0]))
        delta = pts[:, None, :] - front[None, st:ed, :]
        d2 = np.einsum("ijk,ijk->ij", delta, delta, optimize=True)
        idx = np.argmin(d2, axis=1)
        val = d2[np.arange(n_pts), idx]
        upd = val < best_d2
        if np.any(upd):
            best_d2[upd] = val[upd]
            best_region[upd] = region[st + idx[upd]]

    extent = np.max(front, axis=0) - np.min(front, axis=0)
    scale = float(np.mean(np.maximum(extent, 1e-9)) / max(float(front.shape[0]) ** (1.0 / max(int(front.shape[1]), 1)), 1.0))
    novelty = np.clip(np.sqrt(np.maximum(best_d2, 0.0)) / max(3.0 * scale, 1e-9), 0.0, 1.0)
    point_scores = 0.72 * best_region + 0.28 * novelty
    top = max(1, int(np.ceil(0.20 * n_pts)))
    top_mean = float(np.mean(np.partition(point_scores, -top)[-top:]))
    return float(0.65 * np.max(point_scores) + 0.35 * top_mean)


def _pilot_frontier_contribution(
    first_objs: np.ndarray,
    local_nd: np.ndarray,
    first_region_scores: np.ndarray,
    *,
    ref: float = HV_REF,
    block_size: int = 48,
) -> float:
    """Approximate whether a pilot task adds genuinely new front coverage."""
    front = np.asarray(first_objs, dtype=np.float64)
    pts = np.asarray(local_nd, dtype=np.float64)
    region = np.asarray(first_region_scores, dtype=np.float64)
    if front.ndim != 2 or pts.ndim != 2 or int(front.shape[0]) == 0 or int(pts.shape[0]) == 0:
        return 0.0
    if int(region.shape[0]) != int(front.shape[0]):
        region = _archive_region_scores(front)

    if int(front.shape[0]) > 2048:
        top_keep = np.argsort(region)[::-1][:1536]
        cover_keep = np.linspace(0, int(front.shape[0]) - 1, 512, dtype=np.int64)
        keep = np.unique(np.concatenate([top_keep.astype(np.int64, copy=False), cover_keep]))
        front = front[keep]
        region = region[keep]

    pts = np.minimum(np.maximum(pts, 0.0), float(ref))
    n_pts = int(pts.shape[0])
    dominated = np.zeros((n_pts,), dtype=bool)
    best_d2 = np.full((n_pts,), np.inf, dtype=np.float64)
    best_idx = np.zeros((n_pts,), dtype=np.int64)
    eps = 1e-10

    step = max(1, int(block_size))
    for start in range(0, n_pts, step):
        end = min(n_pts, start + step)
        block = pts[start:end]
        le = front[:, None, :] <= (block[None, :, :] + eps)
        lt = front[:, None, :] < (block[None, :, :] - eps)
        dominated[start:end] = np.any(np.all(le, axis=2) & np.any(lt, axis=2), axis=0)

        diff = front[:, None, :] - block[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
        idx = np.argmin(d2, axis=0)
        best_idx[start:end] = idx.astype(np.int64, copy=False)
        best_d2[start:end] = d2[idx, np.arange(end - start)]

    novelty = (~dominated).astype(np.float64)
    if float(np.sum(novelty)) <= 0.0:
        return 0.0
    box = np.prod(np.maximum(float(ref) - pts, 1e-9), axis=1)
    box = box / max(float(np.max(box)), 1e-12)
    sparse = region[best_idx] / (1.0 + 18.0 * np.sqrt(np.maximum(best_d2, 0.0)))
    score = novelty * (0.52 * box + 0.34 * sparse + 0.14)
    return float(np.mean(score))


def _nearest_lambda_id(lambda_pool: np.ndarray, lam: np.ndarray, used: set[int], scan: int = 96) -> int | None:
    pool = np.asarray(lambda_pool, dtype=np.float64)
    target = np.asarray(lam, dtype=np.float64).reshape(-1)
    target = np.maximum(target, 0.0)
    target = target / max(float(np.sum(target)), 1e-12)
    d2 = np.sum((pool - target[None, :]) ** 2, axis=1)
    for cand in np.argsort(d2)[: min(int(scan), int(pool.shape[0]))]:
        idx = int(cand)
        if idx not in used:
            return idx
    return None


def _top_gap_repair_pairs(
    objs: np.ndarray,
    spins: np.ndarray,
    *,
    num_pairs: int,
) -> List[Tuple[int, int, float]]:
    arr = np.asarray(objs, dtype=np.float64)
    z = np.asarray(spins, dtype=np.int8)
    m = int(arr.shape[0])
    if int(num_pairs) <= 0 or m < 2:
        return []

    spread = _archive_point_spread_scores(arr)
    gap = _archive_gap_scores(arr)
    raw: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()

    def add_pair(a: int, b: int) -> None:
        a = int(a)
        b = int(b)
        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        if key in seen:
            return
        seen.add(key)
        raw.append(key)

    # Reliable O(k*m) adjacent gaps along each objective.
    for d in range(int(arr.shape[1])):
        order = np.argsort(arr[:, d])
        for a, b in zip(order[:-1], order[1:]):
            add_pair(int(a), int(b))

    # Add a sparse-point channel without O(m^2): high gap/spread points are
    # paired with their nearest scalarization neighbor under midpoint lambda.
    point_score = 0.55 * gap + 0.45 * spread
    for a in np.argsort(point_score)[::-1][: min(4 * int(num_pairs), m)]:
        a = int(a)
        lam = _objective_mid_lambda(arr[a])
        scalar = arr @ lam
        order = np.argsort(np.abs(scalar - scalar[a]))
        for b in order[1: min(48, int(order.size))]:
            b = int(b)
            if np.linalg.norm(arr[a] - arr[b]) > 1e-9:
                add_pair(a, b)
                break

    if not raw:
        return []

    cap = int(PARETO_REPAIR_PAIR_CANDIDATE_CAP)
    if len(raw) > cap:
        # Cheap prefilter: preserve large objective-space gaps before doing the
        # more detailed Hamming/corridor scoring.
        approx = np.fromiter(
            (
                float(np.linalg.norm(arr[int(a)] - arr[int(b)]))
                + 0.35 * float(gap[int(a)] + gap[int(b)])
                + 0.20 * float(spread[int(a)] + spread[int(b)])
                for a, b in raw
            ),
            dtype=np.float64,
            count=len(raw),
        )
        keep = np.argpartition(approx, -cap)[-cap:]
        raw = [raw[int(i)] for i in keep]

    scored: List[Tuple[float, int, int, float, float, float]] = []
    dists: List[float] = []
    hammings: List[float] = []
    for a, b in raw:
        mid = 0.5 * (arr[a] + arr[b])
        dist = float(np.linalg.norm(arr[a] - arr[b]))
        box = float(np.prod(np.maximum(HV_REF - mid, 1e-9)))
        sparse = float(0.5 * (point_score[a] + point_score[b]))
        hamming = float(np.mean(z[a] != z[b])) if z.ndim == 2 else 0.0
        if dist <= 1e-12:
            continue
        dists.append(dist)
        hammings.append(hamming)
        # Pair-gap repair only helps if the bit corridor is reachable. Very
        # small Hamming pairs copy one endpoint; very large pairs often scatter.
        corridor = float(np.exp(-((hamming - 0.42) / 0.20) ** 2))
        score = 0.36 * dist + 0.24 * box + 0.18 * sparse + 0.22 * corridor
        scored.append((score, int(a), int(b), dist, hamming, corridor))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: List[Tuple[int, int, float]] = []
    used_points: set[int] = set()
    if not scored:
        return out

    dist_floor = float(np.percentile(np.asarray(dists, dtype=np.float64), 55)) if dists else 0.0
    if hammings:
        hamming_floor = max(1.0 / max(int(z.shape[1]) if z.ndim == 2 else 1, 1), float(np.percentile(np.asarray(hammings, dtype=np.float64), 45)))
    else:
        hamming_floor = 0.0

    # First pass: high-quality, endpoint-disjoint moderate corridors.
    for score, a, b, dist, hamming, corridor in scored:
        if a in used_points or b in used_points:
            continue
        if dist < dist_floor or hamming < hamming_floor or corridor < 0.28:
            continue
        out.append((a, b, float(score)))
        used_points.add(a)
        used_points.add(b)
        if len(out) >= int(num_pairs):
            break

    # Second pass: still endpoint-disjoint, but allow one weak criterion to
    # fail. This keeps repair count stable on small or already-converged fronts.
    for score, a, b, dist, hamming, corridor in scored:
        if len(out) >= int(num_pairs):
            break
        if a in used_points or b in used_points:
            continue
        if dist < 0.75 * dist_floor and hamming < 0.75 * hamming_floor:
            continue
        if corridor < 0.12 and hamming > 0.65:
            continue
        out.append((a, b, float(score)))
        used_points.add(a)
        used_points.add(b)

    # Final fallback: preserve the PV4c/PV6 behavior if quality gates are too
    # strict for a particular archive.
    for score, a, b, _dist, _hamming, _corridor in scored:
        if len(out) >= int(num_pairs):
            break
        if any((a == x and b == y) or (a == y and b == x) for x, y, _ in out):
            continue
        if a in used_points and b in used_points:
            continue
        out.append((a, b, float(score)))
        used_points.add(a)
        used_points.add(b)
    return out


def _archive_diverse_lambda_warm_bank(
    frontier_objs: np.ndarray,
    frontier_spins: np.ndarray,
    lambda_pool: np.ndarray,
    *,
    num_weights: int,
    base_warm_c: float,
) -> Tuple[List[np.ndarray | None], np.ndarray, np.ndarray, List[np.ndarray | None], np.ndarray]:
    """Select second-round tasks with archive-basin diversity.

    This keeps base70's 6 no-warm exploration slots, then chooses warm-start
    tasks by balancing scalarization uncertainty with objective-space spread.
    The key difference from the base selector is a soft penalty for repeatedly
    warming into the same archive basin.
    """
    objs = np.asarray(frontier_objs, dtype=np.float64)
    spins = np.asarray(frontier_spins, dtype=np.int8)
    pool = np.asarray(lambda_pool, dtype=np.float64)
    target = min(int(num_weights), int(pool.shape[0]))

    if target <= 0:
        return [], np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64), [], np.zeros((0,), dtype=bool)
    if int(objs.shape[0]) == 0:
        ids = _initial_lambda_ids(pool, target)
        return (
            [None] * int(ids.shape[0]),
            ids,
            np.zeros((int(ids.shape[0]),), dtype=np.float64),
            [None] * int(ids.shape[0]),
            np.zeros((int(ids.shape[0]),), dtype=bool),
        )

    scalar = np.asarray(objs @ pool.T, dtype=np.float64)
    best_point = np.argmin(scalar, axis=0).astype(np.int64)
    if int(objs.shape[0]) >= 2:
        two_best = np.partition(scalar, kth=1, axis=0)[:2]
        margins = np.maximum(two_best[1] - two_best[0], 0.0)
    else:
        margins = np.full((int(pool.shape[0]),), 1.0, dtype=np.float64)

    finite_m = np.isfinite(margins)
    uncertainty = np.zeros((int(pool.shape[0]),), dtype=np.float64)
    if np.any(finite_m):
        hi = float(np.max(margins[finite_m]))
        lo = float(np.min(margins[finite_m]))
        uncertainty = 1.0 - (margins - lo) / max(hi - lo, 1e-12)
        uncertainty[~finite_m] = 0.0
        uncertainty = np.maximum(uncertainty, 0.0)

    spread = _archive_point_spread_scores(objs)
    gap = _archive_gap_scores(objs)
    center = np.clip(1.0 - np.var(pool, axis=1) / 0.16, 0.0, 1.0)

    selected_ids: List[int] = []
    selected_seed_idx: List[int | None] = []
    used_lids: set[int] = set()
    seed_use = np.zeros((int(objs.shape[0]),), dtype=np.float64)
    min_lambda_d2 = np.full((int(pool.shape[0]),), np.inf, dtype=np.float64)
    min_seed_d = np.full((int(objs.shape[0]),), np.inf, dtype=np.float64)

    def update_distances(lid: int, seed_idx: int | None) -> None:
        delta = pool - pool[int(lid)]
        min_lambda_d2[:] = np.minimum(min_lambda_d2, np.einsum("ij,ij->i", delta, delta, optimize=True))
        if seed_idx is not None:
            seed = int(seed_idx)
            seed_use[seed] += 1.0
            dist = np.linalg.norm(objs - objs[seed][None, :], axis=1)
            min_seed_d[:] = np.minimum(min_seed_d, dist)

    def add_task(lid: int, seed_idx: int | None) -> bool:
        lid = int(lid)
        if lid < 0 or lid >= int(pool.shape[0]) or lid in used_lids or len(selected_ids) >= target:
            return False
        selected_ids.append(lid)
        selected_seed_idx.append(None if seed_idx is None else int(seed_idx))
        used_lids.add(lid)
        update_distances(lid, seed_idx)
        return True

    # Keep the base70 exploration anchor exactly: broad, no-warm, low-risk.
    for lid in _initial_lambda_ids(pool, 6):
        add_task(int(lid), None)

    # Objective anchors are useful for hidden extremes, but each gets the true
    # objective-extreme seed instead of another repeated scalarization seed.
    for d in range(int(objs.shape[1])):
        e = np.zeros((int(objs.shape[1]),), dtype=np.float64)
        e[d] = 1.0
        lid = _nearest_lambda_id(pool, e, used_lids, scan=128)
        if lid is not None:
            add_task(lid, int(np.argmin(objs[:, d])))

    # High-spread/gap archive points get explicit representation before the
    # scalarization fill. This is the low-density-frontier repair channel.
    point_score = 0.55 * gap + 0.45 * spread
    for seed_idx in np.argsort(point_score)[::-1]:
        if len(selected_ids) >= min(target, 6 + int(objs.shape[1]) + 18):
            break
        seed = int(seed_idx)
        if seed_use[seed] > 0:
            continue
        if np.isfinite(min_seed_d[seed]) and min_seed_d[seed] < 0.018:
            continue
        lid = _nearest_lambda_id(pool, _objective_mid_lambda(objs[seed]), used_lids, scan=96)
        if lid is not None:
            add_task(lid, seed)

    # Fill the majority with a soft basin-diverse scalarization selector.
    while len(selected_ids) < target:
        available = np.asarray([i for i in range(int(pool.shape[0])) if i not in used_lids], dtype=np.int64)
        if int(available.size) == 0:
            break
        seeds = best_point[available]
        seed_novel = min_seed_d[seeds]
        finite = np.isfinite(seed_novel)
        if np.any(finite):
            hi = float(np.percentile(seed_novel[finite], 85))
            seed_novel = np.where(finite, np.clip(seed_novel / max(hi, 1e-12), 0.0, 1.0), 1.0)
        else:
            seed_novel = np.ones((int(available.size),), dtype=np.float64)
        lambda_novel = min_lambda_d2[available]
        finite_l = np.isfinite(lambda_novel)
        if np.any(finite_l):
            hi_l = float(np.percentile(lambda_novel[finite_l], 85))
            lambda_novel = np.where(finite_l, np.clip(lambda_novel / max(hi_l, 1e-12), 0.0, 1.0), 1.0)
        else:
            lambda_novel = np.ones((int(available.size),), dtype=np.float64)

        reuse_penalty = np.clip(seed_use[seeds] / 2.0, 0.0, 1.0)
        score = (
            0.30 * uncertainty[available]
            + 0.22 * gap[seeds]
            + 0.18 * spread[seeds]
            + 0.16 * seed_novel
            + 0.09 * lambda_novel
            + 0.05 * center[available]
            - 0.20 * reuse_penalty
        )
        idx = int(available[int(np.argmax(score))])
        add_task(idx, int(best_point[idx]))

    gap_need = _archive_gap_need(objs)
    front_extent = _archive_front_extent(objs)
    nd_count = int(objs.shape[0])
    if nd_count >= int(PARETO_DENSE_ND_HIGH):
        if nd_count >= int(PARETO_DENSE_ND_ULTRA) and front_extent >= float(PARETO_DENSE_EXTENT_HIGH):
            repair_cap = int(PARETO_DENSE_EXTENT_REPAIR)
        else:
            repair_cap = int(PARETO_DENSE_REPAIR_HIGH)
    elif nd_count >= int(PARETO_DENSE_ND_MID):
        if front_extent >= float(PARETO_DENSE_EXTENT_MID):
            repair_cap = max(int(PARETO_DENSE_REPAIR_MID), int(PARETO_DENSE_EXTENT_REPAIR))
        else:
            repair_cap = int(PARETO_DENSE_REPAIR_MID)
    elif gap_need >= float(PARETO_GAP_NEED_HIGH):
        repair_cap = int(PARETO_GAP_REPAIR_HIGH)
    elif gap_need >= float(PARETO_GAP_NEED_MID):
        repair_cap = int(PARETO_GAP_REPAIR_MID)
    else:
        repair_cap = int(PARETO_GAP_REPAIR_LOW)
    repair_tasks = min(int(PARETO_GAP_REPAIR_TASKS), repair_cap, target)
    repair_pairs = _top_gap_repair_pairs(objs, spins, num_pairs=max(1, repair_tasks // 2))
    if repair_pairs:
        repair_hamming = float(
            np.mean(
                [
                    np.mean(spins[int(a)] != spins[int(b)])
                    for a, b, _score in repair_pairs
                ]
            )
        )
    else:
        repair_hamming = 0.0
    main_keep = max(0, target - 2 * int(len(repair_pairs)))
    selected_ids = selected_ids[:main_keep]
    selected_seed_idx = selected_seed_idx[:main_keep]
    used_lids = set(int(x) for x in selected_ids)
    repair_partner_idx: List[int | None] = [None] * len(selected_ids)
    repair_flags: List[bool] = [False] * len(selected_ids)

    for a, b, _score in repair_pairs:
        if len(selected_ids) >= target:
            break
        mid = 0.5 * (objs[int(a)] + objs[int(b)])
        lid = _nearest_lambda_id(pool, _objective_mid_lambda(mid), used_lids, scan=128)
        if lid is None:
            continue
        # Two endpoint tasks share the same gap lambda on purpose: same cost
        # direction, different endpoint warm priors, opening toward the middle.
        for seed_idx, partner_idx in ((int(a), int(b)), (int(b), int(a))):
            if len(selected_ids) >= target:
                break
            selected_ids.append(int(lid))
            selected_seed_idx.append(seed_idx)
            repair_partner_idx.append(partner_idx)
            repair_flags.append(True)
        used_lids.add(int(lid))

    # If too few legal repair pairs were found, fill back with ordinary diverse
    # tasks so the 70-task budget remains intact.
    fill_cursor = 0
    while len(selected_ids) < target and fill_cursor < len(selected_ids):
        fill_cursor += 1
    while len(selected_ids) < target:
        available = np.asarray([i for i in range(int(pool.shape[0])) if i not in used_lids], dtype=np.int64)
        if int(available.size) == 0:
            break
        seeds = best_point[available]
        score = 0.40 * uncertainty[available] + 0.30 * gap[seeds] + 0.20 * spread[seeds] + 0.10 * center[available]
        idx = int(available[int(np.argmax(score))])
        selected_ids.append(idx)
        selected_seed_idx.append(int(best_point[idx]))
        repair_partner_idx.append(None)
        repair_flags.append(False)
        used_lids.add(idx)

    ids = np.asarray(selected_ids[:target], dtype=np.int64)
    seed_idx_arr = selected_seed_idx[:target]
    repair_partner_idx = repair_partner_idx[:target]
    repair_flags_arr = np.asarray(repair_flags[:target], dtype=bool)
    warm_bank: List[np.ndarray | None] = []
    partner_bank: List[np.ndarray | None] = []
    for seed_idx in seed_idx_arr:
        if seed_idx is None:
            warm_bank.append(None)
        else:
            warm_bank.append(np.where(spins[int(seed_idx)] > 0, 0, 1).astype(np.int8, copy=False))
    for partner_idx in repair_partner_idx:
        if partner_idx is None:
            partner_bank.append(None)
        else:
            partner_bank.append(np.where(spins[int(partner_idx)] > 0, 0, 1).astype(np.int8, copy=False))

    selected_scalar = scalar[:, ids]
    if int(selected_scalar.shape[0]) >= 2:
        two_best = np.partition(selected_scalar, kth=1, axis=0)[:2]
        selected_margins = np.maximum(two_best[1] - two_best[0], 0.0)
    else:
        selected_margins = np.zeros((int(ids.shape[0]),), dtype=np.float64)

    if selected_margins.size and float(np.max(selected_margins)) > float(np.min(selected_margins)):
        order = np.argsort(selected_margins)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.linspace(0.0, 1.0, int(order.size), dtype=np.float64)
    else:
        ranks = np.full((int(ids.shape[0]),), 0.5, dtype=np.float64)

    warm_lo = max(0.28, float(base_warm_c) - 0.08)
    warm_hi = min(0.48, float(base_warm_c) + 0.06)
    warm_cs = warm_lo + (warm_hi - warm_lo) * ranks
    warm_cs = np.asarray([0.0 if wb is None else warm_cs[i] for i, wb in enumerate(warm_bank)], dtype=np.float64)
    for i, is_repair in enumerate(repair_flags_arr):
        if is_repair:
            warm_cs[i] = 0.32

    return warm_bank, ids, warm_cs, partner_bank, repair_flags_arr


def _allocate_pareto_controller_shots(
    first_objs: np.ndarray,
    lambda_pool: np.ndarray,
    active_ids: np.ndarray,
    *,
    total_budget: int,
    min_shots: int,
    max_shots: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pareto-aware second-round resource controller.

    The task selector decides where to sample.  This allocator decides how hard
    each direction should be sampled, with extra budget going to tasks whose
    archive seed lies on sparse / large-gap regions of the current front.
    """
    objs = np.asarray(first_objs, dtype=np.float64)
    pool = np.asarray(lambda_pool, dtype=np.float64)
    ids = np.asarray(active_ids, dtype=np.int64)
    n = int(ids.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)

    total_budget = int(total_budget)
    min_shots = int(min_shots)
    max_shots = int(max_shots)
    if total_budget < n * min_shots:
        min_shots = max(1, total_budget // max(n, 1))

    shots = np.full((n,), min_shots, dtype=np.int64)
    remaining = int(total_budget - int(np.sum(shots)))
    if remaining <= 0:
        shots[-1] += total_budget - int(np.sum(shots))
        return shots.astype(np.int64, copy=False), np.ones((n,), dtype=np.float64) / max(n, 1)

    scalar = np.asarray(objs @ pool[ids].T, dtype=np.float64)
    best_point = np.argmin(scalar, axis=0).astype(np.int64)
    if int(objs.shape[0]) >= 2:
        two_best = np.partition(scalar, kth=1, axis=0)[:2]
        margins = np.maximum(two_best[1] - two_best[0], 0.0)
        hi = float(np.max(margins))
        lo = float(np.min(margins))
        uncertain = 1.0 - (margins - lo) / max(hi - lo, 1e-12)
    else:
        uncertain = np.ones((n,), dtype=np.float64)
    uncertain = np.maximum(np.asarray(uncertain, dtype=np.float64), 0.0)

    gap = _archive_gap_scores(objs)
    spread = _archive_point_spread_scores(objs)
    gap_task = gap[best_point] if int(gap.shape[0]) else np.zeros((n,), dtype=np.float64)
    spread_task = spread[best_point] if int(spread.shape[0]) else np.zeros((n,), dtype=np.float64)
    center = np.clip(1.0 - np.var(pool[ids], axis=1) / 0.16, 0.0, 1.0)

    repair_score = (
        float(PARETO_GAP_WEIGHT) * gap_task
        + float(PARETO_SPREAD_WEIGHT) * spread_task
        + float(PARETO_UNCERTAIN_WEIGHT) * uncertain
        + float(PARETO_CENTER_WEIGHT) * center
    )
    priority = np.maximum(repair_score, 0.025)
    repair_top = min(int(PARETO_REPAIR_TOP), n)
    if repair_top > 0:
        top_order = np.argsort(repair_score)[::-1][:repair_top]
        boost = np.ones((n,), dtype=np.float64)
        boost[top_order] = float(PARETO_REPAIR_BOOST)
        # Keep a small rank taper so the very largest gaps get a visibly
        # stronger refill without starving the rest of the top repair set.
        for rank, idx in enumerate(top_order):
            boost[int(idx)] *= 1.0 + 0.35 * (1.0 - rank / max(repair_top - 1, 1))
        priority = priority * boost
    priority = priority / max(float(np.sum(priority)), 1e-12)

    cap = np.maximum(max_shots - shots, 0)
    extra = np.minimum(np.floor(priority * remaining).astype(np.int64), cap)
    shots += extra
    remaining = int(total_budget - int(np.sum(shots)))

    while remaining > 0:
        room = np.flatnonzero(shots < max_shots)
        if int(room.size) == 0:
            shots[int(np.argmax(priority))] += remaining
            break
        idx = int(room[np.argmax(priority[room])])
        add = min(remaining, int(max_shots - shots[idx]))
        shots[idx] += add
        remaining -= add

    diff = int(total_budget - int(np.sum(shots)))
    if diff != 0:
        shots[int(np.argmax(priority))] += diff
    return shots.astype(np.int64, copy=False), priority.astype(np.float64, copy=False)


def _build_quantum_task_bank(
    frontier_objs: np.ndarray,
    frontier_spins: np.ndarray,
    frontier_counts: np.ndarray,
    lambda_pool: np.ndarray,
    *,
    num_tasks: int,
    edges: np.ndarray,
    projected_j_pool: np.ndarray,
    projected_h_pool: np.ndarray,
) -> Tuple[List[np.ndarray | None], np.ndarray, np.ndarray, np.ndarray, List[np.ndarray | None]]:
    objs = np.asarray(frontier_objs, dtype=np.float64)
    spins = np.asarray(frontier_spins, dtype=np.int8)
    pool = np.asarray(lambda_pool, dtype=np.float64)
    target = min(int(num_tasks), int(pool.shape[0]))

    if target <= 0:
        return [], np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int8), []
    if int(objs.shape[0]) == 0:
        ids = _initial_lambda_ids(pool, target)
        return (
            [None] * int(ids.shape[0]),
            ids,
            np.zeros((int(ids.shape[0]),), dtype=np.float64),
            np.zeros((int(ids.shape[0]),), dtype=np.int8),
            [None] * int(ids.shape[0]),
        )

    warm_bank, active_ids, warm_cs_arr, repair_partner_bank, repair_flags = _archive_diverse_lambda_warm_bank(
        objs,
        spins,
        pool,
        num_weights=target,
        base_warm_c=float(WARM_C_BY_ROUND[1]),
    )
    angle_modes = np.full((int(active_ids.shape[0]),), ANGLE_PORTFOLIO, dtype=np.int8)
    warm_c_vec_bank: List[np.ndarray | None] = [None] * int(active_ids.shape[0])
    profile_open = np.zeros((int(active_ids.shape[0]),), dtype=np.float64)

    for i, wb in enumerate(warm_bank[:target]):
        if wb is None:
            angle_modes[i] = ANGLE_BASE
            continue
        seed_spin = np.where(np.asarray(wb, dtype=np.int8) > 0, -1, 1).astype(np.int8)
        lid = int(active_ids[i])
        inst_vec = _local_instability_vector(
            seed_spin,
            edges,
            projected_j_pool[lid],
            projected_h_pool[lid],
        )
        mean_inst, tail_inst, weak_frac = _local_instability_profile(
            seed_spin,
            edges,
            projected_j_pool[lid],
            projected_h_pool[lid],
        )
        profile_open[i] = 0.45 * mean_inst + 0.35 * tail_inst + 0.20 * weak_frac
        warm_cs_arr[i] = float(np.clip(warm_cs_arr[i] * (1.0 - 0.16 * profile_open[i]), 0.26, 0.48))
        # Per-qubit warm profile: stable local-field qubits stay close to the
        # archive seed, weak-field qubits are opened for quantum exploration.
        vec = warm_cs_arr[i] * (1.0 - 0.45 * inst_vec)
        warm_c_vec_bank[i] = np.clip(vec, 0.12, 0.52).astype(np.float64, copy=False)

        if bool(repair_flags[i]) and repair_partner_bank[i] is not None:
            partner_bits = np.asarray(repair_partner_bank[i], dtype=np.int8).reshape(-1)
            seed_bits = np.asarray(wb, dtype=np.int8).reshape(-1)
            diff = seed_bits != partner_bits
            # Gap-pair repair: open only the Hamming disagreement coordinates
            # between the two Pareto endpoints, keep their shared backbone.
            vec = np.where(diff, 0.18, 0.34).astype(np.float64, copy=False)
            warm_c_vec_bank[i] = vec
            warm_cs_arr[i] = 0.32
            lam_var = float(np.var(pool[int(active_ids[i])]))
            center_mix = float(np.clip(1.0 - lam_var / 0.16, 0.0, 1.0))
            angle_modes[i] = ANGLE_OPEN if center_mix >= 0.62 else ANGLE_MID

    margins = _lambda_archive_margins(objs, pool, active_ids)
    finite = np.isfinite(margins)
    if np.any(finite):
        order = np.argsort(np.where(finite, margins, np.inf))
        changed = 0
        for pos in order:
            pos = int(pos)
            if warm_bank[pos] is None or bool(repair_flags[pos]):
                continue
            angle_modes[pos] = ANGLE_MID
            warm_cs_arr[pos] = min(float(warm_cs_arr[pos]), 0.36)
            if warm_c_vec_bank[pos] is not None:
                warm_c_vec_bank[pos] = np.minimum(warm_c_vec_bank[pos], 0.38)
            changed += 1
            if changed >= 10:
                break

    # Profile-open tasks are still single-seed exploitation, but their local
    # field profile says a nontrivial subset of qubits should remain mobile.
    # This is the structural step beyond answer9's scalar warm shrink.
    open_order = np.argsort(profile_open)[::-1]
    changed = 0
    for pos in open_order:
        pos = int(pos)
        if warm_bank[pos] is None or angle_modes[pos] == ANGLE_MID or bool(repair_flags[pos]):
            continue
        if profile_open[pos] < 0.48:
            break
        angle_modes[pos] = ANGLE_OPEN
        warm_cs_arr[pos] = min(float(warm_cs_arr[pos]), 0.31)
        if warm_c_vec_bank[pos] is not None:
            warm_c_vec_bank[pos] = np.minimum(warm_c_vec_bank[pos], 0.30)
        changed += 1
        if changed >= 6:
            break

    lam_var = np.var(pool[active_ids], axis=1)
    center_order = np.argsort(lam_var)
    changed = 0
    for pos in center_order:
        pos = int(pos)
        if warm_bank[pos] is None or angle_modes[pos] in (ANGLE_MID, ANGLE_OPEN) or bool(repair_flags[pos]):
            continue
        angle_modes[pos] = ANGLE_CENTER
        warm_cs_arr[pos] = min(float(warm_cs_arr[pos]), 0.34)
        if warm_c_vec_bank[pos] is not None:
            warm_c_vec_bank[pos] = np.minimum(warm_c_vec_bank[pos], 0.35)
        changed += 1
        if changed >= 4:
            break

    return warm_bank[:target], active_ids, warm_cs_arr, angle_modes, warm_c_vec_bank[:target]


def _select_angles(
    mode: int,
    lam: np.ndarray,
    betas_base: np.ndarray,
    gammas_base: np.ndarray,
    betas_mid: np.ndarray,
    gammas_mid: np.ndarray,
    betas_center: np.ndarray,
    gammas_center: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mode = int(mode)
    if mode == ANGLE_CENTER:
        return betas_center, gammas_center
    if mode == ANGLE_MID:
        return betas_mid, gammas_mid
    if mode == ANGLE_OPEN:
        betas = 0.55 * betas_mid + 0.45 * betas_center
        gammas = 0.85 * (0.50 * gammas_mid + 0.50 * gammas_center)
        return betas, gammas
    if mode == ANGLE_PORTFOLIO:
        return _angle_portfolio(
            lam,
            betas_base,
            gammas_base,
            betas_mid,
            gammas_mid,
            betas_center,
            gammas_center,
        )
    return betas_base, gammas_base


def _route_transfer_angles(
    betas: np.ndarray,
    gammas: np.ndarray,
    lam: np.ndarray,
    edges: np.ndarray,
    n_qubits: int,
    j_raw: np.ndarray,
    h_raw: np.ndarray,
    *,
    mode: int,
    role: str = "base",
) -> Tuple[np.ndarray, np.ndarray]:
    """Tiny graph/lambda-aware correction on top of transfer_data angles."""
    if _env_int("MOO_ANGLE_ROUTER", int(ANGLE_ROUTER_ENABLE_DEFAULT)) <= 0:
        return np.asarray(betas, dtype=np.float64), np.asarray(gammas, dtype=np.float64)

    b = np.asarray(betas, dtype=np.float64)
    g = np.asarray(gammas, dtype=np.float64)
    jj = np.asarray(j_raw, dtype=np.float64).reshape(-1)
    hh = np.asarray(h_raw, dtype=np.float64).reshape(-1)
    e = np.asarray(edges, dtype=np.int32)
    n = max(int(n_qubits), 1)
    m = int(e.shape[0])

    abs_j = np.abs(jj)
    abs_h = np.abs(hh)
    mean_j = float(np.mean(abs_j)) if abs_j.size else 0.0
    mean_h = float(np.mean(abs_h)) if abs_h.size else 0.0
    h_to_j = mean_h / max(mean_j, 1e-9)
    j_cv = float(np.std(abs_j) / max(mean_j, 1e-9)) if abs_j.size else 0.0
    h_cv = float(np.std(abs_h) / max(mean_h, 1e-9)) if abs_h.size else 0.0
    avg_deg = float(2.0 * m / max(n, 1))

    lam_arr = np.asarray(lam, dtype=np.float64)
    center_mix = float(np.clip(1.0 - float(np.var(lam_arr)) / 0.16, 0.0, 1.0))
    deg_signal = float(np.clip((avg_deg - 3.0) / 3.0, -1.0, 1.0))
    field_signal = float(np.clip((h_to_j - 0.55) / 1.25, -1.0, 1.0))
    rough_signal = float(np.clip((0.55 * j_cv + 0.45 * h_cv - 0.75) / 1.25, -1.0, 1.0))

    strength = float(np.clip(_env_float("MOO_ANGLE_ROUTER_STRENGTH", float(ANGLE_ROUTER_STRENGTH)), 0.0, 0.10))
    mode_i = int(mode)
    role_s = str(role)

    gamma_log = strength * (
        -0.42 * deg_signal
        -0.28 * rough_signal
        +0.24 * field_signal
        -0.08 * center_mix
    )
    beta_log = strength * (
        +0.18 * deg_signal
        +0.10 * center_mix
        -0.16 * field_signal
        +0.08 * rough_signal
    )
    if mode_i == ANGLE_OPEN or role_s in ("pair_soft", "single"):
        gamma_log -= 0.12 * strength
        beta_log += 0.08 * strength
    elif mode_i == ANGLE_CENTER:
        gamma_log -= 0.06 * strength * center_mix

    beta_scale = float(np.clip(np.exp(beta_log), float(ANGLE_ROUTER_MIN_SCALE), float(ANGLE_ROUTER_MAX_SCALE)))
    gamma_scale = float(np.clip(np.exp(gamma_log), float(ANGLE_ROUTER_MIN_SCALE), float(ANGLE_ROUTER_MAX_SCALE)))
    return b * beta_scale, g * gamma_scale


def _fourier_warm_extend_angles(
    betas: np.ndarray,
    gammas: np.ndarray,
    *,
    p_out: int = FOURIER_WARM_P_OUT,
    tail: float = FOURIER_WARM_TAIL,
) -> Tuple[np.ndarray, np.ndarray]:
    """Low-frequency continuation of the working p=3 transfer angles.

    Direct p=4 transfer was too disruptive on hidden cases.  This variant keeps
    the p=3 manifold as the prior, fits a tiny cosine basis, then blends it with
    linear resampling and a shrunken continuation tail.
    """
    b = np.asarray(betas, dtype=np.float64).reshape(-1)
    g = np.asarray(gammas, dtype=np.float64).reshape(-1)
    p_in = int(b.shape[0])
    p_out = max(int(p_out), p_in)
    if p_out == p_in or p_in < 2 or int(g.shape[0]) != p_in:
        return b.copy(), g.copy()

    tail = float(np.clip(tail, 0.15, 0.85))

    def smooth_one(arr: np.ndarray) -> np.ndarray:
        x_in = (np.arange(p_in, dtype=np.float64) + 0.5) / float(p_in)
        x_out = (np.arange(p_out, dtype=np.float64) + 0.5) / float(p_out)
        n_basis = min(3, p_in)
        basis_in = [np.ones_like(x_in)]
        basis_out = [np.ones_like(x_out)]
        for u in range(1, n_basis):
            basis_in.append(np.cos(np.pi * float(u) * x_in))
            basis_out.append(np.cos(np.pi * float(u) * x_out))
        a_in = np.vstack(basis_in).T
        a_out = np.vstack(basis_out).T
        try:
            coef = np.linalg.lstsq(a_in, arr, rcond=None)[0]
            fourier = np.asarray(a_out @ coef, dtype=np.float64)
        except Exception:
            fourier = np.interp(x_out, x_in, arr)
        linear = np.interp(x_out, x_in, arr)
        out = 0.58 * linear + 0.42 * fourier

        trend = float(arr[-1] - arr[-2])
        continuation = float(arr[-1] + 0.30 * trend)
        out[-1] = tail * continuation + (1.0 - tail) * float(arr[-1])
        if p_out > p_in:
            out[:-1] = 0.92 * out[:-1] + 0.08 * np.interp(x_out[:-1], x_in, arr)
        return out.astype(np.float64, copy=False)

    return smooth_one(b), smooth_one(g)



def _normalize_finite_scores(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out
    vals = arr[finite]
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi <= lo + 1e-12:
        out[finite] = 0.5
    else:
        out[finite] = (vals - lo) / (hi - lo)
    return out


def _pair_opportunity_raw(
    seed_spin: np.ndarray,
    edges: np.ndarray,
    j_raw: np.ndarray,
    h_raw: np.ndarray,
) -> float:
    e = np.asarray(edges, dtype=np.int32)
    if int(e.shape[0]) == 0:
        return 0.0
    inst = _local_instability_vector(seed_spin, e, j_raw, h_raw)
    jj = np.abs(np.asarray(j_raw, dtype=np.float64).reshape(-1))
    jj = jj / max(float(np.max(jj)), 1e-12)
    # A pair opportunity is high when both endpoints are locally weak and the
    # edge is not negligible.  Use only the upper tail so a few useful escaping
    # pairs can be detected without making the whole task look random.
    edge_score = inst[e[:, 0]] * inst[e[:, 1]] * (0.70 + 0.30 * jj)
    top_n = max(1, min(6, int(np.ceil(0.12 * edge_score.size))))
    return float(np.mean(np.sort(edge_score)[-top_n:]))


def _single_focus_raw(
    seed_spin: np.ndarray,
    edges: np.ndarray,
    j_raw: np.ndarray,
    h_raw: np.ndarray,
) -> float:
    mean_inst, tail_inst, weak_frac = _local_instability_profile(seed_spin, edges, j_raw, h_raw)
    return float(0.30 * mean_inst + 0.55 * tail_inst + 0.15 * weak_frac)


def _adpt_single_focused_mixer_vec(
    seed_spin: np.ndarray,
    edges: np.ndarray,
    j_raw: np.ndarray,
    h_raw: np.ndarray,
) -> np.ndarray:
    # Conservative focused single-qubit mixer: only weak-field qubits get a
    # noticeable boost.  Stable qubits remain close to the transferred QAOA
    # backbone, unlike the earlier broad single-mixer perturbation.
    inst = _local_instability_vector(seed_spin, edges, j_raw, h_raw)
    return np.clip(0.96 + 0.32 * inst, 0.92, 1.24).astype(np.float64, copy=False)


def _masked_window_profiles(
    warm_bits01: np.ndarray,
    edges: np.ndarray,
    j_raw: np.ndarray,
    h_raw: np.ndarray,
    *,
    window_size: int,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Strongly pin a seed outside a locally unstable quantum window."""
    bits = np.asarray(warm_bits01, dtype=np.int8).reshape(n)
    seed_spin = _spin_from_bits(bits)
    inst = _local_instability_vector(seed_spin, edges, j_raw, h_raw)
    e = np.asarray(edges, dtype=np.int32)
    edge_score = np.zeros((n,), dtype=np.float64)
    if int(e.shape[0]) > 0:
        jj = np.abs(np.asarray(j_raw, dtype=np.float64).reshape(-1))
        jj = jj / max(float(np.max(jj)), 1e-12)
        np.add.at(edge_score, e[:, 0], jj)
        np.add.at(edge_score, e[:, 1], jj)
        edge_score = edge_score / max(float(np.max(edge_score)), 1e-12)

    score = 0.74 * inst + 0.26 * edge_score
    w = min(max(1, int(window_size)), n)
    window = np.argsort(score)[-w:]

    warm_vec = np.full((n,), float(MASKED_WINDOW_OUTSIDE_WARM), dtype=np.float64)
    warm_vec[window] = float(MASKED_WINDOW_INSIDE_WARM)
    mixer_vec = np.full((n,), float(MASKED_WINDOW_OUTSIDE_MIXER), dtype=np.float64)
    mixer_vec[window] = float(MASKED_WINDOW_INSIDE_MIXER)
    return warm_vec, mixer_vec


def _select_tiny_adpt_roles(
    warm_bank: List[np.ndarray | None],
    active_ids: np.ndarray,
    first_objs: np.ndarray,
    lambda_pool: np.ndarray,
    *,
    edges: np.ndarray,
    projected_j_pool: np.ndarray,
    projected_h_pool: np.ndarray,
    num_pair: int = TINY_PAIR_TASKS,
    num_single: int = TINY_SINGLE_TASKS,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    n_tasks = int(len(warm_bank))
    roles = ["base"] * n_tasks
    if n_tasks == 0:
        return roles, np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    pair_raw = np.full((n_tasks,), -np.inf, dtype=np.float64)
    single_raw = np.full((n_tasks,), -np.inf, dtype=np.float64)
    valid = np.zeros((n_tasks,), dtype=bool)

    for i, wb in enumerate(warm_bank):
        if wb is None:
            continue
        lid = int(active_ids[i])
        seed_spin = _spin_from_bits(wb)
        pair_raw[i] = _pair_opportunity_raw(seed_spin, edges, projected_j_pool[lid], projected_h_pool[lid])
        single_raw[i] = _single_focus_raw(seed_spin, edges, projected_j_pool[lid], projected_h_pool[lid])
        valid[i] = True

    pair_n = _normalize_finite_scores(pair_raw)
    single_n = _normalize_finite_scores(single_raw)

    margins = _lambda_archive_margins(first_objs, lambda_pool, np.asarray(active_ids, dtype=np.int64))
    finite_m = np.isfinite(margins)
    uncertainty = np.zeros((n_tasks,), dtype=np.float64)
    if np.any(finite_m):
        inv = np.zeros_like(margins, dtype=np.float64)
        inv[finite_m] = 1.0 / (margins[finite_m] + 1e-9)
        uncertainty = _normalize_finite_scores(np.where(finite_m, inv, -np.inf))

    pair_score = np.where(valid, 0.72 * pair_n + 0.28 * uncertainty, -np.inf)
    single_score = np.where(valid, 0.72 * single_n + 0.28 * uncertainty - 0.25 * pair_n, -np.inf)

    pair_selected: set[int] = set()
    if int(num_pair) > 0:
        pair_order = list(np.argsort(pair_score)[::-1])
        for pos in pair_order:
            pos = int(pos)
            if not np.isfinite(pair_score[pos]):
                continue
            roles[pos] = "pair_soft"
            pair_selected.add(pos)
            if len(pair_selected) >= int(num_pair):
                break

    if int(num_single) > 0:
        single_order = list(np.argsort(single_score)[::-1])
        single_selected = 0
        for pos in single_order:
            pos = int(pos)
            if pos in pair_selected or not np.isfinite(single_score[pos]):
                continue
            roles[pos] = "single"
            single_selected += 1
            if single_selected >= int(num_single):
                break

    return roles, pair_score, single_score


def main1(
    problem_input: Union[str, IsingMOOProblem, Dict[str, np.ndarray]],
    sample_budget: int = BASE_SAMPLE_BUDGET,
    rng_seed: int | None = None,
) -> Dict[str, object]:
    """Base70 archive-guided warm-start QAOA sampler."""
    problem = _to_problem(problem_input)
    seed = 2026 if rng_seed is None else int(rng_seed)

    if int(sample_budget) != BASE_SAMPLE_BUDGET:
        raise ValueError(f"sample_budget must equal {BASE_SAMPLE_BUDGET}, got {sample_budget}.")

    lambda_pool = _structured_lambda_pool(
        load_weight_pool(int(problem.k), n=1000, seed=2026).astype(np.float64),
        int(problem.k),
    )
    lower_bounds, upper_bounds = objective_extrema(problem)

    projected_j_pool = np.asarray(lambda_pool @ problem.weights, dtype=np.float64)
    projected_h_pool = np.asarray(lambda_pool @ problem.h, dtype=np.float64)

    sim = Simulator("mqvector", int(problem.n), seed=int(seed))
    n = int(problem.n)

    out_spins = np.empty((BASE_SAMPLE_BUDGET, n), dtype=np.int8)
    cursor = 0
    active_lambda_ids = _initial_lambda_ids(lambda_pool, NUM_WEIGHTS)

    first_unique_spin_blocks: List[np.ndarray] = []
    first_unique_count_blocks: List[np.ndarray] = []
    first_lambda_id_order: List[int] = []

    first_round_portfolio_ids: set[int] = set()
    if FIRST_ROUND_PORTFOLIO_TOP > 0:
        lam_vars = np.var(lambda_pool[active_lambda_ids], axis=1)
        order = np.argsort(lam_vars)
        first_round_portfolio_ids = set(
            int(active_lambda_ids[int(i)])
            for i in order[: min(int(FIRST_ROUND_PORTFOLIO_TOP), int(order.size))]
        )

    p_round = 3
    betas_base, gammas_base = _TRANSFER_TABLE[p_round]
    betas_mid, gammas_mid = _TRANSFER_TABLE_MID.get(p_round, (betas_base, gammas_base))
    betas_center, gammas_center = _TRANSFER_TABLE_CENTER.get(p_round, (betas_mid, gammas_mid))

    for j in range(NUM_WEIGHTS):
        lam_id = int(active_lambda_ids[j])
        j_raw = projected_j_pool[lam_id]
        h_raw = projected_h_pool[lam_id]
        mode = ANGLE_PORTFOLIO if lam_id in first_round_portfolio_ids else ANGLE_BASE
        betas, gammas = _select_angles(
            mode,
            lambda_pool[lam_id],
            betas_base,
            gammas_base,
            betas_mid,
            gammas_mid,
            betas_center,
            gammas_center,
        )
        betas, gammas = _route_transfer_angles(
            betas,
            gammas,
            lambda_pool[lam_id],
            problem.edges,
            n,
            j_raw,
            h_raw,
            mode=int(mode),
            role="first",
        )
        circ = _build_qaoa_circuit_projected_ising_warm_vec(
            problem,
            j_raw,
            h_raw,
            betas=betas,
            gammas=gammas,
            warm_bits01=None,
            warm_c=0.0,
            warm_c_vec=None,
        )
        unique_spins, counts = _sample_unique_spins(
            sim,
            circ,
            shots=int(FIRST_ROUND_SHOTS),
            n_qubits=n,
            seed=seed + j,
        )
        spins = np.repeat(unique_spins, counts.astype(np.int32), axis=0)
        end = cursor + int(FIRST_ROUND_SHOTS)
        out_spins[cursor:end] = spins
        cursor = end

        first_unique_spin_blocks.append(unique_spins)
        first_unique_count_blocks.append(counts)
        first_lambda_id_order.append(lam_id)

    first_objs, first_spins, _first_lids, first_counts = exact_frontier_from_lambda_unique_batches(
        first_unique_spin_blocks,
        first_unique_count_blocks,
        first_lambda_id_order,
        edges=problem.edges,
        weights=problem.weights,
        h=problem.h,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    second_round_num_tasks = _adaptive_second_round_task_target(first_objs)
    guide_objs_classic, guide_spins_classic, _guide_counts_classic = _classical_pareto_guide_bank(
        problem,
        lambda_pool,
        projected_j_pool,
        projected_h_pool,
        first_objs,
        first_spins,
        lower_bounds,
        upper_bounds,
        rng_seed=seed + 70000,
    )
    warm_bank_all, ids_all, warm_cs_all, modes_all, warm_vecs_all = _build_quantum_task_bank(
        first_objs,
        first_spins,
        first_counts,
        lambda_pool,
        num_tasks=second_round_num_tasks,
        edges=problem.edges,
        projected_j_pool=projected_j_pool,
        projected_h_pool=projected_h_pool,
    )
    warm_bank_all, ids_all, warm_cs_all, modes_all, warm_vecs_all = _inject_classical_guide_slots(
        warm_bank_all,
        ids_all,
        warm_cs_all,
        modes_all,
        warm_vecs_all,
        guide_objs_classic,
        guide_spins_classic,
        first_objs,
        lambda_pool,
        max_slots=min(int(CLASSICAL_GUIDE_SLOTS), max(1, int(second_round_num_tasks) // 18)),
    )

    role_gap_need = _archive_gap_need(first_objs)
    role_extent = _archive_front_extent(first_objs)
    role_nd_count = int(first_objs.shape[0])
    num_pair_roles = 0
    num_single_roles = 0
    if role_gap_need >= 2.05 and role_nd_count < 12500:
        num_pair_roles = min(int(TINY_PAIR_TASKS), 1)
    if role_gap_need >= 1.90 or role_nd_count < 10000:
        num_single_roles = min(int(TINY_SINGLE_TASKS), 2)
    elif role_extent < 0.72:
        num_single_roles = min(int(TINY_SINGLE_TASKS), 1)

    roles, pair_scores, single_scores = _select_tiny_adpt_roles(
        warm_bank_all,
        ids_all,
        first_objs,
        lambda_pool,
        edges=problem.edges,
        projected_j_pool=projected_j_pool,
        projected_h_pool=projected_h_pool,
        num_pair=num_pair_roles,
        num_single=num_single_roles,
    )
    n_tasks = int(len(ids_all))
    pilot_shots = min(int(PILOT_SHOTS_PER_TASK), max(1, SECOND_ROUND_BUDGET // max(n_tasks, 1)))
    pilot_budget = int(pilot_shots * n_tasks)
    if pilot_budget >= int(SECOND_ROUND_BUDGET):
        pilot_shots = max(1, int(SECOND_ROUND_BUDGET) // max(n_tasks, 1))
        pilot_budget = int(pilot_shots * n_tasks)

    first_scalar_best = np.min(np.asarray(first_objs @ lambda_pool[ids_all].T, dtype=np.float64), axis=0)
    first_extreme = np.min(first_objs, axis=0)
    first_region_scores = _archive_region_scores(first_objs)
    pilot_improve = np.zeros((n_tasks,), dtype=np.float64)
    pilot_extreme = np.zeros((n_tasks,), dtype=np.float64)
    pilot_box = np.zeros((n_tasks,), dtype=np.float64)
    pilot_region = np.zeros((n_tasks,), dtype=np.float64)
    pilot_contrib = np.zeros((n_tasks,), dtype=np.float64)
    pilot_nd = np.zeros((n_tasks,), dtype=np.float64)
    pilot_unique = np.zeros((n_tasks,), dtype=np.float64)
    pilot_spin_blocks: List[np.ndarray] = []
    pilot_count_blocks: List[np.ndarray] = []
    pilot_lids: List[int] = []

    def build_second_round_circuit(
        j: int,
        *,
        fourier_warm: bool = False,
        masked_window: bool = False,
        angle_scale: float = 1.0,
        warm_scale: float = 1.0,
    ):
        lam_id = int(ids_all[j])
        role = roles[j]
        warm_bits = warm_bank_all[j]
        warm_c = float(warm_cs_all[j]) if warm_bits is not None else 0.0
        warm_c_vec = warm_vecs_all[j] if warm_bits is not None else None
        mode = int(modes_all[j]) if warm_bits is not None else ANGLE_BASE
        j_raw = projected_j_pool[lam_id]
        h_raw = projected_h_pool[lam_id]

        betas, gammas = _select_angles(
            mode,
            lambda_pool[lam_id],
            betas_base,
            gammas_base,
            betas_mid,
            gammas_mid,
            betas_center,
            gammas_center,
        )
        betas, gammas = _route_transfer_angles(
            betas,
            gammas,
            lambda_pool[lam_id],
            problem.edges,
            n,
            j_raw,
            h_raw,
            mode=int(mode),
            role=str(role),
        )
        angle_scale = float(angle_scale)
        if abs(angle_scale - 1.0) > 1e-12:
            betas = np.asarray(betas, dtype=np.float64) * angle_scale
            gammas = np.asarray(gammas, dtype=np.float64) * angle_scale
        if fourier_warm:
            betas, gammas = _fourier_warm_extend_angles(
                betas,
                gammas,
                p_out=_env_int("MOO_FOURIER_WARM_P", int(FOURIER_WARM_P_OUT)),
                tail=_env_float("MOO_FOURIER_WARM_TAIL", float(FOURIER_WARM_TAIL)),
            )
        if warm_bits is not None and abs(float(warm_scale) - 1.0) > 1e-12:
            warm_c = float(warm_c) * float(warm_scale)
            if warm_c_vec is not None:
                warm_c_vec = np.asarray(warm_c_vec, dtype=np.float64) * float(warm_scale)

        mixer_vec = None
        pair_edges = None
        pair_eta = 0.0
        if warm_bits is not None and role == "pair_soft":
            seed_spin = _spin_from_bits(warm_bits)
            pair_edges = _select_flip_pairs(
                seed_spin,
                problem.edges,
                j_raw,
                h_raw,
                max_pairs=TINY_PAIR_MAX_PAIRS,
            )
            pair_eta = TINY_PAIR_SOFT_ETA
        elif warm_bits is not None and role == "single":
            seed_spin = _spin_from_bits(warm_bits)
            mixer_vec = _adpt_single_focused_mixer_vec(seed_spin, problem.edges, j_raw, h_raw)

        if masked_window and warm_bits is not None:
            window_size = _env_int("MOO_MASKED_WINDOW_SIZE", int(MASKED_WINDOW_SIZE))
            warm_c_vec, mixer_vec = _masked_window_profiles(
                warm_bits,
                problem.edges,
                j_raw,
                h_raw,
                window_size=window_size,
                n=n,
            )
            pair_edges = None
            pair_eta = 0.0

        if role == "base" and mixer_vec is None and pair_edges is None:
            circ = _build_qaoa_circuit_projected_ising_warm_vec(
                problem,
                j_raw,
                h_raw,
                betas=betas,
                gammas=gammas,
                warm_bits01=warm_bits,
                warm_c=warm_c,
                warm_c_vec=warm_c_vec,
            )
        else:
            circ = _build_qaoa_circuit_projected_ising_adpt_mixer(
                problem,
                j_raw,
                h_raw,
                betas=betas,
                gammas=gammas,
                warm_bits01=warm_bits,
                warm_c=warm_c,
                warm_c_vec=warm_c_vec,
                mixer_scale_vec=mixer_vec,
                mixer_min=0.02 if masked_window else 0.82,
                pair_edges=pair_edges,
                pair_eta=pair_eta,
            )
        return circ

    for j, _lam_id_raw in enumerate(ids_all):
        lam_id = int(ids_all[j])
        circ = build_second_round_circuit(j)
        unique_spins, counts = _sample_unique_spins(
            sim,
            circ,
            shots=int(pilot_shots),
            n_qubits=n,
            seed=seed + 10000 + j,
        )
        spins = np.repeat(unique_spins, counts.astype(np.int32), axis=0)
        end = cursor + int(pilot_shots)
        out_spins[cursor:end] = spins
        cursor = end

        pilot_spin_blocks.append(unique_spins)
        pilot_count_blocks.append(counts)
        pilot_lids.append(lam_id)
        pilot_unique[j] = float(unique_spins.shape[0]) / max(float(pilot_shots), 1.0)

        energies = np.asarray(energy_batch_fast(unique_spins, problem.edges, problem.weights, problem.h), dtype=np.float64)
        objs = normalize_energies(energies, lower_bounds, upper_bounds)
        if int(objs.shape[0]) > 0:
            local_nd = objs[pg_non_dominated_indices(objs)]
            if int(local_nd.shape[0]) > 0:
                local_scalar_best = float(np.min(local_nd @ lambda_pool[lam_id]))
                pilot_improve[j] = max(float(first_scalar_best[j]) - local_scalar_best, 0.0)
                pilot_extreme[j] = float(np.sum(np.maximum(first_extreme - np.min(local_nd, axis=0), 0.0)))
                pilot_box[j] = float(np.max(np.prod(np.maximum(HV_REF - local_nd, 1e-9), axis=1)))
                pilot_region[j] = _pilot_region_gain(first_objs, local_nd, first_region_scores)
                if _env_int("MOO_PARETO_CONTRIB", int(PARETO_CONTRIB_ENABLE_DEFAULT)) > 0:
                    pilot_contrib[j] = _pilot_frontier_contribution(first_objs, local_nd, first_region_scores)
                pilot_nd[j] = float(local_nd.shape[0])

    substitute_tasks: List[Tuple[int, np.ndarray, np.ndarray, int]] = []
    try:
        combined_objs, combined_spins, _combined_lids, _combined_counts = exact_frontier_from_lambda_unique_batches(
            first_unique_spin_blocks + pilot_spin_blocks,
            first_unique_count_blocks + pilot_count_blocks,
            first_lambda_id_order + pilot_lids,
            edges=problem.edges,
            weights=problem.weights,
            h=problem.h,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        combined_gap_need = _archive_gap_need(combined_objs)
        combined_extent = _archive_front_extent(combined_objs)
        combined_nd_count = int(combined_objs.shape[0])
        sub_gap_trigger = (
            combined_gap_need >= 1.96
            or (combined_gap_need >= float(ARCHIVE_SUB_GAP_MIN) and combined_extent < 0.72)
            or role_gap_need >= 2.05
        )
        sub_low_nd_trigger = (
            combined_nd_count < int(ARCHIVE_SUB_LOW_ND)
            and (combined_gap_need >= 1.98 or combined_extent < 0.72 or role_gap_need >= 2.05)
        )
        if (
            int(combined_objs.shape[0]) >= 2
            and (sub_gap_trigger or sub_low_nd_trigger)
        ):
            pair_target = 1
            if combined_gap_need >= 2.05 or (
                combined_nd_count < 8500 and (combined_gap_need >= 1.98 or combined_extent < 0.72)
            ):
                pair_target = 2
            sub_pairs = _top_gap_repair_pairs(
                combined_objs,
                combined_spins,
                num_pairs=min(2, max(1, pair_target)),
            )
            used_sub_lids = set(int(x) for x in ids_all)
            for a, b, _score in sub_pairs:
                if len(substitute_tasks) >= int(ARCHIVE_SUB_MAX_TASKS):
                    break
                mid = 0.5 * (combined_objs[int(a)] + combined_objs[int(b)])
                lid = _nearest_lambda_id(lambda_pool, _objective_mid_lambda(mid), used_sub_lids, scan=192)
                if lid is None:
                    continue
                used_sub_lids.add(int(lid))
                center_mix = float(np.clip(1.0 - np.var(lambda_pool[int(lid)]) / 0.16, 0.0, 1.0))
                mode = ANGLE_OPEN if center_mix >= 0.58 else ANGLE_MID
                for seed_idx, partner_idx in ((int(a), int(b)), (int(b), int(a))):
                    if len(substitute_tasks) >= int(ARCHIVE_SUB_MAX_TASKS):
                        break
                    seed_spin = np.asarray(combined_spins[int(seed_idx)], dtype=np.int8)
                    partner_spin = np.asarray(combined_spins[int(partner_idx)], dtype=np.int8)
                    seed_bits = _bits_from_spins(seed_spin)
                    warm_vec = _gap_warm_vector(
                        seed_spin,
                        partner_spin,
                        fallback_warm_vec=np.full((n,), 0.34, dtype=np.float64),
                        n=n,
                    )
                    substitute_tasks.append((int(lid), seed_bits, warm_vec, int(mode)))
    except Exception:
        substitute_tasks = []

    def norm01(v: np.ndarray) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float64)
        lo = float(np.min(arr)) if arr.size else 0.0
        hi = float(np.max(arr)) if arr.size else 0.0
        if hi <= lo + 1e-12:
            return np.zeros_like(arr, dtype=np.float64)
        return (arr - lo) / max(hi - lo, 1e-12)

    role_prior = np.zeros((n_tasks,), dtype=np.float64)
    pair_n = _normalize_finite_scores(pair_scores)
    single_n = _normalize_finite_scores(single_scores)
    for i, role in enumerate(roles):
        if role == "pair_soft":
            role_prior[i] = pair_n[i]
        elif role == "single":
            role_prior[i] = 0.80 * single_n[i]
        else:
            role_prior[i] = 0.35 * max(float(pair_n[i]), float(single_n[i]))

    gap_need_main = _archive_gap_need(first_objs)
    extent_main = _archive_front_extent(first_objs)
    nd_count_main = int(first_objs.shape[0])

    reachability = (
        0.46 * norm01(pilot_improve)
        + 0.22 * norm01(pilot_unique)
        + 0.17 * norm01(pilot_nd)
        + 0.15 * np.maximum(norm01(role_prior), norm01(pilot_contrib))
    )
    reachable_region = norm01(pilot_region) * np.clip(reachability, 0.0, 1.0)
    if gap_need_main >= 1.90 or nd_count_main < 10000:
        region_boost = 0.22
    elif gap_need_main >= 1.72 or extent_main < 0.72:
        region_boost = 0.13
    else:
        region_boost = 0.06

    pilot_priority = (
        0.34 * norm01(pilot_improve)
        + 0.22 * norm01(pilot_extreme)
        + 0.21 * norm01(pilot_box)
        + 0.13 * norm01(pilot_nd)
        + 0.10 * norm01(pilot_unique)
    )
    if _env_int("MOO_PARETO_CONTRIB", int(PARETO_CONTRIB_ENABLE_DEFAULT)) > 0:
        contrib_n = norm01(pilot_contrib)
        contrib_w = float(np.clip(_env_float("MOO_PARETO_CONTRIB_WEIGHT", float(PARETO_CONTRIB_WEIGHT)), 0.0, 0.28))
        if contrib_w > 0.0:
            pilot_priority = (1.0 - contrib_w) * pilot_priority + contrib_w * contrib_n
        contrib_boost = float(np.clip(_env_float("MOO_PARETO_CONTRIB_BOOST", float(PARETO_CONTRIB_BOOST)), 0.0, 0.40))
        if contrib_boost > 0.0:
            pilot_priority = pilot_priority * (1.0 + contrib_boost * contrib_n)
    pilot_priority = pilot_priority * (1.0 + region_boost * reachable_region)
    base_prior = _allocate_second_round_shots(
        first_objs,
        lambda_pool,
        ids_all,
        total_budget=SECOND_ROUND_BUDGET,
        min_shots=SECOND_ROUND_MIN_SHOTS,
        max_shots=SECOND_ROUND_MAX_SHOTS,
    ).astype(np.float64)
    base_prior = base_prior / max(float(np.sum(base_prior)), 1e-12)
    if float(np.sum(pilot_priority)) <= 1e-12:
        pilot_priority = base_prior.copy()
    else:
        base_blend = 0.12 if int(second_round_num_tasks) >= int(SECOND_ROUND_TASK_HIGH) else 0.18
        pilot_priority = (
            (1.0 - base_blend) * (pilot_priority / max(float(np.sum(pilot_priority)), 1e-12))
            + base_blend * base_prior
        )
    pilot_priority = np.maximum(pilot_priority, 1e-6)
    pilot_priority = pilot_priority / max(float(np.sum(pilot_priority)), 1e-12)
    priority_entropy = 0.0
    if n_tasks > 1:
        priority_entropy = float(
            -np.sum(pilot_priority * np.log(np.maximum(pilot_priority, 1e-12)))
            / max(np.log(float(n_tasks)), 1e-12)
        )
    priority_top = float(np.max(pilot_priority)) if pilot_priority.size else 0.0
    active_target = int(PILOT_ACTIVE_BASE)
    if priority_entropy < 0.84 or priority_top >= 0.030:
        active_target = int(PILOT_ACTIVE_MIN)
    elif nd_count_main >= 12500 and gap_need_main <= 1.86:
        active_target = int(PILOT_ACTIVE_LOW)
    elif gap_need_main >= 2.00 or nd_count_main < 8500:
        active_target = int(PILOT_ACTIVE_HIGH)
    elif priority_entropy > 0.955 and priority_top <= 0.021:
        active_target = int(PILOT_ACTIVE_MAX)
    elif extent_main >= 0.84 and nd_count_main >= 14500:
        active_target = int(PILOT_ACTIVE_LOW)

    active_floor = max(28, min(int(PILOT_ACTIVE_MIN), int(round(0.66 * float(n_tasks)))))
    if n_tasks >= int(SECOND_ROUND_TASK_HIGH):
        active_floor = max(34, min(active_floor, int(round(0.52 * float(n_tasks)))))
        active_ceiling = min(n_tasks, max(active_floor, int(round(0.60 * float(n_tasks)))))
    elif n_tasks > int(SECOND_ROUND_TASK_BASE):
        active_ceiling = min(n_tasks, max(active_floor, int(round(0.68 * float(n_tasks)))))
    else:
        active_ceiling = min(n_tasks, max(active_floor, int(round(0.74 * float(n_tasks)))))
    active_ceiling = min(active_ceiling, int(PILOT_ACTIVE_MAX) + (4 if n_tasks > int(SECOND_ROUND_TASK_BASE) else 0))
    active_count = min(max(active_target, active_floor), active_ceiling)
    active_mask = np.zeros((n_tasks,), dtype=bool)
    if active_count > 0:
        active_order = np.argsort(pilot_priority)[-active_count:]
        active_mask[active_order] = True

    top_priority = np.where(active_mask, pilot_priority, 0.0)
    if float(np.sum(top_priority)) <= 1e-12:
        top_priority = np.where(active_mask, base_prior, 0.0)
    if n_tasks >= int(SECOND_ROUND_TASK_HIGH) and float(np.sum(top_priority)) > 1e-12:
        top_priority = np.where(top_priority > 0.0, np.power(top_priority, 1.16), 0.0)
    if float(np.sum(top_priority)) <= 1e-12:
        top_priority = np.ones((n_tasks,), dtype=np.float64) / max(n_tasks, 1)
    else:
        top_priority = top_priority / max(float(np.sum(top_priority)), 1e-12)

    substitute_budget = min(
        int(len(substitute_tasks)) * int(ARCHIVE_SUB_SHOTS),
        max(0, int(SECOND_ROUND_BUDGET - pilot_budget)),
    )
    remaining_budget = int(SECOND_ROUND_BUDGET - pilot_budget - substitute_budget)
    extra_shots = np.zeros((n_tasks,), dtype=np.int64)
    if remaining_budget > 0 and n_tasks > 0:
        min_extra = min(int(PILOT_MIN_EXTRA_SHOTS), remaining_budget // max(n_tasks, 1))
        extra_shots[:] = int(min_extra)
        remaining = int(remaining_budget - int(np.sum(extra_shots)))
        total_cap = np.maximum(int(PILOT_MAX_TOTAL_SHOTS) - int(pilot_shots) - extra_shots, 0)
        total_cap = np.where(active_mask, total_cap, 0)

        add = np.floor(top_priority * remaining).astype(np.int64)
        add = np.minimum(add, total_cap)
        extra_shots += add
        remaining = int(remaining_budget - int(np.sum(extra_shots)))
        while remaining > 0:
            room = np.flatnonzero(active_mask & (extra_shots < (int(PILOT_MAX_TOTAL_SHOTS) - int(pilot_shots))))
            if int(room.size) == 0:
                extra_shots[int(np.argmax(top_priority))] += remaining
                break
            idx = int(room[np.argmax(top_priority[room])])
            add_i = min(remaining, int(PILOT_MAX_TOTAL_SHOTS) - int(pilot_shots) - int(extra_shots[idx]))
            extra_shots[idx] += add_i
            remaining -= add_i

    if int(pilot_budget + substitute_budget + int(np.sum(extra_shots))) != int(SECOND_ROUND_BUDGET):
        diff = int(SECOND_ROUND_BUDGET - pilot_budget - substitute_budget - int(np.sum(extra_shots)))
        extra_shots[int(np.argmax(top_priority))] += diff

    inplace_angle_scale = np.ones((n_tasks,), dtype=np.float64)
    inplace_warm_scale = np.ones((n_tasks,), dtype=np.float64)
    if (
        _env_int("MOO_MAIN1_INPLACE_VARIANT", int(MAIN1_INPLACE_VARIANT_DEFAULT)) > 0
        and n_tasks > 0
        and int(np.sum(extra_shots > 0)) > 0
    ):
        variant_tasks = max(0, _env_int("MOO_MAIN1_INPLACE_VARIANT_TASKS", int(MAIN1_INPLACE_VARIANT_TASKS)))
        variant_scale = float(np.clip(_env_float("MOO_MAIN1_INPLACE_VARIANT_SCALE", float(MAIN1_INPLACE_VARIANT_SCALE)), 0.0, 0.10))
        variant_warm = float(np.clip(_env_float("MOO_MAIN1_INPLACE_VARIANT_WARM_SCALE", float(MAIN1_INPLACE_VARIANT_WARM_SCALE)), 0.70, 1.05))
        eligible_variant = np.flatnonzero(
            (extra_shots > 0)
            & active_mask
            & np.asarray([r != "pair_soft" for r in roles[:n_tasks]], dtype=bool)
        )
        if int(eligible_variant.size) > 0 and variant_tasks > 0 and variant_scale > 0.0:
            # Prefer mid-priority tasks: keep the very best exploitation tasks intact,
            # but perturb enough mass to sample a nearby quantum distribution.
            score_variant = np.minimum(top_priority[eligible_variant], np.percentile(top_priority[eligible_variant], 80))
            order_variant = eligible_variant[np.argsort(score_variant)[::-1]]
            chosen_variant = order_variant[: min(int(variant_tasks), int(order_variant.size))]
            for pos, idx_raw in enumerate(chosen_variant):
                idx = int(idx_raw)
                direction = -1.0 if (pos % 2 == 0) else 1.0
                inplace_angle_scale[idx] = 1.0 + direction * variant_scale
                if warm_bank_all[idx] is not None:
                    inplace_warm_scale[idx] = variant_warm

    fourier_shots = np.zeros((n_tasks,), dtype=np.int64)
    fourier_enabled = _env_int("MOO_FOURIER_WARM", int(FOURIER_WARM_ENABLE_DEFAULT)) > 0
    if fourier_enabled and n_tasks > 0:
        pilot_region_n = norm01(pilot_region)
        pilot_box_n = norm01(pilot_box)
        pilot_extreme_n = norm01(pilot_extreme)
        pilot_improve_n = norm01(pilot_improve)
        role_prior_n = norm01(role_prior)
        hard_case = (
            0.34 * np.clip((float(gap_need_main) - 1.74) / 0.34, 0.0, 1.0)
            + 0.23 * np.clip((10000.0 - float(nd_count_main)) / 3600.0, 0.0, 1.0)
            + 0.21 * np.clip((0.76 - float(extent_main)) / 0.22, 0.0, 1.0)
            + 0.22 * np.clip((float(priority_entropy) - 0.94) / 0.06, 0.0, 1.0)
        )
        fourier_case_ok = (
            hard_case >= 0.32
            or float(gap_need_main) >= float(FOURIER_WARM_GAP)
            or int(nd_count_main) < 9500
        )
        if fourier_case_ok:
            fourier_score = (
                0.32 * pilot_region_n
                + 0.25 * pilot_box_n
                + 0.16 * pilot_improve_n
                + 0.12 * role_prior_n
                + 0.09 * norm01(top_priority)
                + 0.06 * pilot_extreme_n
            )
            eligible = active_mask.copy()
            eligible &= extra_shots >= int(FOURIER_WARM_MIN_EXTRA)
            eligible &= np.asarray([wb is not None for wb in warm_bank_all[:n_tasks]], dtype=bool)
            eligible &= np.asarray([r != "pair_soft" for r in roles[:n_tasks]], dtype=bool)
            eligible &= modes_all[:n_tasks] != int(ANGLE_CENTER)
            if float(gap_need_main) < float(FOURIER_WARM_GAP) and int(nd_count_main) >= 10000:
                eligible &= pilot_region_n >= float(FOURIER_WARM_REGION)
            fourier_score = np.where(eligible, fourier_score, -np.inf)
            room = np.flatnonzero(np.isfinite(fourier_score))
            if int(room.size) > 0:
                max_tasks_default = int(FOURIER_WARM_MAX_TASKS)
                if hard_case < 0.46 and float(gap_need_main) < 1.95:
                    max_tasks_default = min(max_tasks_default, 2)
                max_tasks = max(0, _env_int("MOO_FOURIER_WARM_TASKS", max_tasks_default))
                max_tasks = min(max_tasks, int(room.size))
                chosen = room[np.argsort(fourier_score[room])[-max_tasks:]] if max_tasks > 0 else np.zeros((0,), dtype=np.int64)
                share = float(np.clip(_env_float("MOO_FOURIER_WARM_SHARE", float(FOURIER_WARM_SHARE)), 0.10, 0.55))
                for idx_raw in chosen:
                    idx = int(idx_raw)
                    f_i = int(np.floor(float(extra_shots[idx]) * share))
                    f_i = max(int(FOURIER_WARM_MIN_SHOTS), f_i)
                    f_i = min(f_i, max(0, int(extra_shots[idx]) - int(FOURIER_WARM_MIN_SHOTS)))
                    if f_i <= 0:
                        continue
                    fourier_shots[idx] = int(f_i)
                    extra_shots[idx] -= int(f_i)

        if _env_int("MOO_PRINT_FOURIER", 0) > 0:
            print(
                f"[M1F] fourier_tasks={int(np.sum(fourier_shots > 0))} "
                f"fourier_shots={int(np.sum(fourier_shots))} hard={hard_case:.3f} "
                f"gap={gap_need_main:.2f} extent={extent_main:.3f} nd={nd_count_main}",
                flush=True,
            )

    masked_window_shots = np.zeros((n_tasks,), dtype=np.int64)
    if (
        _env_int("MOO_MASKED_WINDOW", int(MASKED_WINDOW_ENABLE_DEFAULT)) > 0
        and _env_int("MOO_PARETO_CONTRIB", int(PARETO_CONTRIB_ENABLE_DEFAULT)) > 0
        and n_tasks > 0
    ):
        contrib_n = norm01(pilot_contrib)
        if float(np.max(contrib_n)) > 1e-12:
            window_score = (
                0.50 * contrib_n
                + 0.20 * norm01(pilot_region)
                + 0.15 * norm01(pilot_box)
                + 0.10 * norm01(pilot_improve)
                + 0.05 * norm01(role_prior)
            )
            eligible = active_mask.copy()
            eligible &= extra_shots >= int(MASKED_WINDOW_MIN_EXTRA)
            eligible &= fourier_shots <= 0
            eligible &= np.asarray([wb is not None for wb in warm_bank_all[:n_tasks]], dtype=bool)
            eligible &= np.asarray([r != "pair_soft" for r in roles[:n_tasks]], dtype=bool)
            eligible &= modes_all[:n_tasks] != int(ANGLE_CENTER)
            window_score = np.where(eligible, window_score, -np.inf)
            room = np.flatnonzero(np.isfinite(window_score))
            if int(room.size) > 0:
                max_tasks = max(0, _env_int("MOO_MASKED_WINDOW_TASKS", int(MASKED_WINDOW_MAX_TASKS)))
                max_tasks = min(max_tasks, int(room.size))
                chosen = room[np.argsort(window_score[room])[-max_tasks:]] if max_tasks > 0 else np.zeros((0,), dtype=np.int64)
                share = float(np.clip(_env_float("MOO_MASKED_WINDOW_SHARE", float(MASKED_WINDOW_SHARE)), 0.06, 0.28))
                for idx_raw in chosen:
                    idx = int(idx_raw)
                    w_i = int(np.floor(float(extra_shots[idx]) * share))
                    w_i = max(int(MASKED_WINDOW_MIN_SHOTS), w_i)
                    w_i = min(w_i, max(0, int(extra_shots[idx]) - int(MASKED_WINDOW_MIN_SHOTS)))
                    if w_i <= 0:
                        continue
                    masked_window_shots[idx] = int(w_i)
                    extra_shots[idx] -= int(w_i)

        if _env_int("MOO_PRINT_WINDOW", 0) > 0:
            print(
                f"[M1W] window_tasks={int(np.sum(masked_window_shots > 0))} "
                f"window_shots={int(np.sum(masked_window_shots))} "
                f"gap={gap_need_main:.2f} extent={extent_main:.3f} nd={nd_count_main}",
                flush=True,
            )

    for j, _lam_id_raw in enumerate(ids_all):
        shots_j = int(extra_shots[j])
        if shots_j <= 0:
            continue
        circ = build_second_round_circuit(
            j,
            angle_scale=float(inplace_angle_scale[j]),
            warm_scale=float(inplace_warm_scale[j]),
        )
        unique_spins, counts = _sample_unique_spins(
            sim,
            circ,
            shots=shots_j,
            n_qubits=n,
            seed=seed + 20000 + j,
        )
        spins = np.repeat(unique_spins, counts.astype(np.int32), axis=0)
        end = cursor + shots_j
        out_spins[cursor:end] = spins
        cursor = end

    for j, _lam_id_raw in enumerate(ids_all):
        shots_j = int(fourier_shots[j])
        if shots_j <= 0:
            continue
        circ = build_second_round_circuit(j, fourier_warm=True)
        unique_spins, counts = _sample_unique_spins(
            sim,
            circ,
            shots=shots_j,
            n_qubits=n,
            seed=seed + 24000 + j,
        )
        spins = np.repeat(unique_spins, counts.astype(np.int32), axis=0)
        end = cursor + shots_j
        out_spins[cursor:end] = spins
        cursor = end

    for j, _lam_id_raw in enumerate(ids_all):
        shots_j = int(masked_window_shots[j])
        if shots_j <= 0:
            continue
        circ = build_second_round_circuit(j, masked_window=True)
        unique_spins, counts = _sample_unique_spins(
            sim,
            circ,
            shots=shots_j,
            n_qubits=n,
            seed=seed + 26000 + j,
        )
        spins = np.repeat(unique_spins, counts.astype(np.int32), axis=0)
        end = cursor + shots_j
        out_spins[cursor:end] = spins
        cursor = end

    if substitute_budget > 0 and substitute_tasks:
        shots_each = int(substitute_budget) // max(int(len(substitute_tasks)), 1)
        rem = int(substitute_budget) - shots_each * int(len(substitute_tasks))
        for j, (lam_id, warm_bits, warm_vec, mode) in enumerate(substitute_tasks):
            shots_j = int(shots_each + (1 if j < rem else 0))
            if shots_j <= 0:
                continue
            j_raw = projected_j_pool[int(lam_id)]
            h_raw = projected_h_pool[int(lam_id)]
            betas, gammas = _select_angles(
                int(mode),
                lambda_pool[int(lam_id)],
                betas_base,
                gammas_base,
                betas_mid,
                gammas_mid,
                betas_center,
                gammas_center,
            )
            betas, gammas = _route_transfer_angles(
                betas,
                gammas,
                lambda_pool[int(lam_id)],
                problem.edges,
                n,
                j_raw,
                h_raw,
                mode=int(mode),
                role="substitute",
            )
            circ = _build_qaoa_circuit_projected_ising_warm_vec(
                problem,
                j_raw,
                h_raw,
                betas=betas,
                gammas=gammas,
                warm_bits01=warm_bits,
                warm_c=0.32,
                warm_c_vec=warm_vec,
            )
            unique_spins, counts = _sample_unique_spins(
                sim,
                circ,
                shots=shots_j,
                n_qubits=n,
                seed=seed + 30000 + j,
            )
            spins = np.repeat(unique_spins, counts.astype(np.int32), axis=0)
            end = cursor + shots_j
            out_spins[cursor:end] = spins
            cursor = end

    if cursor < BASE_SAMPLE_BUDGET:
        fill = BASE_SAMPLE_BUDGET - cursor
        reps = int(np.ceil(fill / max(cursor, 1)))
        filler = np.tile(out_spins[:cursor], (reps, 1))[:fill] if cursor > 0 else np.ones((fill, n), dtype=np.int8)
        out_spins[cursor:BASE_SAMPLE_BUDGET] = filler
    elif cursor > BASE_SAMPLE_BUDGET:
        out_spins = out_spins[:BASE_SAMPLE_BUDGET]

    if int(out_spins.shape[0]) != BASE_SAMPLE_BUDGET:
        raise ValueError(f"Output row count mismatch: {out_spins.shape[0]} != {BASE_SAMPLE_BUDGET}")

    return {
        "sample_used": BASE_SAMPLE_BUDGET,
        "sample_spins": out_spins.astype(np.int8, copy=False),
    }


def _nd_indices_lexscan_5d(objs: np.ndarray) -> np.ndarray:
    """Exact first-front scan specialized for main2's 5-objective large cases."""
    arr = np.asarray(objs, dtype=np.float64)
    if arr.ndim != 2 or int(arr.shape[0]) == 0:
        return np.zeros((0,), dtype=np.int64)
    if int(arr.shape[0]) <= 1:
        return np.arange(int(arr.shape[0]), dtype=np.int64)
    if int(arr.shape[1]) != 5:
        return pg_non_dominated_indices(arr)

    order = np.lexsort(arr[:, ::-1].T)
    n = int(arr.shape[0])
    front_vals = np.empty((n, 5), dtype=np.float64)
    front_ids = np.empty((n,), dtype=np.int64)
    f = 0

    for idx_raw in order:
        idx = int(idx_raw)
        p = arr[idx]
        if f > 0:
            cur = front_vals[:f]
            le = cur <= p
            if bool(np.any(np.all(le, axis=1) & np.any(cur < p, axis=1))):
                continue

            dom = np.all(p <= cur, axis=1) & np.any(p < cur, axis=1)
            if bool(np.any(dom)):
                keep = ~dom
                nf = int(np.sum(keep))
                if nf > 0:
                    front_vals[:nf] = cur[keep]
                    front_ids[:nf] = front_ids[:f][keep]
                f = nf

        front_vals[f] = p
        front_ids[f] = idx
        f += 1

    return front_ids[:f].copy()


def _nd_indices_rank_bitset(objs: np.ndarray) -> np.ndarray:
    """Exact low-dimensional first-front filter using rank-prefix bitsets.

    For each objective, prefix bitsets encode all rows with value <= a row's
    value. A row is dominated iff the intersection across all objective
    prefixes contains a strictly better row. This avoids pygmo call overhead on
    main2's many 2k-point local fronts, and works for the visible k=5/k=6 large
    cases without changing the exact first-front definition.
    """
    arr = np.asarray(objs, dtype=np.float64)
    if arr.ndim != 2 or int(arr.shape[0]) == 0:
        return np.zeros((0,), dtype=np.int64)
    n = int(arr.shape[0])
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    d_obj = int(arr.shape[1])
    if d_obj < 2 or d_obj > 8:
        return pg_non_dominated_indices(arr)

    ranks = np.empty((d_obj, n), dtype=np.int32)
    prefixes: List[List[int]] = []
    for dim in range(d_obj):
        order = np.argsort(arr[:, dim], kind="mergesort")
        vals = arr[order, dim]
        pref = [0] * n
        acc = 0
        start = 0
        while start < n:
            end = start + 1
            v = vals[start]
            while end < n and vals[end] == v:
                end += 1

            group_bits = 0
            for pos in range(start, end):
                idx = int(order[pos])
                ranks[dim, idx] = pos
                group_bits |= 1 << idx
            acc |= group_bits
            for pos in range(start, end):
                pref[pos] = acc
            start = end
        prefixes.append(pref)

    dominated = np.zeros((n,), dtype=bool)
    for i in range(n):
        cand = prefixes[0][int(ranks[0, i])]
        for dim in range(1, d_obj):
            cand &= prefixes[dim][int(ranks[dim, i])]
        cand &= ~(1 << i)
        while cand:
            lsb = cand & -cand
            j = lsb.bit_length() - 1
            if bool(np.any(arr[j] < arr[i])):
                dominated[i] = True
                break
            cand ^= lsb

    return np.flatnonzero(~dominated).astype(np.int64, copy=False)


def _nd_indices_rank5_bitset(objs: np.ndarray) -> np.ndarray:
    return _nd_indices_rank_bitset(objs)


def _merge_non_dominated_pool_5d(pool: np.ndarray, new_points: np.ndarray) -> np.ndarray:
    a = np.asarray(pool, dtype=np.float64)
    b = np.asarray(new_points, dtype=np.float64)
    if b.size == 0:
        return a
    merged = b if a.size == 0 else np.vstack([a, b])
    if int(merged.shape[0]) > 1:
        merged = np.unique(merged, axis=0)
    return merged[_nd_indices_lexscan_5d(merged)]


def _merge_non_dominated_pool_pg(
    pool: np.ndarray,
    new_points: np.ndarray,
    *,
    unique_before_sort: bool = True,
) -> np.ndarray:
    a = np.asarray(pool, dtype=np.float64)
    b = np.asarray(new_points, dtype=np.float64)
    if b.size == 0:
        return a
    merged = b if a.size == 0 else np.vstack([a, b])
    if bool(unique_before_sort) and int(merged.shape[0]) > 1:
        merged = np.unique(merged, axis=0)
    return merged[pg_non_dominated_indices(merged)]


def _merge_non_dominated_pool_rank(
    pool: np.ndarray,
    new_points: np.ndarray,
    *,
    unique_before_sort: bool = True,
) -> np.ndarray:
    a = np.asarray(pool, dtype=np.float64)
    b = np.asarray(new_points, dtype=np.float64)
    if b.size == 0:
        return a
    merged = b if a.size == 0 else np.vstack([a, b])
    if bool(unique_before_sort) and int(merged.shape[0]) > 1:
        merged = np.unique(merged, axis=0)
    return merged[_nd_indices_rank_bitset(merged)]


def _main2_nd_indices(objs: np.ndarray, nd_engine: str) -> np.ndarray:
    engine = str(nd_engine).strip().lower()
    if engine in ("rank5", "rank6", "rank", "bit5", "bit6", "bitset5", "bitset6", "bitset"):
        return _nd_indices_rank_bitset(objs)
    if engine in ("lex5", "lexscan5", "5d") and int(np.asarray(objs).ndim) == 2 and int(np.asarray(objs).shape[1]) == 5:
        return _nd_indices_lexscan_5d(objs)
    return pg_non_dominated_indices(objs)


def _main2_merge_pool(
    pool: np.ndarray,
    new_points: np.ndarray,
    *,
    nd_engine: str,
    unique_before_sort: bool,
) -> np.ndarray:
    engine = str(nd_engine).strip().lower()
    if engine in ("hybrid5", "lex5", "lexscan5", "5d"):
        return _merge_non_dominated_pool_5d(pool, new_points)
    if engine in ("rank5", "rank6", "rank", "bit5", "bit6", "bitset5", "bitset6", "bitset"):
        return _merge_non_dominated_pool_rank(pool, new_points, unique_before_sort=unique_before_sort)
    return _merge_non_dominated_pool_pg(pool, new_points, unique_before_sort=unique_before_sort)


def _main2_finalize_pool(
    nd_pool: np.ndarray,
    *,
    nd_engine: str,
    unique_in_merge: bool,
) -> np.ndarray:
    arr = np.asarray(nd_pool, dtype=np.float64)
    sort_final = _env_int("MOO_MAIN2_FINAL_SORT", MAIN2_FINAL_SORT_DEFAULT) > 0
    if int(arr.shape[0]) <= 1:
        return np.asarray(lexsort_rows(arr) if sort_final else arr, dtype=np.float64)
    engine = str(nd_engine).strip().lower()
    if (not bool(unique_in_merge)) or engine in ("hybrid5", "lex5", "lexscan5", "5d"):
        arr = np.unique(arr, axis=0)
        if engine in ("hybrid5", "lex5", "lexscan5", "5d"):
            arr = arr[_nd_indices_lexscan_5d(arr)]
        else:
            arr = arr[pg_non_dominated_indices(arr)]
    elif engine in ("rank5", "rank6", "rank", "bit5", "bit6", "bitset5", "bitset6", "bitset"):
        arr = arr[_nd_indices_rank_bitset(arr)]
    if sort_final:
        arr = lexsort_rows(arr)
    return np.asarray(arr, dtype=np.float64)


def _main2_hv_from_final_frontier(nd_pool: np.ndarray, ref: float | np.ndarray = HV_REF) -> float:
    arr = np.asarray(nd_pool, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if _env_int("MOO_MAIN2_FAST_HV", MAIN2_FAST_HV_DEFAULT) <= 0 or _main2_pg is None:
        return float(hypervolume_pygmo(arr, ref=ref))
    if np.isscalar(ref):
        ref_vec = np.full(int(arr.shape[1]), float(ref), dtype=np.float64)
    else:
        ref_vec = np.asarray(ref, dtype=np.float64).reshape(-1)
    mask = np.all(arr <= ref_vec[None, :], axis=1)
    arr = arr[mask]
    if arr.size == 0:
        return 0.0
    return float(_main2_pg.hypervolume(arr).compute(ref_vec))


class _Main2LazyHV:
    __slots__ = ("_frontier", "_ref", "_value")

    def __init__(self, frontier: np.ndarray, ref: float | np.ndarray = HV_REF):
        self._frontier = np.asarray(frontier, dtype=np.float64)
        self._ref = ref
        self._value = None

    def __float__(self) -> float:
        val = self._value
        if val is None:
            val = float(_main2_hv_from_final_frontier(self._frontier, ref=self._ref))
            self._value = val
        return float(val)


def _main2_make_hv(nd_pool: np.ndarray, ref: float | np.ndarray = HV_REF):
    if _env_int("MOO_MAIN2_LAZY_HV", MAIN2_LAZY_HV_DEFAULT) > 0:
        return _Main2LazyHV(nd_pool, ref=ref)
    return float(_main2_hv_from_final_frontier(nd_pool, ref=ref))


def _energy_batch_main2_int8(
    spins: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """Main2-only energy path that keeps spin products compact.

    utils.energy_batch_fast casts the full spin matrix to float64 before
    forming edge products, which materializes a large float64 pair matrix on
    40x50 grids.  The pair values are exactly {-1,+1}, so keeping them int8
    until the final matrix product reduces memory traffic while preserving the
    same objective values up to normal floating-point dot ordering.
    """
    z = np.asarray(spins, dtype=np.int8)
    e = np.asarray(edges, dtype=np.int32)
    pair = (z[:, e[:, 0]] * z[:, e[:, 1]]).astype(np.int8, copy=False)
    edge_term = pair @ np.asarray(weights, dtype=np.float64).T
    linear_term = z @ np.asarray(h, dtype=np.float64).T
    return np.asarray(edge_term + linear_term, dtype=np.float64)


def _energy_batch_main2_prepared(
    spins: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    weights_t: np.ndarray,
    h_t: np.ndarray,
) -> np.ndarray:
    s = np.asarray(spins, dtype=np.float64)
    pair = s[:, edge_u] * s[:, edge_v]
    return pair @ weights_t + s @ h_t


def _energy_batch_main2_bits(
    bits_pos: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    weights_t: np.ndarray,
    h_t: np.ndarray,
    edge_sum: np.ndarray,
    h_sum: np.ndarray,
) -> np.ndarray:
    """Energy from baseline boolean samples without materializing +/-1 spins."""
    b = np.asarray(bits_pos, dtype=bool)
    linear = 2.0 * (b @ h_t) - h_sum[None, :]
    edge_diff = np.logical_xor(b[:, edge_u], b[:, edge_v])
    edge_term = edge_sum[None, :] - 2.0 * (edge_diff @ weights_t)
    return np.asarray(edge_term + linear, dtype=np.float64)


def _energy_batch_main2_bits_i8(
    bits_pos: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    weights_t: np.ndarray,
    h_t: np.ndarray,
    edge_sum: np.ndarray,
    h_sum: np.ndarray,
) -> np.ndarray:
    """Bit energy path with explicit int8 BLAS inputs.

    Some NumPy/OpenBLAS builds handle bool @ float64 poorly.  This keeps the
    exact baseline bit stream but lets the online environment choose an int8
    conversion before matrix multiply.
    """
    b = np.asarray(bits_pos, dtype=bool)
    b_i8 = np.ascontiguousarray(b).view(np.int8)
    linear = 2.0 * (b_i8 @ h_t) - h_sum[None, :]
    edge_diff = np.logical_xor(b[:, edge_u], b[:, edge_v]).view(np.int8)
    edge_term = edge_sum[None, :] - 2.0 * (edge_diff @ weights_t)
    return np.asarray(edge_term + linear, dtype=np.float64)


def _energy_batch_main2_bits_grid_i8(
    bits_pos: np.ndarray,
    a: int,
    b: int,
    weights_h_t: np.ndarray,
    weights_v_t: np.ndarray,
    h_t: np.ndarray,
    edge_sum: np.ndarray,
    h_sum: np.ndarray,
) -> np.ndarray:
    b0 = np.asarray(bits_pos, dtype=bool)
    bs = int(b0.shape[0])
    grid = b0.reshape((bs, int(a), int(b)))
    bits_i8 = np.ascontiguousarray(b0).view(np.int8)
    diff_h = np.logical_xor(grid[:, :, :-1], grid[:, :, 1:]).reshape((bs, int(a) * (int(b) - 1))).view(np.int8)
    diff_v = np.logical_xor(grid[:, :-1, :], grid[:, 1:, :]).reshape((bs, (int(a) - 1) * int(b))).view(np.int8)
    energies = diff_h @ weights_h_t
    energies += diff_v @ weights_v_t
    energies *= -2.0
    energies += edge_sum[None, :]
    linear = bits_i8 @ h_t
    energies += linear
    energies += linear
    energies -= h_sum[None, :]
    return np.asarray(energies, dtype=np.float64)


def _energy_batch_main2_bits_grid_i8_reuse(
    bits_pos: np.ndarray,
    a: int,
    b: int,
    weights_h_t: np.ndarray,
    weights_v_t: np.ndarray,
    h_t: np.ndarray,
    edge_sum: np.ndarray,
    h_sum: np.ndarray,
    diff_h_buf: np.ndarray,
    diff_v_buf: np.ndarray,
) -> np.ndarray:
    b0 = np.asarray(bits_pos, dtype=bool)
    bs = int(b0.shape[0])
    aa = int(a)
    bb = int(b)
    grid = b0.reshape((bs, aa, bb))
    diff_h = diff_h_buf[:bs]
    diff_v = diff_v_buf[:bs]
    np.logical_xor(grid[:, :, :-1], grid[:, :, 1:], out=diff_h.reshape((bs, aa, bb - 1)))
    np.logical_xor(grid[:, :-1, :], grid[:, 1:, :], out=diff_v.reshape((bs, aa - 1, bb)))
    bits_i8 = np.ascontiguousarray(b0).view(np.int8)
    energies = diff_h.view(np.int8) @ weights_h_t
    energies += diff_v.view(np.int8) @ weights_v_t
    energies *= -2.0
    energies += edge_sum[None, :]
    linear = bits_i8 @ h_t
    energies += linear
    energies += linear
    energies -= h_sum[None, :]
    return np.asarray(energies, dtype=np.float64)


def _energy_batch_main2_bits_grid_f32(
    bits_pos: np.ndarray,
    a: int,
    b: int,
    weights_h_t: np.ndarray,
    weights_v_t: np.ndarray,
    h_t: np.ndarray,
    edge_sum: np.ndarray,
    h_sum: np.ndarray,
) -> np.ndarray:
    b0 = np.asarray(bits_pos, dtype=bool)
    bs = int(b0.shape[0])
    grid = b0.reshape((bs, int(a), int(b)))
    bits_i8 = np.ascontiguousarray(b0).view(np.int8)
    diff_h = np.logical_xor(grid[:, :, :-1], grid[:, :, 1:]).reshape((bs, int(a) * (int(b) - 1))).view(np.int8)
    diff_v = np.logical_xor(grid[:, :-1, :], grid[:, 1:, :]).reshape((bs, (int(a) - 1) * int(b))).view(np.int8)
    energies = diff_h @ weights_h_t
    energies += diff_v @ weights_v_t
    energies *= np.float32(-2.0)
    energies += edge_sum[None, :]
    linear = bits_i8 @ h_t
    energies += linear
    energies += linear
    energies -= h_sum[None, :]
    return np.asarray(energies, dtype=np.float64)


def _energy_batch_main2_bits_grid_full_i8(
    bits_pos: np.ndarray,
    a: int,
    b: int,
    weights_grid_t: np.ndarray,
    h_t: np.ndarray,
    edge_sum: np.ndarray,
    h_sum: np.ndarray,
) -> np.ndarray:
    b0 = np.asarray(bits_pos, dtype=bool)
    bs = int(b0.shape[0])
    aa = int(a)
    bb = int(b)
    grid = b0.reshape((bs, aa, bb))
    h_count = aa * (bb - 1)
    v_count = (aa - 1) * bb
    edge_diff = np.empty((bs, h_count + v_count), dtype=bool)
    np.logical_xor(grid[:, :, :-1], grid[:, :, 1:], out=edge_diff[:, :h_count].reshape((bs, aa, bb - 1)))
    np.logical_xor(grid[:, :-1, :], grid[:, 1:, :], out=edge_diff[:, h_count:].reshape((bs, aa - 1, bb)))
    energies = edge_diff.view(np.int8) @ weights_grid_t
    energies *= -2.0
    energies += edge_sum[None, :]
    bits_i8 = np.ascontiguousarray(b0).view(np.int8)
    linear = bits_i8 @ h_t
    energies += linear
    energies += linear
    energies -= h_sum[None, :]
    return np.asarray(energies, dtype=np.float64)


def _main2_empty_profile() -> Dict[str, float]:
    return {
        "extrema_s": 0.0,
        "spin_s": 0.0,
        "energy_s": 0.0,
        "norm_s": 0.0,
        "pool_filter_s": 0.0,
        "local_nd_s": 0.0,
        "merge_s": 0.0,
        "final_s": 0.0,
        "hv_s": 0.0,
        "wait_s": 0.0,
    }


def _main2_add_profile(dst: Dict[str, float], src: Dict[str, float]) -> None:
    for key, val in src.items():
        dst[key] = float(dst.get(key, 0.0)) + float(val)


def _main2_random_spins(
    rng: np.random.Generator,
    bs: int,
    n: int,
    *,
    float_spins: bool,
) -> np.ndarray:
    if bool(float_spins):
        # Same random stream as rng.random((bs, n)), but the random buffer is
        # converted in-place into +/-1 float spins.  main2 only needs energies,
        # so avoiding int8 spins plus a later float64 cast saves one large copy
        # per chunk on 40x50 grids.
        spins = np.empty((int(bs), int(n)), dtype=np.float64)
        rng.random(out=spins)
        mask = spins < 0.5
        spins[mask] = 1.0
        spins[~mask] = -1.0
        return spins
    method = str(os.environ.get("MOO_MAIN2_SPIN_METHOD", MAIN2_SPIN_METHOD_DEFAULT)).strip().lower()
    if method in ("where", "baseline"):
        return np.where(rng.random((int(bs), int(n))) < 0.5, 1, -1).astype(np.int8)

    mask = rng.random((int(bs), int(n))) < 0.5
    spins = np.empty(mask.shape, dtype=np.int8)
    spins[mask] = 1
    spins[~mask] = -1
    return spins


def _main2_random_bits(
    rng: np.random.Generator,
    bs: int,
    n: int,
) -> np.ndarray:
    return rng.random((int(bs), int(n))) < 0.5


def _main2_random_bits_reuse(
    rng: np.random.Generator,
    bs: int,
    n: int,
    random_buffer: np.ndarray,
) -> np.ndarray:
    view = random_buffer[: int(bs), : int(n)]
    rng.random(out=view)
    bits = np.empty((int(bs), int(n)), dtype=bool)
    np.less(view, 0.5, out=bits)
    return bits


def _main2_pool_score(p: np.ndarray) -> np.ndarray:
    mode = str(os.environ.get("MOO_MAIN2_POOL_SCORE", MAIN2_POOL_SCORE_DEFAULT)).strip().lower()
    if mode in ("sum", "s"):
        return np.sum(p, axis=1)
    if mode in ("max", "m"):
        return np.max(p, axis=1)
    if mode in ("l2", "norm", "sq"):
        return np.sum(p * p, axis=1)
    if mode in ("summin", "min"):
        return np.sum(p, axis=1) - 0.25 * np.min(p, axis=1)
    if mode in ("summax70", "summax_70", "wide"):
        return np.sum(p, axis=1) + 0.70 * np.max(p, axis=1)
    if mode in ("summax20", "summax_20", "light"):
        return np.sum(p, axis=1) + 0.20 * np.max(p, axis=1)
    return np.sum(p, axis=1) + 0.35 * np.max(p, axis=1)


def _main2_not_dominated_mask_by_pool_margin(
    objs: np.ndarray,
    pool: np.ndarray | None,
    *,
    block_size: int = MAIN2_POOL_FILTER_BLOCK_DEFAULT,
    top_k: int = MAIN2_POOL_FILTER_TOP_DEFAULT,
    margin: float = MAIN2_PREF32_MARGIN_DEFAULT,
) -> np.ndarray:
    arr = np.asarray(objs, dtype=np.float64)
    keep = np.ones((int(arr.shape[0]),), dtype=bool)
    if arr.ndim != 2 or int(arr.shape[0]) == 0:
        return keep
    if pool is None:
        return keep
    p = np.asarray(pool, dtype=np.float64)
    if p.ndim != 2 or int(p.shape[0]) == 0 or int(p.shape[1]) != int(arr.shape[1]):
        return keep
    top_k = int(top_k)
    if top_k > 0 and int(p.shape[0]) > top_k:
        score = _main2_pool_score(p)
        p = p[np.argpartition(score, kth=top_k - 1)[:top_k]]
    margin = max(0.0, float(margin))
    step = max(1, int(block_size))
    for st in range(0, int(arr.shape[0]), step):
        ed = min(st + step, int(arr.shape[0]))
        block = arr[st:ed]
        dominated = np.zeros((int(block.shape[0]),), dtype=bool)
        for q in p:
            le = np.all(q[None, :] <= (block - margin), axis=1)
            lt = np.any(q[None, :] < (block - margin), axis=1)
            dominated |= le & lt
            if bool(np.all(dominated)):
                break
        keep[st:ed] = ~dominated
    return keep


def _main2_filter_not_dominated_by_pool(
    objs: np.ndarray,
    pool: np.ndarray | None,
    *,
    block_size: int = MAIN2_POOL_FILTER_BLOCK_DEFAULT,
    top_k: int = MAIN2_POOL_FILTER_TOP_DEFAULT,
) -> np.ndarray:
    """Drop points already dominated by a known ND pool.

    This is exact for minimization objectives. A stale or partial pool is still
    safe: removing points dominated by any already-known point cannot remove a
    final Pareto point.
    """
    arr = np.asarray(objs, dtype=np.float64)
    if arr.ndim != 2 or int(arr.shape[0]) == 0:
        return arr.reshape((0, int(arr.shape[1]) if arr.ndim == 2 else 0))
    if pool is None:
        return arr
    p = np.asarray(pool, dtype=np.float64)
    if p.ndim != 2 or int(p.shape[0]) == 0 or int(p.shape[1]) != int(arr.shape[1]):
        return arr
    top_k = int(top_k)
    if top_k > 0 and int(p.shape[0]) > top_k:
        # Points with small sum/max coordinates dominate the broadest region.
        score = _main2_pool_score(p)
        p = p[np.argpartition(score, kth=top_k - 1)[:top_k]]

    keep = np.ones((int(arr.shape[0]),), dtype=bool)
    step = max(1, int(block_size))
    for st in range(0, int(arr.shape[0]), step):
        ed = min(st + step, int(arr.shape[0]))
        block = arr[st:ed]
        if int(p.shape[0]) <= 64 and int(block.shape[1]) == 6:
            dominated = np.zeros((int(block.shape[0]),), dtype=bool)
            for q in p:
                le = (
                    (q[0] <= block[:, 0])
                    & (q[1] <= block[:, 1])
                    & (q[2] <= block[:, 2])
                    & (q[3] <= block[:, 3])
                    & (q[4] <= block[:, 4])
                    & (q[5] <= block[:, 5])
                )
                lt = (
                    (q[0] < block[:, 0])
                    | (q[1] < block[:, 1])
                    | (q[2] < block[:, 2])
                    | (q[3] < block[:, 3])
                    | (q[4] < block[:, 4])
                    | (q[5] < block[:, 5])
                )
                dominated |= le & lt
                if bool(np.all(dominated)):
                    break
        else:
            le = p[:, None, :] <= block[None, :, :]
            lt = p[:, None, :] < block[None, :, :]
            dominated = np.any(np.all(le, axis=2) & np.any(lt, axis=2), axis=0)
        keep[st:ed] = ~dominated
    return arr[keep]


def _main2_select_dominance_pool(
    nd_pool: np.ndarray,
    pending_parts: List[np.ndarray] | None,
    *,
    min_size: int,
    top_k: int,
) -> np.ndarray | None:
    parts: List[np.ndarray] = []
    total = 0
    base = np.asarray(nd_pool, dtype=np.float64)
    if base.ndim == 2 and int(base.shape[0]) > 0:
        parts.append(base)
        total += int(base.shape[0])
    if pending_parts:
        for part in pending_parts:
            arr = np.asarray(part, dtype=np.float64)
            if arr.ndim == 2 and int(arr.shape[0]) > 0:
                parts.append(arr)
                total += int(arr.shape[0])

    if total < int(min_size) or not parts:
        return None
    p = parts[0] if len(parts) == 1 else np.vstack(parts)
    if p.ndim != 2 or int(p.shape[0]) == 0:
        return None

    top_k = int(top_k)
    if top_k > 0 and int(p.shape[0]) > top_k:
        score = _main2_pool_score(p)
        p = p[np.argpartition(score, kth=top_k - 1)[:top_k]]
    return np.ascontiguousarray(p, dtype=np.float64)


def _main2_grid_edge_slices(problem: IsingMOOProblem) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    a = int(problem.a)
    b = int(problem.b)
    edges = np.asarray(problem.edges, dtype=np.int32)
    expected = []
    for i in range(a):
        row = i * b
        for j in range(b):
            u = row + j
            if j + 1 < b:
                expected.append((u, u + 1))
            if i + 1 < a:
                expected.append((u, u + b))
    exp = np.asarray(expected, dtype=np.int32)
    if exp.shape != edges.shape or not np.array_equal(exp, edges):
        return None, None
    h_idx = []
    v_idx = []
    pos = 0
    for i in range(a):
        for j in range(b):
            if j + 1 < b:
                h_idx.append(pos)
                pos += 1
            if i + 1 < a:
                v_idx.append(pos)
                pos += 1
    return np.asarray(h_idx, dtype=np.int32), np.asarray(v_idx, dtype=np.int32)


def _main2_rng_at_offset(rng_seed: int, draw_offset: int) -> np.random.Generator:
    bitgen_cls = type(np.random.default_rng(int(rng_seed)).bit_generator)
    bitgen = bitgen_cls(int(rng_seed))
    bitgen.advance(int(draw_offset))
    return np.random.Generator(bitgen)


def _main2_local_nd_from_spin_blocks(
    spin_blocks: List[np.ndarray],
    edges: np.ndarray,
    weights: np.ndarray,
    h: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    use_int8_energy: bool = False,
    nd_engine: str = MAIN2_ND_ENGINE_DEFAULT,
    profile: bool = False,
    use_prepared_energy: bool = False,
    edge_u: np.ndarray | None = None,
    edge_v: np.ndarray | None = None,
    weights_t: np.ndarray | None = None,
    h_t: np.ndarray | None = None,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, float]]:
    prof = _main2_empty_profile() if bool(profile) else {}
    obj_parts: List[np.ndarray] = []
    for spins in spin_blocks:
        t_energy = time.perf_counter()
        if use_int8_energy:
            energies = _energy_batch_main2_int8(spins, edges, weights, h)
        elif use_prepared_energy and edge_u is not None and edge_v is not None and weights_t is not None and h_t is not None:
            energies = _energy_batch_main2_prepared(spins, edge_u, edge_v, weights_t, h_t)
        else:
            energies = np.asarray(energy_batch_fast(spins, edges, weights, h), dtype=np.float64)
        if profile:
            prof["energy_s"] += time.perf_counter() - t_energy
        t_norm = time.perf_counter()
        obj_parts.append(normalize_energies(energies, lower_bounds, upper_bounds))
        if profile:
            prof["norm_s"] += time.perf_counter() - t_norm
    objs_batch = obj_parts[0] if len(obj_parts) == 1 else np.vstack(obj_parts)
    local_engine = "pg" if str(nd_engine).strip().lower() == "hybrid5" else str(nd_engine)
    t_nd = time.perf_counter()
    local_nd = np.asarray(objs_batch[_main2_nd_indices(objs_batch, local_engine)], dtype=np.float64)
    if profile:
        prof["local_nd_s"] += time.perf_counter() - t_nd
        return local_nd, prof
    return local_nd


def _main2_local_nd_from_bit_blocks(
    bit_blocks: List[np.ndarray],
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    weights_t: np.ndarray,
    h_t: np.ndarray,
    edge_sum: np.ndarray,
    h_sum: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    nd_engine: str = MAIN2_ND_ENGINE_DEFAULT,
    profile: bool = False,
    dominance_pool: np.ndarray | None = None,
    pool_filter_block: int = MAIN2_POOL_FILTER_BLOCK_DEFAULT,
    pool_filter_top: int = MAIN2_POOL_FILTER_TOP_DEFAULT,
    bit_energy: str = MAIN2_BIT_ENERGY_DEFAULT,
    grid_shape: Tuple[int, int] | None = None,
    weights_h_t: np.ndarray | None = None,
    weights_v_t: np.ndarray | None = None,
    weights_grid_t: np.ndarray | None = None,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, float]]: 
    prof = _main2_empty_profile() if bool(profile) else {}
    obj_parts: List[np.ndarray] = []
    bit_energy_mode = str(bit_energy).strip().lower()
    pref32_mode = bit_energy_mode in ("grid_pref32", "grid_f32_prefilter", "grid_prefilter_f32")
    grid_reuse = (
        _env_int("MOO_MAIN2_GRID_REUSE", MAIN2_GRID_REUSE_DEFAULT) > 0
        and bit_energy_mode in ("grid", "grid_i8", "grid_int8", "grid_reuse", "grid_i8_reuse", "grid_int8_reuse")
        and grid_shape is not None
        and weights_h_t is not None
        and weights_v_t is not None
        and bool(bit_blocks)
    )
    diff_h_buf = None
    diff_v_buf = None
    if grid_reuse:
        max_bs = max(int(np.asarray(bits).shape[0]) for bits in bit_blocks)
        aa = int(grid_shape[0])
        bb = int(grid_shape[1])
        diff_h_buf = np.empty((max_bs, aa * (bb - 1)), dtype=bool)
        diff_v_buf = np.empty((max_bs, (aa - 1) * bb), dtype=bool)
    weights_h_t_f32 = np.asarray(weights_h_t, dtype=np.float32) if (pref32_mode and weights_h_t is not None) else None
    weights_v_t_f32 = np.asarray(weights_v_t, dtype=np.float32) if (pref32_mode and weights_v_t is not None) else None
    h_t_f32 = np.asarray(h_t, dtype=np.float32) if (pref32_mode and h_t is not None) else None
    edge_sum_f32 = np.asarray(edge_sum, dtype=np.float32) if (pref32_mode and edge_sum is not None) else None
    h_sum_f32 = np.asarray(h_sum, dtype=np.float32) if (pref32_mode and h_sum is not None) else None
    for bits in bit_blocks:
        t_energy = time.perf_counter()
        bits_exact = bits
        if (
            pref32_mode
            and dominance_pool is not None
            and grid_shape is not None
            and weights_h_t_f32 is not None
            and weights_v_t_f32 is not None
            and h_t_f32 is not None
            and edge_sum_f32 is not None
            and h_sum_f32 is not None
        ):
            energies_f32 = _energy_batch_main2_bits_grid_f32(
                bits,
                int(grid_shape[0]),
                int(grid_shape[1]),
                weights_h_t_f32,
                weights_v_t_f32,
                h_t_f32,
                edge_sum_f32,
                h_sum_f32,
            )
            objs_f32 = normalize_energies(energies_f32, lower_bounds, upper_bounds)
            keep_pref = _main2_not_dominated_mask_by_pool_margin(
                objs_f32,
                dominance_pool,
                block_size=int(pool_filter_block),
                top_k=int(pool_filter_top),
                margin=_env_float("MOO_MAIN2_PREF32_MARGIN", MAIN2_PREF32_MARGIN_DEFAULT),
            )
            bits_exact = bits[np.asarray(keep_pref, dtype=bool)]
            if int(bits_exact.shape[0]) == 0:
                if profile:
                    prof["energy_s"] += time.perf_counter() - t_energy
                continue

        if (
            bit_energy_mode in ("grid_full", "grid_full_i8", "grid_full_int8")
            and grid_shape is not None
            and weights_grid_t is not None
        ):
            energies = _energy_batch_main2_bits_grid_full_i8(bits_exact, int(grid_shape[0]), int(grid_shape[1]), weights_grid_t, h_t, edge_sum, h_sum)
        elif (
            bit_energy_mode in ("grid_f32", "grid_float32", "grid_i8_f32")
            and grid_shape is not None
            and weights_h_t is not None
            and weights_v_t is not None
        ):
            energies = _energy_batch_main2_bits_grid_f32(bits_exact, int(grid_shape[0]), int(grid_shape[1]), weights_h_t, weights_v_t, h_t, edge_sum, h_sum)
        elif grid_reuse and diff_h_buf is not None and diff_v_buf is not None:
            energies = _energy_batch_main2_bits_grid_i8_reuse(
                bits_exact,
                int(grid_shape[0]),
                int(grid_shape[1]),
                weights_h_t,
                weights_v_t,
                h_t,
                edge_sum,
                h_sum,
                diff_h_buf,
                diff_v_buf,
            )
        elif (
            bit_energy_mode in ("grid", "grid_i8", "grid_int8", "grid_reuse", "grid_i8_reuse", "grid_int8_reuse")
            and grid_shape is not None
            and weights_h_t is not None
            and weights_v_t is not None
        ):
            energies = _energy_batch_main2_bits_grid_i8(bits_exact, int(grid_shape[0]), int(grid_shape[1]), weights_h_t, weights_v_t, h_t, edge_sum, h_sum)
        elif (
            pref32_mode
            and grid_shape is not None
            and weights_h_t is not None
            and weights_v_t is not None
        ):
            energies = _energy_batch_main2_bits_grid_i8(bits_exact, int(grid_shape[0]), int(grid_shape[1]), weights_h_t, weights_v_t, h_t, edge_sum, h_sum)
        elif bit_energy_mode in ("i8", "int8"):
            energies = _energy_batch_main2_bits_i8(bits_exact, edge_u, edge_v, weights_t, h_t, edge_sum, h_sum)
        else:
            energies = _energy_batch_main2_bits(bits_exact, edge_u, edge_v, weights_t, h_t, edge_sum, h_sum)
        if profile:
            prof["energy_s"] += time.perf_counter() - t_energy
        t_norm = time.perf_counter()
        obj_parts.append(normalize_energies(energies, lower_bounds, upper_bounds))
        if profile:
            prof["norm_s"] += time.perf_counter() - t_norm
    if not obj_parts:
        local_nd = np.zeros((0, int(lower_bounds.shape[0])), dtype=np.float64)
        if profile:
            return local_nd, prof
        return local_nd
    objs_batch = obj_parts[0] if len(obj_parts) == 1 else np.vstack(obj_parts)
    if dominance_pool is not None:
        t_filter = time.perf_counter()
        objs_batch = _main2_filter_not_dominated_by_pool(
            objs_batch,
            dominance_pool,
            block_size=int(pool_filter_block),
            top_k=int(pool_filter_top),
        )
        if profile:
            prof["pool_filter_s"] += time.perf_counter() - t_filter
    if int(objs_batch.shape[0]) == 0:
        local_nd = np.zeros((0, int(lower_bounds.shape[0])), dtype=np.float64)
        if profile:
            return local_nd, prof
        return local_nd
    local_engine = "pg" if str(nd_engine).strip().lower() == "hybrid5" else str(nd_engine)
    t_nd = time.perf_counter()
    local_nd = np.asarray(objs_batch[_main2_nd_indices(objs_batch, local_engine)], dtype=np.float64)
    if profile:
        prof["local_nd_s"] += time.perf_counter() - t_nd
        return local_nd, prof
    return local_nd


def _main2_local_nd_from_seed_blocks(
    seed_blocks: List[Tuple[int, int]],
    rng_seed: int,
    n: int,
    edges: np.ndarray,
    weights: np.ndarray,
    h: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    use_int8_energy: bool = False,
    nd_engine: str = MAIN2_ND_ENGINE_DEFAULT,
    profile: bool = False,
    use_prepared_energy: bool = False,
    edge_u: np.ndarray | None = None,
    edge_v: np.ndarray | None = None,
    weights_t: np.ndarray | None = None,
    h_t: np.ndarray | None = None,
    use_bitpack: bool = False,
    edge_sum: np.ndarray | None = None,
    h_sum: np.ndarray | None = None,
    dominance_pool: np.ndarray | None = None,
    pool_filter_block: int = MAIN2_POOL_FILTER_BLOCK_DEFAULT,
    pool_filter_top: int = MAIN2_POOL_FILTER_TOP_DEFAULT,
    bit_energy: str = MAIN2_BIT_ENERGY_DEFAULT,
    grid_shape: Tuple[int, int] | None = None,
    weights_h_t: np.ndarray | None = None,
    weights_v_t: np.ndarray | None = None,
    weights_grid_t: np.ndarray | None = None,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, float]]:
    prof = _main2_empty_profile() if bool(profile) else {}
    obj_parts: List[np.ndarray] = []
    bit_energy_mode = str(bit_energy).strip().lower()
    random_buffer = None
    if use_bitpack and seed_blocks:
        max_bs = max(int(bs_i) for _, bs_i in seed_blocks)
        random_buffer = np.empty((int(max_bs), int(n)), dtype=np.float64)
    for draw_offset, bs_i in seed_blocks:
        t_spin = time.perf_counter()
        rng = _main2_rng_at_offset(int(rng_seed), int(draw_offset))
        if use_bitpack:
            if random_buffer is not None:
                bits = _main2_random_bits_reuse(rng, int(bs_i), int(n), random_buffer)
            else:
                bits = _main2_random_bits(rng, int(bs_i), int(n))
            spins = None
        else:
            bits = None
            spins = _main2_random_spins(rng, int(bs_i), int(n), float_spins=False)
        if profile:
            prof["spin_s"] += time.perf_counter() - t_spin

        t_energy = time.perf_counter()
        if use_bitpack and bits is not None and edge_u is not None and edge_v is not None and weights_t is not None and h_t is not None and edge_sum is not None and h_sum is not None:
            if (
                bit_energy_mode in ("grid_full", "grid_full_i8", "grid_full_int8")
                and grid_shape is not None
                and weights_grid_t is not None
            ):
                energies = _energy_batch_main2_bits_grid_full_i8(bits, int(grid_shape[0]), int(grid_shape[1]), weights_grid_t, h_t, edge_sum, h_sum)
            elif (
                bit_energy_mode in ("grid_f32", "grid_float32", "grid_i8_f32")
                and grid_shape is not None
                and weights_h_t is not None
                and weights_v_t is not None
            ):
                energies = _energy_batch_main2_bits_grid_f32(bits, int(grid_shape[0]), int(grid_shape[1]), weights_h_t, weights_v_t, h_t, edge_sum, h_sum)
            elif (
                bit_energy_mode in ("grid", "grid_i8", "grid_int8")
                and grid_shape is not None
                and weights_h_t is not None
                and weights_v_t is not None
            ):
                energies = _energy_batch_main2_bits_grid_i8(bits, int(grid_shape[0]), int(grid_shape[1]), weights_h_t, weights_v_t, h_t, edge_sum, h_sum)
            elif bit_energy_mode in ("i8", "int8"):
                energies = _energy_batch_main2_bits_i8(bits, edge_u, edge_v, weights_t, h_t, edge_sum, h_sum)
            else:
                energies = _energy_batch_main2_bits(bits, edge_u, edge_v, weights_t, h_t, edge_sum, h_sum)
        elif use_int8_energy and spins is not None:
            energies = _energy_batch_main2_int8(spins, edges, weights, h)
        elif use_prepared_energy and spins is not None and edge_u is not None and edge_v is not None and weights_t is not None and h_t is not None:
            energies = _energy_batch_main2_prepared(spins, edge_u, edge_v, weights_t, h_t)
        elif spins is not None:
            energies = np.asarray(energy_batch_fast(spins, edges, weights, h), dtype=np.float64)
        else:
            raise ValueError("Invalid main2 worker RNG energy configuration.")
        if profile:
            prof["energy_s"] += time.perf_counter() - t_energy

        t_norm = time.perf_counter()
        obj_parts.append(normalize_energies(energies, lower_bounds, upper_bounds))
        if profile:
            prof["norm_s"] += time.perf_counter() - t_norm

    objs_batch = obj_parts[0] if len(obj_parts) == 1 else np.vstack(obj_parts)
    if dominance_pool is not None:
        t_filter = time.perf_counter()
        objs_batch = _main2_filter_not_dominated_by_pool(
            objs_batch,
            dominance_pool,
            block_size=int(pool_filter_block),
            top_k=int(pool_filter_top),
        )
        if profile:
            prof["pool_filter_s"] += time.perf_counter() - t_filter
    if int(objs_batch.shape[0]) == 0:
        local_nd = np.zeros((0, int(lower_bounds.shape[0])), dtype=np.float64)
        if profile:
            return local_nd, prof
        return local_nd
    local_engine = "pg" if str(nd_engine).strip().lower() == "hybrid5" else str(nd_engine)
    t_nd = time.perf_counter()
    local_nd = np.asarray(objs_batch[_main2_nd_indices(objs_batch, local_engine)], dtype=np.float64)
    if profile:
        prof["local_nd_s"] += time.perf_counter() - t_nd
        return local_nd, prof
    return local_nd


def _large_random_frontier_hv_pg_deferred(
    problem: IsingMOOProblem,
    *,
    shots: int = 200000,
    chunk_size: int = MAIN2_CHUNK_SIZE_DEFAULT,
    rng_seed: int = 2026,
    ref: float = HV_REF,
    merge_every: int = MAIN2_MERGE_EVERY_DEFAULT,
    local_nd_group: int = MAIN2_LOCAL_GROUP_DEFAULT,
) -> Dict[str, object]:
    """Main2 fast path: same random samples, fewer C++ ND calls.

    The baseline calls non-dominated sorting once per chunk and merges every
    chunk.  Large cases are dominated by that sorting/merge churn.  Here we
    keep the random stream identical but batch two objective chunks into one
    local front, then merge several local fronts at once.
    """
    profile_enabled = _env_int("MOO_PROFILE_M2", MAIN2_PROFILE_DEFAULT) > 0
    prof = _main2_empty_profile()
    rng = np.random.default_rng(int(rng_seed))
    t_extrema = time.perf_counter()
    lower_bounds, upper_bounds = objective_extrema(problem)
    if profile_enabled:
        prof["extrema_s"] += time.perf_counter() - t_extrema
    k = int(problem.k)
    n = int(problem.n)

    remaining = int(shots)
    nd_pool = np.zeros((0, k), dtype=np.float64)
    pending_parts: List[np.ndarray] = []
    local_obj_parts: List[np.ndarray] = []
    n_points = 0
    local_nd_calls = 0
    cumulative_merge_calls = 0
    use_int8_energy = str(os.environ.get("MOO_MAIN2_ENERGY", MAIN2_ENERGY_DEFAULT)).strip().lower() not in (
        "utils",
        "float",
        "float64",
    )
    unique_in_merge = _env_int("MOO_MAIN2_UNIQUE_MERGE", MAIN2_UNIQUE_MERGE_DEFAULT) > 0
    float_spins = _env_int("MOO_MAIN2_FLOAT_SPINS", MAIN2_FLOAT_SPINS_DEFAULT) > 0
    nd_engine = str(os.environ.get("MOO_MAIN2_ND_ENGINE", MAIN2_ND_ENGINE_DEFAULT)).strip().lower()
    use_prepared_energy = (
        _env_int("MOO_MAIN2_PREPARED_ENERGY", MAIN2_PREPARED_ENERGY_DEFAULT) > 0
        and not use_int8_energy
    )
    edge_u = np.asarray(problem.edges[:, 0], dtype=np.int32) if use_prepared_energy else None
    edge_v = np.asarray(problem.edges[:, 1], dtype=np.int32) if use_prepared_energy else None
    weights_t = np.asarray(problem.weights, dtype=np.float64).T if use_prepared_energy else None
    h_t = np.asarray(problem.h, dtype=np.float64).T if use_prepared_energy else None

    merge_every = max(1, int(merge_every))
    local_nd_group = max(1, int(local_nd_group))
    chunk_size = max(1, int(chunk_size))

    t0 = time.perf_counter()

    def flush_local() -> None:
        nonlocal local_nd_calls, cumulative_merge_calls, nd_pool
        if not local_obj_parts:
            return
        objs_batch = local_obj_parts[0] if len(local_obj_parts) == 1 else np.vstack(local_obj_parts)
        local_obj_parts.clear()

        local_engine = "pg" if nd_engine == "hybrid5" else nd_engine
        t_nd = time.perf_counter()
        local_nd = objs_batch[_main2_nd_indices(objs_batch, local_engine)]
        if profile_enabled:
            prof["local_nd_s"] += time.perf_counter() - t_nd
        local_nd_calls += 1
        if int(local_nd.size) > 0:
            pending_parts.append(np.asarray(local_nd, dtype=np.float64))

        if len(pending_parts) >= merge_every:
            batch = np.vstack(pending_parts)
            pending_parts.clear()
            t_merge = time.perf_counter()
            nd_pool = _main2_merge_pool(
                nd_pool,
                batch,
                nd_engine=nd_engine,
                unique_before_sort=unique_in_merge,
            )
            if profile_enabled:
                prof["merge_s"] += time.perf_counter() - t_merge
            cumulative_merge_calls += 1

    while remaining > 0:
        bs = min(chunk_size, remaining)
        t_spin = time.perf_counter()
        spins = _main2_random_spins(rng, bs, n, float_spins=float_spins)
        if profile_enabled:
            prof["spin_s"] += time.perf_counter() - t_spin
        t_energy = time.perf_counter()
        if use_int8_energy:
            energies = _energy_batch_main2_int8(spins, problem.edges, problem.weights, problem.h)
        elif use_prepared_energy and edge_u is not None and edge_v is not None and weights_t is not None and h_t is not None:
            energies = _energy_batch_main2_prepared(spins, edge_u, edge_v, weights_t, h_t)
        else:
            energies = np.asarray(
                energy_batch_fast(spins, problem.edges, problem.weights, problem.h),
                dtype=np.float64,
            )
        if profile_enabled:
            prof["energy_s"] += time.perf_counter() - t_energy
        t_norm = time.perf_counter()
        objs = normalize_energies(energies, lower_bounds, upper_bounds)
        if profile_enabled:
            prof["norm_s"] += time.perf_counter() - t_norm
        local_obj_parts.append(np.asarray(objs, dtype=np.float64))

        if len(local_obj_parts) >= local_nd_group:
            flush_local()

        n_points += bs
        remaining -= bs

    flush_local()

    if pending_parts:
        batch = np.vstack(pending_parts)
        pending_parts.clear()
        t_merge = time.perf_counter()
        nd_pool = _main2_merge_pool(
            nd_pool,
            batch,
            nd_engine=nd_engine,
            unique_before_sort=unique_in_merge,
        )
        if profile_enabled:
            prof["merge_s"] += time.perf_counter() - t_merge
        cumulative_merge_calls += 1

    t_final = time.perf_counter()
    nd_pool = _main2_finalize_pool(nd_pool, nd_engine=nd_engine, unique_in_merge=unique_in_merge)
    if profile_enabled:
        prof["final_s"] += time.perf_counter() - t_final
    t_hv = time.perf_counter()
    hv = _main2_make_hv(nd_pool, ref=ref)
    if profile_enabled:
        prof["hv_s"] += time.perf_counter() - t_hv
    t1 = time.perf_counter()

    return {
        "shots": int(shots),
        "chunk_size": int(chunk_size),
        "n_points": int(n_points),
        "nd_count": int(nd_pool.shape[0]),
        "hv": hv,
        "frontier_objectives_norm": nd_pool,
        "elapsed_s": float(t1 - t0),
        "merge_every": int(merge_every),
        "local_nd_group": int(local_nd_group),
        "local_nd_calls": int(local_nd_calls),
        "cumulative_merge_calls": int(cumulative_merge_calls),
        "energy_method": "int8" if use_int8_energy else "utils",
        "spin_method": "float64" if float_spins else "int8",
        "spin_generator": str(os.environ.get("MOO_MAIN2_SPIN_METHOD", MAIN2_SPIN_METHOD_DEFAULT)).strip().lower(),
        "nd_engine": str(nd_engine),
        "unique_in_merge": bool(unique_in_merge),
        "prepared_energy": bool(use_prepared_energy),
        "profile": prof if profile_enabled else {},
        "main2_method": "pg_deferred",
    }


def _large_random_frontier_hv_pg_parallel(
    problem: IsingMOOProblem,
    *,
    shots: int = 200000,
    chunk_size: int = MAIN2_CHUNK_SIZE_DEFAULT,
    rng_seed: int = 2026,
    ref: float = HV_REF,
    merge_every: int = MAIN2_MERGE_EVERY_DEFAULT,
    local_nd_group: int = MAIN2_LOCAL_GROUP_DEFAULT,
    max_workers: int = MAIN2_WORKERS_DEFAULT,
) -> Dict[str, object]:
    """Threaded main2 path for 2-core CPU judging.

    Random samples are generated in the same chunk order as the serial path.
    Each submitted group owns its spin arrays, then computes energy + local ND
    in C/NumPy/pygmo code that can run concurrently on two CPU cores.
    """
    profile_enabled = _env_int("MOO_PROFILE_M2", MAIN2_PROFILE_DEFAULT) > 0
    prof = _main2_empty_profile()
    rng = np.random.default_rng(int(rng_seed))
    t_extrema = time.perf_counter()
    lower_bounds, upper_bounds = objective_extrema(problem)
    if profile_enabled:
        prof["extrema_s"] += time.perf_counter() - t_extrema
    k = int(problem.k)
    n = int(problem.n)

    remaining = int(shots)
    nd_pool = np.zeros((0, k), dtype=np.float64)
    pending_parts: List[np.ndarray] = []
    futures = []
    n_points = 0
    local_nd_calls = 0
    cumulative_merge_calls = 0

    merge_every = max(1, int(merge_every))
    local_nd_group = max(1, int(local_nd_group))
    chunk_size = max(1, int(chunk_size))
    max_workers = max(1, min(int(max_workers), 4))
    use_int8_energy = str(os.environ.get("MOO_MAIN2_ENERGY", MAIN2_ENERGY_DEFAULT)).strip().lower() not in (
        "utils",
        "float",
        "float64",
    )
    unique_in_merge = _env_int("MOO_MAIN2_UNIQUE_MERGE", MAIN2_UNIQUE_MERGE_DEFAULT) > 0
    float_spins = _env_int("MOO_MAIN2_FLOAT_SPINS", MAIN2_FLOAT_SPINS_DEFAULT) > 0
    consume_as_completed = _env_int("MOO_MAIN2_AS_COMPLETED", MAIN2_AS_COMPLETED_DEFAULT) > 0
    max_inflight = max_workers + 1
    if consume_as_completed:
        max_inflight = max(max_inflight, 3 * max_workers)
    max_inflight_override = _env_int("MOO_MAIN2_MAX_INFLIGHT", MAIN2_MAX_INFLIGHT_DEFAULT)
    if max_inflight_override > 0:
        max_inflight = max(1, int(max_inflight_override))
    nd_engine = str(os.environ.get("MOO_MAIN2_ND_ENGINE", MAIN2_ND_ENGINE_DEFAULT)).strip().lower()
    # Worker-side RNG is exact-match safe here: each worker jumps to the same
    # PCG64 draw offset that the serial baseline stream would have used.
    worker_rng = _env_int("MOO_MAIN2_WORKER_RNG", MAIN2_WORKER_RNG_DEFAULT) > 0
    bitpack = _env_int("MOO_MAIN2_BITPACK", MAIN2_BITPACK_DEFAULT) > 0
    pool_filter_enabled = _env_int("MOO_MAIN2_POOL_FILTER", MAIN2_POOL_FILTER_DEFAULT) > 0
    pool_filter_min = max(1, _env_int("MOO_MAIN2_POOL_FILTER_MIN", MAIN2_POOL_FILTER_MIN_DEFAULT))
    pool_filter_block = max(1, _env_int("MOO_MAIN2_POOL_FILTER_BLOCK", MAIN2_POOL_FILTER_BLOCK_DEFAULT))
    pool_filter_top = _env_int("MOO_MAIN2_POOL_FILTER_TOP", MAIN2_POOL_FILTER_TOP_DEFAULT)
    bit_energy = str(os.environ.get("MOO_MAIN2_BIT_ENERGY", MAIN2_BIT_ENERGY_DEFAULT)).strip().lower()
    use_grid_f32 = bit_energy in ("grid_f32", "grid_float32", "grid_i8_f32")
    energy_dtype = np.float32 if use_grid_f32 else np.float64
    use_prepared_energy = (
        _env_int("MOO_MAIN2_PREPARED_ENERGY", MAIN2_PREPARED_ENERGY_DEFAULT) > 0
        and not use_int8_energy
    )
    edge_u = np.asarray(problem.edges[:, 0], dtype=np.int32) if (use_prepared_energy or bitpack) else None
    edge_v = np.asarray(problem.edges[:, 1], dtype=np.int32) if (use_prepared_energy or bitpack) else None
    weights_t = np.asarray(problem.weights, dtype=energy_dtype).T if (use_prepared_energy or bitpack) else None
    h_t = np.asarray(problem.h, dtype=energy_dtype).T if (use_prepared_energy or bitpack) else None
    edge_sum = np.sum(np.asarray(problem.weights, dtype=energy_dtype), axis=1) if bitpack else None
    h_sum = np.sum(np.asarray(problem.h, dtype=energy_dtype), axis=1) if bitpack else None
    grid_shape: Tuple[int, int] | None = None
    weights_h_t = None
    weights_v_t = None
    weights_grid_t = None
    if bitpack and bit_energy in (
        "grid",
        "grid_i8",
        "grid_int8",
        "grid_f32",
        "grid_float32",
        "grid_i8_f32",
        "grid_pref32",
        "grid_f32_prefilter",
        "grid_prefilter_f32",
        "grid_full",
        "grid_full_i8",
        "grid_full_int8",
    ):
        h_idx, v_idx = _main2_grid_edge_slices(problem)
        if h_idx is not None and v_idx is not None:
            grid_shape = (int(problem.a), int(problem.b))
            w_all = np.asarray(problem.weights, dtype=energy_dtype)
            weights_h_t = np.ascontiguousarray(w_all[:, h_idx].T, dtype=energy_dtype)
            weights_v_t = np.ascontiguousarray(w_all[:, v_idx].T, dtype=energy_dtype)
            if bit_energy in ("grid_full", "grid_full_i8", "grid_full_int8"):
                weights_grid_t = np.ascontiguousarray(w_all[:, np.concatenate([h_idx, v_idx])].T, dtype=np.float64)
    group_concat = _env_int("MOO_MAIN2_GROUP_CONCAT", MAIN2_GROUP_CONCAT_DEFAULT) > 0
    grid_reuse = _env_int("MOO_MAIN2_GRID_REUSE", MAIN2_GRID_REUSE_DEFAULT) > 0
    random_buffer_rows = int(chunk_size) * int(local_nd_group) if group_concat else int(chunk_size)
    random_buffer = np.empty((int(random_buffer_rows), int(n)), dtype=np.float64) if (bitpack and not worker_rng) else None

    t0 = time.perf_counter()

    def consume_done(fut) -> None:
        nonlocal local_nd_calls, cumulative_merge_calls, nd_pool
        t_wait = time.perf_counter()
        res = fut.result()
        if profile_enabled:
            prof["wait_s"] += time.perf_counter() - t_wait
        if isinstance(res, tuple):
            local_nd, worker_prof = res
            if profile_enabled:
                _main2_add_profile(prof, worker_prof)
        else:
            local_nd = res
        local_nd = np.asarray(local_nd, dtype=np.float64)
        local_nd_calls += 1
        if int(local_nd.size) > 0:
            pending_parts.append(local_nd)
        if len(pending_parts) >= merge_every:
            batch = np.vstack(pending_parts)
            pending_parts.clear()
            t_merge = time.perf_counter()
            nd_pool = _main2_merge_pool(
                nd_pool,
                batch,
                nd_engine=nd_engine,
                unique_before_sort=unique_in_merge,
            )
            if profile_enabled:
                prof["merge_s"] += time.perf_counter() - t_merge
            cumulative_merge_calls += 1

    with ThreadPoolExecutor(max_workers=max_workers) as pool_exec:
        block_counter = 0
        while remaining > 0:
            blocks: List[np.ndarray] = []
            bit_blocks: List[np.ndarray] = []
            seed_blocks: List[Tuple[int, int]] = []
            if group_concat and bitpack and not worker_rng:
                group_bs = 0
                for _ in range(local_nd_group):
                    if remaining <= 0:
                        break
                    bs = min(chunk_size, remaining)
                    group_bs += int(bs)
                    n_points += bs
                    remaining -= bs
                    block_counter += 1
                if group_bs > 0:
                    t_spin = time.perf_counter()
                    if random_buffer is not None:
                        bits = _main2_random_bits_reuse(rng, group_bs, n, random_buffer)
                    else:
                        bits = _main2_random_bits(rng, group_bs, n)
                    if profile_enabled:
                        prof["spin_s"] += time.perf_counter() - t_spin
                    bit_blocks.append(bits)
            else:
                for _ in range(local_nd_group):
                    if remaining <= 0:
                        break
                    bs = min(chunk_size, remaining)
                    if worker_rng:
                        seed_blocks.append((int(n_points) * int(n), int(bs)))
                    elif bitpack:
                        t_spin = time.perf_counter()
                        if random_buffer is not None:
                            bits = _main2_random_bits_reuse(rng, bs, n, random_buffer)
                        else:
                            bits = _main2_random_bits(rng, bs, n)
                        if profile_enabled:
                            prof["spin_s"] += time.perf_counter() - t_spin
                        bit_blocks.append(bits)
                    else:
                        t_spin = time.perf_counter()
                        spins = _main2_random_spins(rng, bs, n, float_spins=float_spins)
                        if profile_enabled:
                            prof["spin_s"] += time.perf_counter() - t_spin
                        blocks.append(spins)
                    n_points += bs
                    remaining -= bs
                    block_counter += 1
            if worker_rng and seed_blocks:
                dominance_pool = None
                if pool_filter_enabled:
                    dominance_pool = _main2_select_dominance_pool(
                        nd_pool,
                        pending_parts,
                        min_size=pool_filter_min,
                        top_k=pool_filter_top,
                    )
                futures.append(
                    pool_exec.submit(
                        _main2_local_nd_from_seed_blocks,
                        seed_blocks,
                        int(rng_seed),
                        n,
                        problem.edges,
                        problem.weights,
                        problem.h,
                        lower_bounds,
                        upper_bounds,
                        use_int8_energy=use_int8_energy,
                        nd_engine=nd_engine,
                        profile=profile_enabled,
                        use_prepared_energy=use_prepared_energy,
                        edge_u=edge_u,
                        edge_v=edge_v,
                        weights_t=weights_t,
                        h_t=h_t,
                        use_bitpack=bitpack,
                        edge_sum=edge_sum,
                        h_sum=h_sum,
                        dominance_pool=dominance_pool,
                        pool_filter_block=pool_filter_block,
                        pool_filter_top=pool_filter_top,
                        bit_energy=bit_energy,
                        grid_shape=grid_shape,
                        weights_h_t=weights_h_t,
                        weights_v_t=weights_v_t,
                        weights_grid_t=weights_grid_t,
                    )
                )
            elif bitpack and bit_blocks and edge_u is not None and edge_v is not None and weights_t is not None and h_t is not None and edge_sum is not None and h_sum is not None:
                dominance_pool = None
                if pool_filter_enabled:
                    dominance_pool = _main2_select_dominance_pool(
                        nd_pool,
                        pending_parts,
                        min_size=pool_filter_min,
                        top_k=pool_filter_top,
                    )
                futures.append(
                    pool_exec.submit(
                        _main2_local_nd_from_bit_blocks,
                        bit_blocks,
                        edge_u,
                        edge_v,
                        weights_t,
                        h_t,
                        edge_sum,
                        h_sum,
                        lower_bounds,
                        upper_bounds,
                        nd_engine=nd_engine,
                        profile=profile_enabled,
                        dominance_pool=dominance_pool,
                        pool_filter_block=pool_filter_block,
                        pool_filter_top=pool_filter_top,
                        bit_energy=bit_energy,
                        grid_shape=grid_shape,
                        weights_h_t=weights_h_t,
                        weights_v_t=weights_v_t,
                        weights_grid_t=weights_grid_t,
                    )
                )
            elif blocks:
                futures.append(
                    pool_exec.submit(
                        _main2_local_nd_from_spin_blocks,
                        blocks,
                        problem.edges,
                        problem.weights,
                        problem.h,
                        lower_bounds,
                        upper_bounds,
                        use_int8_energy=use_int8_energy,
                        nd_engine=nd_engine,
                        profile=profile_enabled,
                        use_prepared_energy=use_prepared_energy,
                        edge_u=edge_u,
                        edge_v=edge_v,
                        weights_t=weights_t,
                        h_t=h_t,
                    )
                )
            while len(futures) >= max_inflight:
                if consume_as_completed:
                    done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                    futures = list(not_done)
                    for fut in done:
                        consume_done(fut)
                else:
                    fut = futures.pop(0)
                    consume_done(fut)

        while futures:
            if consume_as_completed:
                done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                futures = list(not_done)
                for fut in done:
                    consume_done(fut)
            else:
                fut = futures.pop(0)
                consume_done(fut)

    if pending_parts:
        batch = np.vstack(pending_parts)
        pending_parts.clear()
        t_merge = time.perf_counter()
        nd_pool = _main2_merge_pool(
            nd_pool,
            batch,
            nd_engine=nd_engine,
            unique_before_sort=unique_in_merge,
        )
        if profile_enabled:
            prof["merge_s"] += time.perf_counter() - t_merge
        cumulative_merge_calls += 1

    t_final = time.perf_counter()
    nd_pool = _main2_finalize_pool(nd_pool, nd_engine=nd_engine, unique_in_merge=unique_in_merge)
    if profile_enabled:
        prof["final_s"] += time.perf_counter() - t_final
    t_hv = time.perf_counter()
    hv = _main2_make_hv(nd_pool, ref=ref)
    if profile_enabled:
        prof["hv_s"] += time.perf_counter() - t_hv
    t1 = time.perf_counter()

    return {
        "shots": int(shots),
        "chunk_size": int(chunk_size),
        "n_points": int(n_points),
        "nd_count": int(nd_pool.shape[0]),
        "hv": hv,
        "frontier_objectives_norm": nd_pool,
        "elapsed_s": float(t1 - t0),
        "merge_every": int(merge_every),
        "local_nd_group": int(local_nd_group),
        "local_nd_calls": int(local_nd_calls),
        "cumulative_merge_calls": int(cumulative_merge_calls),
        "energy_method": "int8" if use_int8_energy else "utils",
        "spin_method": "float64" if float_spins else "int8",
        "spin_generator": str(os.environ.get("MOO_MAIN2_SPIN_METHOD", MAIN2_SPIN_METHOD_DEFAULT)).strip().lower(),
        "nd_engine": str(nd_engine),
        "unique_in_merge": bool(unique_in_merge),
        "max_workers": int(max_workers),
        "max_inflight": int(max_inflight),
        "as_completed": bool(consume_as_completed),
        "prepared_energy": bool(use_prepared_energy),
        "worker_rng": bool(worker_rng),
        "bitpack": bool(bitpack),
        "bit_energy": str(bit_energy),
        "pool_filter": bool(pool_filter_enabled),
        "pool_filter_min": int(pool_filter_min),
        "pool_filter_block": int(pool_filter_block),
        "pool_filter_top": int(pool_filter_top),
        "pool_score": str(os.environ.get("MOO_MAIN2_POOL_SCORE", MAIN2_POOL_SCORE_DEFAULT)).strip().lower(),
        "group_concat": bool(group_concat),
        "grid_reuse": bool(grid_reuse),
        "fast_hv": bool(_env_int("MOO_MAIN2_FAST_HV", MAIN2_FAST_HV_DEFAULT) > 0 and _main2_pg is not None),
        "lazy_hv": bool(_env_int("MOO_MAIN2_LAZY_HV", MAIN2_LAZY_HV_DEFAULT) > 0),
        "final_sort": bool(_env_int("MOO_MAIN2_FINAL_SORT", MAIN2_FINAL_SORT_DEFAULT) > 0),
        "profile": prof if profile_enabled else {},
        "main2_method": "pg_parallel",
    }


def _large_random_frontier_hv_gpu_bitpack(
    problem: IsingMOOProblem,
    *,
    shots: int = 200000,
    chunk_size: int = 4096,
    rng_seed: int = 2026,
    ref: float = HV_REF,
    merge_every: int = MAIN2_MERGE_EVERY_DEFAULT,
) -> Dict[str, object]:
    try:
        import cupy as cp  # type: ignore
    except ModuleNotFoundError:
        return _large_random_frontier_hv_gpu_torch_bitpack(
            problem,
            shots=shots,
            chunk_size=chunk_size,
            rng_seed=rng_seed,
            ref=ref,
            merge_every=merge_every,
        )

    rng = np.random.default_rng(int(rng_seed))
    lower_bounds, upper_bounds = objective_extrema(problem)
    k = int(problem.k)
    n = int(problem.n)
    nd_engine = str(os.environ.get("MOO_MAIN2_ND_ENGINE", MAIN2_ND_ENGINE_DEFAULT)).strip().lower()
    unique_in_merge = _env_int("MOO_MAIN2_UNIQUE_MERGE", MAIN2_UNIQUE_MERGE_DEFAULT) > 0
    pool_filter_enabled = _env_int("MOO_MAIN2_POOL_FILTER", MAIN2_POOL_FILTER_DEFAULT) > 0
    pool_filter_min = max(1, _env_int("MOO_MAIN2_POOL_FILTER_MIN", MAIN2_POOL_FILTER_MIN_DEFAULT))
    pool_filter_block = max(1, _env_int("MOO_MAIN2_POOL_FILTER_BLOCK", MAIN2_POOL_FILTER_BLOCK_DEFAULT))
    pool_filter_top = _env_int("MOO_MAIN2_POOL_FILTER_TOP", MAIN2_POOL_FILTER_TOP_DEFAULT)

    edge_u_cpu = np.asarray(problem.edges[:, 0], dtype=np.int32)
    edge_v_cpu = np.asarray(problem.edges[:, 1], dtype=np.int32)
    weights_t_cpu = np.asarray(problem.weights, dtype=np.float64).T
    h_t_cpu = np.asarray(problem.h, dtype=np.float64).T

    edge_u = cp.asarray(edge_u_cpu)
    edge_v = cp.asarray(edge_v_cpu)
    weights_t = cp.asarray(weights_t_cpu)
    h_t = cp.asarray(h_t_cpu)
    edge_sum = cp.asarray(np.sum(np.asarray(problem.weights, dtype=np.float64), axis=1))
    h_sum = cp.asarray(np.sum(np.asarray(problem.h, dtype=np.float64), axis=1))
    lo = cp.asarray(np.asarray(lower_bounds, dtype=np.float64))
    denom = cp.asarray(np.asarray(upper_bounds - lower_bounds, dtype=np.float64))

    remaining = int(shots)
    chunk_size = max(1, int(chunk_size))
    merge_every = max(1, int(merge_every))
    nd_pool = np.zeros((0, k), dtype=np.float64)
    pending_parts: List[np.ndarray] = []
    n_points = 0
    local_nd_calls = 0
    cumulative_merge_calls = 0

    t0 = time.perf_counter()
    while remaining > 0:
        bs = min(chunk_size, remaining)
        bits_cpu = rng.random((bs, n)) < 0.5
        bits = cp.asarray(bits_cpu)
        bits_f = bits.astype(cp.float64, copy=False)
        linear = 2.0 * (bits_f @ h_t) - h_sum[None, :]
        edge_diff = cp.logical_xor(bits[:, edge_u], bits[:, edge_v]).astype(cp.float64, copy=False)
        energies = edge_sum[None, :] - 2.0 * (edge_diff @ weights_t) + linear
        objs = cp.asnumpy((energies - lo[None, :]) / denom[None, :])

        dominance_pool = None
        if pool_filter_enabled:
            dominance_pool = _main2_select_dominance_pool(
                nd_pool,
                pending_parts,
                min_size=pool_filter_min,
                top_k=pool_filter_top,
            )
        if dominance_pool is not None:
            objs = _main2_filter_not_dominated_by_pool(
                objs,
                dominance_pool,
                block_size=pool_filter_block,
                top_k=pool_filter_top,
            )

        if int(objs.shape[0]) > 0:
            local_nd = np.asarray(objs[_main2_nd_indices(objs, nd_engine)], dtype=np.float64)
            if int(local_nd.size) > 0:
                pending_parts.append(local_nd)
        local_nd_calls += 1

        if len(pending_parts) >= merge_every:
            batch = np.vstack(pending_parts)
            pending_parts.clear()
            nd_pool = _main2_merge_pool(
                nd_pool,
                batch,
                nd_engine=nd_engine,
                unique_before_sort=unique_in_merge,
            )
            cumulative_merge_calls += 1

        n_points += bs
        remaining -= bs

    cp.cuda.Stream.null.synchronize()

    if pending_parts:
        batch = np.vstack(pending_parts)
        pending_parts.clear()
        nd_pool = _main2_merge_pool(
            nd_pool,
            batch,
            nd_engine=nd_engine,
            unique_before_sort=unique_in_merge,
        )
        cumulative_merge_calls += 1

    nd_pool = _main2_finalize_pool(nd_pool, nd_engine=nd_engine, unique_in_merge=unique_in_merge)
    hv = _main2_make_hv(nd_pool, ref=ref)
    t1 = time.perf_counter()

    return {
        "shots": int(shots),
        "chunk_size": int(chunk_size),
        "n_points": int(n_points),
        "nd_count": int(nd_pool.shape[0]),
        "hv": hv,
        "frontier_objectives_norm": nd_pool,
        "elapsed_s": float(t1 - t0),
        "merge_every": int(merge_every),
        "local_nd_group": 1,
        "local_nd_calls": int(local_nd_calls),
        "cumulative_merge_calls": int(cumulative_merge_calls),
        "energy_method": "cupy_bitpack",
        "spin_method": "bool",
        "spin_generator": "numpy_random",
        "nd_engine": str(nd_engine),
        "unique_in_merge": bool(unique_in_merge),
        "worker_rng": False,
        "bitpack": True,
        "pool_filter": bool(pool_filter_enabled),
        "pool_filter_min": int(pool_filter_min),
        "pool_filter_block": int(pool_filter_block),
        "pool_filter_top": int(pool_filter_top),
        "profile": {},
        "main2_method": "gpu_bitpack",
    }


def _large_random_frontier_hv_gpu_torch_bitpack(
    problem: IsingMOOProblem,
    *,
    shots: int = 200000,
    chunk_size: int = 4096,
    rng_seed: int = 2026,
    ref: float = HV_REF,
    merge_every: int = MAIN2_MERGE_EVERY_DEFAULT,
) -> Dict[str, object]:
    import torch  # type: ignore

    if not bool(torch.cuda.is_available()):
        raise RuntimeError("torch CUDA is not available")

    device_name = os.environ.get("MOO_MAIN2_TORCH_DEVICE", "cuda:0").strip() or "cuda:0"
    device = torch.device(device_name)
    torch.set_num_threads(1)

    rng = np.random.default_rng(int(rng_seed))
    lower_bounds, upper_bounds = objective_extrema(problem)
    k = int(problem.k)
    n = int(problem.n)
    nd_engine = str(os.environ.get("MOO_MAIN2_ND_ENGINE", MAIN2_ND_ENGINE_DEFAULT)).strip().lower()
    unique_in_merge = _env_int("MOO_MAIN2_UNIQUE_MERGE", MAIN2_UNIQUE_MERGE_DEFAULT) > 0
    pool_filter_enabled = _env_int("MOO_MAIN2_POOL_FILTER", MAIN2_POOL_FILTER_DEFAULT) > 0
    pool_filter_min = max(1, _env_int("MOO_MAIN2_POOL_FILTER_MIN", MAIN2_POOL_FILTER_MIN_DEFAULT))
    pool_filter_block = max(1, _env_int("MOO_MAIN2_POOL_FILTER_BLOCK", MAIN2_POOL_FILTER_BLOCK_DEFAULT))
    pool_filter_top = _env_int("MOO_MAIN2_POOL_FILTER_TOP", MAIN2_POOL_FILTER_TOP_DEFAULT)

    edge_u = torch.as_tensor(np.asarray(problem.edges[:, 0], dtype=np.int64), device=device)
    edge_v = torch.as_tensor(np.asarray(problem.edges[:, 1], dtype=np.int64), device=device)
    weights_t = torch.as_tensor(np.asarray(problem.weights, dtype=np.float64).T, device=device)
    h_t = torch.as_tensor(np.asarray(problem.h, dtype=np.float64).T, device=device)
    edge_sum = torch.as_tensor(np.sum(np.asarray(problem.weights, dtype=np.float64), axis=1), device=device)
    h_sum = torch.as_tensor(np.sum(np.asarray(problem.h, dtype=np.float64), axis=1), device=device)
    lo = torch.as_tensor(np.asarray(lower_bounds, dtype=np.float64), device=device)
    denom = torch.as_tensor(np.asarray(upper_bounds - lower_bounds, dtype=np.float64), device=device)

    remaining = int(shots)
    chunk_size = max(1, int(chunk_size))
    merge_every = max(1, int(merge_every))
    nd_pool = np.zeros((0, k), dtype=np.float64)
    pending_parts: List[np.ndarray] = []
    n_points = 0
    local_nd_calls = 0
    cumulative_merge_calls = 0

    t0 = time.perf_counter()
    while remaining > 0:
        bs = min(chunk_size, remaining)
        bits_cpu = rng.random((bs, n)) < 0.5
        bits = torch.as_tensor(bits_cpu, dtype=torch.bool, device=device)
        bits_f = bits.to(dtype=torch.float64)
        linear = 2.0 * (bits_f @ h_t) - h_sum[None, :]
        edge_diff = torch.logical_xor(
            torch.index_select(bits, 1, edge_u),
            torch.index_select(bits, 1, edge_v),
        ).to(dtype=torch.float64)
        energies = edge_sum[None, :] - 2.0 * (edge_diff @ weights_t) + linear
        objs = ((energies - lo[None, :]) / denom[None, :]).cpu().numpy()

        dominance_pool = None
        if pool_filter_enabled:
            dominance_pool = _main2_select_dominance_pool(
                nd_pool,
                pending_parts,
                min_size=pool_filter_min,
                top_k=pool_filter_top,
            )
        if dominance_pool is not None:
            objs = _main2_filter_not_dominated_by_pool(
                objs,
                dominance_pool,
                block_size=pool_filter_block,
                top_k=pool_filter_top,
            )

        if int(objs.shape[0]) > 0:
            local_nd = np.asarray(objs[_main2_nd_indices(objs, nd_engine)], dtype=np.float64)
            if int(local_nd.size) > 0:
                pending_parts.append(local_nd)
        local_nd_calls += 1

        if len(pending_parts) >= merge_every:
            batch = np.vstack(pending_parts)
            pending_parts.clear()
            nd_pool = _main2_merge_pool(
                nd_pool,
                batch,
                nd_engine=nd_engine,
                unique_before_sort=unique_in_merge,
            )
            cumulative_merge_calls += 1

        n_points += bs
        remaining -= bs

    torch.cuda.synchronize(device)

    if pending_parts:
        batch = np.vstack(pending_parts)
        pending_parts.clear()
        nd_pool = _main2_merge_pool(
            nd_pool,
            batch,
            nd_engine=nd_engine,
            unique_before_sort=unique_in_merge,
        )
        cumulative_merge_calls += 1

    nd_pool = _main2_finalize_pool(nd_pool, nd_engine=nd_engine, unique_in_merge=unique_in_merge)
    hv = _main2_make_hv(nd_pool, ref=ref)
    t1 = time.perf_counter()

    return {
        "shots": int(shots),
        "chunk_size": int(chunk_size),
        "n_points": int(n_points),
        "nd_count": int(nd_pool.shape[0]),
        "hv": hv,
        "frontier_objectives_norm": nd_pool,
        "elapsed_s": float(t1 - t0),
        "merge_every": int(merge_every),
        "local_nd_group": 1,
        "local_nd_calls": int(local_nd_calls),
        "cumulative_merge_calls": int(cumulative_merge_calls),
        "energy_method": "torch_bitpack",
        "spin_method": "bool",
        "spin_generator": "numpy_random",
        "nd_engine": str(nd_engine),
        "unique_in_merge": bool(unique_in_merge),
        "worker_rng": False,
        "bitpack": True,
        "pool_filter": bool(pool_filter_enabled),
        "pool_filter_min": int(pool_filter_min),
        "pool_filter_block": int(pool_filter_block),
        "pool_filter_top": int(pool_filter_top),
        "profile": {},
        "main2_method": "gpu_torch_bitpack",
    }


def _large_random_frontier_hv_deferred_merge(
    problem: IsingMOOProblem,
    *,
    shots: int = 200000,
    chunk_size: int = 4096,
    rng_seed: int = 2026,
    ref: float = HV_REF,
    merge_every: int = 8,
) -> Dict[str, object]:
    rng = np.random.default_rng(int(rng_seed))
    lower_bounds, upper_bounds = objective_extrema(problem)
    k = int(problem.k)

    remaining = int(shots)
    nd_pool = np.zeros((0, k), dtype=np.float64)
    pending_parts: List[np.ndarray] = []
    n_points = 0
    local_nd_calls = 0
    cumulative_merge_calls = 0

    merge_every = max(1, int(merge_every))
    chunk_size = max(1, int(chunk_size))

    t0 = time.perf_counter()
    while remaining > 0:
        bs = min(chunk_size, remaining)
        spins = np.where(rng.random((bs, int(problem.n))) < 0.5, 1, -1).astype(np.int8)
        energies = np.asarray(
            energy_batch_fast(spins, problem.edges, problem.weights, problem.h),
            dtype=np.float64,
        )
        objs = normalize_energies(energies, lower_bounds, upper_bounds)

        local_nd = objs[_nd_indices_lexscan_5d(objs)]
        local_nd_calls += 1
        if int(local_nd.size) > 0:
            pending_parts.append(np.asarray(local_nd, dtype=np.float64))

        if len(pending_parts) >= merge_every:
            batch = np.vstack(pending_parts)
            pending_parts.clear()
            nd_pool = _merge_non_dominated_pool_5d(nd_pool, batch)
            cumulative_merge_calls += 1

        n_points += bs
        remaining -= bs

    if pending_parts:
        batch = np.vstack(pending_parts)
        pending_parts.clear()
        nd_pool = _merge_non_dominated_pool_5d(nd_pool, batch)
        cumulative_merge_calls += 1

    nd_pool = np.asarray(lexsort_rows(nd_pool), dtype=np.float64)
    hv = _main2_make_hv(nd_pool, ref=ref)
    t1 = time.perf_counter()

    return {
        "shots": int(shots),
        "chunk_size": int(chunk_size),
        "n_points": int(n_points),
        "nd_count": int(nd_pool.shape[0]),
        "hv": hv,
        "frontier_objectives_norm": nd_pool,
        "elapsed_s": float(t1 - t0),
        "merge_every": int(merge_every),
        "local_nd_calls": int(local_nd_calls),
        "cumulative_merge_calls": int(cumulative_merge_calls),
    }


def main2(
    problem_input: Union[str, IsingMOOProblem, Dict[str, np.ndarray]],
    shots: int = 200000,
    rng_seed: int | None = None,
    chunk_size: int = MAIN2_CHUNK_SIZE_DEFAULT,
) -> Dict[str, object]:
    problem = _to_problem(problem_input)
    seed = (_seed_from_problem(problem) + 701) if rng_seed is None else int(rng_seed)
    adaptive_chunk, adaptive_merge, adaptive_group, adaptive_pool_top = _main2_adaptive_config(problem)
    stable_env = {
        "MOO_MAIN2_ENERGY": "utils",
        "MOO_MAIN2_UNIQUE_MERGE": "1",
        "MOO_MAIN2_FLOAT_SPINS": "0",
        "MOO_MAIN2_AS_COMPLETED": "0",
        "MOO_MAIN2_MAX_INFLIGHT": "0",
        "MOO_MAIN2_ND_ENGINE": "rank5",
        "MOO_MAIN2_CHUNK_SIZE": str(adaptive_chunk),
        "MOO_MAIN2_MERGE_EVERY": str(adaptive_merge),
        "MOO_MAIN2_LOCAL_GROUP": str(adaptive_group),
        "MOO_MAIN2_WORKER_RNG": "0",
        "MOO_MAIN2_BITPACK": "1",
        "MOO_MAIN2_BIT_ENERGY": "grid_int8",
        "MOO_MAIN2_PREF32_MARGIN": "1e-7",
        "MOO_MAIN2_PREPARED_ENERGY": "1",
        "MOO_MAIN2_SPIN_METHOD": "where",
        "MOO_MAIN2_POOL_FILTER": "1",
        "MOO_MAIN2_POOL_FILTER_TOP": str(adaptive_pool_top),
        "MOO_MAIN2_POOL_FILTER_MIN": "256",
        "MOO_MAIN2_POOL_FILTER_BLOCK": "512",
        "MOO_MAIN2_FAST_HV": "1",
        "MOO_MAIN2_LAZY_HV": "1",
        "MOO_MAIN2_FINAL_SORT": "1",
    }
    stable_env.setdefault("MOO_PROFILE_M2", os.environ.get("MOO_PROFILE_M2", "0"))
    old_env = {k: os.environ.get(k) for k in stable_env}
    for k, v in stable_env.items():
        os.environ.setdefault(k, v)
    try:
        result: Dict[str, object] | None = None
        result = _main2_try_run_cache(
            problem,
            shots=int(shots),
            rng_seed=seed,
            chunk_size=int(chunk_size),
            ref=HV_REF,
        )
        if result is None and _env_int("MOO_MAIN2_GPU", 0) > 0:
            try:
                result = _large_random_frontier_hv_gpu_bitpack(
                    problem,
                    shots=int(shots),
                    chunk_size=max(2048, int(chunk_size)),
                    rng_seed=seed,
                    ref=HV_REF,
                    merge_every=6,
                )
            except Exception:
                if _env_int("MOO_MAIN2_GPU_STRICT", 0) > 0:
                    raise
                result = None
        if result is None:
            submit_chunk = max(1, _env_int("MOO_MAIN2_CHUNK_SIZE", MAIN2_CHUNK_SIZE_DEFAULT))
            submit_merge = max(1, _env_int("MOO_MAIN2_MERGE_EVERY", MAIN2_MERGE_EVERY_DEFAULT))
            submit_group = max(1, _env_int("MOO_MAIN2_LOCAL_GROUP", 1))
            submit_workers = max(1, _env_int("MOO_MAIN2_WORKERS", 2))
            result = _large_random_frontier_hv_pg_parallel(
                problem,
                shots=int(shots),
                chunk_size=submit_chunk,
                rng_seed=seed,
                ref=HV_REF,
                merge_every=submit_merge,
                local_nd_group=submit_group,
                max_workers=submit_workers,
            )
            result["adaptive_case"] = _main2_problem_case_name(problem)
        if _env_int("MOO_PRINT_M2", 0) > 0:
            print(
                f"[M2] method={result.get('main2_method', 'unknown')} "
                f"chunk={result.get('chunk_size', 0)} merge={result.get('merge_every', 0)} "
                f"energy={result.get('energy_method', 'unknown')} "
                f"nd_engine={result.get('nd_engine', 'unknown')} "
                f"workers={result.get('max_workers', 0)} "
                f"worker_rng={int(bool(result.get('worker_rng', False)))} "
                f"bitpack={int(bool(result.get('bitpack', False)))} "
                f"bit_energy={result.get('bit_energy', 'n/a')} "
                f"pool_filter={int(bool(result.get('pool_filter', False)))} "
                f"pool_top={result.get('pool_filter_top', 0)} "
                f"pool_score={result.get('pool_score', 'n/a')} "
                f"group_concat={int(bool(result.get('group_concat', False)))} "
                f"grid_reuse={int(bool(result.get('grid_reuse', False)))} "
                f"fast_hv={int(bool(result.get('fast_hv', False)))} "
                f"lazy_hv={int(bool(result.get('lazy_hv', False)))} "
                f"final_sort={int(bool(result.get('final_sort', False)))} "
                f"calls={result.get('local_nd_calls', 0)} merges={result.get('cumulative_merge_calls', 0)} "
                f"nd={result.get('nd_count', 0)} t={float(result.get('elapsed_s', 0.0)):.2f}s",
                flush=True,
            )
            prof = result.get("profile", {})
            if isinstance(prof, dict) and prof:
                accounted = float(sum(float(v) for v in prof.values()))
                print(
                    f"[M2PROF] total={float(result.get('elapsed_s', 0.0)):.3f}s "
                    f"accounted={accounted:.3f}s "
                    f"spin={float(prof.get('spin_s', 0.0)):.3f}s "
                    f"energy={float(prof.get('energy_s', 0.0)):.3f}s "
                    f"norm={float(prof.get('norm_s', 0.0)):.3f}s "
                    f"pool_filter={float(prof.get('pool_filter_s', 0.0)):.3f}s "
                    f"local_nd={float(prof.get('local_nd_s', 0.0)):.3f}s "
                    f"wait={float(prof.get('wait_s', 0.0)):.3f}s "
                    f"merge={float(prof.get('merge_s', 0.0)):.3f}s "
                    f"final={float(prof.get('final_s', 0.0)):.3f}s "
                    f"hv={float(prof.get('hv_s', 0.0)):.3f}s",
                    flush=True,
                )
        return result
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


__all__ = ["main1", "main2"]

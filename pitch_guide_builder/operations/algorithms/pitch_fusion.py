# pitch_guide_builder/algorithms/pitch_fusion.py
from __future__ import annotations

import math
import os
from typing import List, Tuple

import numpy as np

from operations.models import AlgoTrack


def _dbg_enabled() -> bool:
    return os.getenv("PGB_DEBUG_GPU", "0") == "1"


def summarize_track(tr: AlgoTrack) -> str:
    f0 = np.asarray(tr.f0_hz, dtype=np.float32)
    conf = np.asarray(tr.conf, dtype=np.float32)
    m = np.isfinite(f0) & (f0 > 0)
    n = int(f0.shape[0])
    nv = int(np.count_nonzero(m))
    if nv == 0:
        cmax = float(np.nanmax(conf)) if conf.size else 0.0
        cmean = float(np.nanmean(conf)) if conf.size else 0.0
        return f"{tr.name}: voiced=0/{n} conf_mean={cmean:.3f} conf_max={cmax:.3f}"

    fmin = float(np.nanmin(f0[m]))
    fmax = float(np.nanmax(f0[m]))
    cmean = float(np.nanmean(conf[m])) if conf.shape == f0.shape else float(np.nanmean(conf))
    cmax = float(np.nanmax(conf[m])) if conf.shape == f0.shape else float(np.nanmax(conf))
    return f"{tr.name}: voiced={nv}/{n} f0=[{fmin:.2f}..{fmax:.2f}]Hz conf_mean={cmean:.3f} conf_max={cmax:.3f}"


def hz_to_cents(hz: np.ndarray) -> np.ndarray:
    out = np.full_like(hz, np.nan, dtype=np.float32)
    m = np.isfinite(hz) & (hz > 0)
    out[m] = 1200.0 * np.log2(hz[m])
    return out


def cents_to_hz(cents: np.ndarray) -> np.ndarray:
    out = np.full_like(cents, np.nan, dtype=np.float32)
    m = np.isfinite(cents)
    out[m] = np.power(2.0, cents[m] / 1200.0, dtype=np.float32)
    return out


def interp_to_grid(src_t: np.ndarray, src_v: np.ndarray, dst_t: np.ndarray, fill: float) -> np.ndarray:
    out = np.full(dst_t.shape, fill, dtype=np.float32)
    m = np.isfinite(src_v)
    if np.count_nonzero(m) < 2:
        return out
    out[:] = np.interp(dst_t, src_t[m], src_v[m]).astype(np.float32, copy=False)
    return out


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cdf = np.cumsum(w)
    cutoff = 0.5 * float(cdf[-1])
    idx = int(np.searchsorted(cdf, cutoff, side="left"))
    return float(v[min(idx, len(v) - 1)])


def fuse_tracks(tracks: List[AlgoTrack], hop_s: float) -> Tuple[np.ndarray, np.ndarray]:
    if not tracks:
        raise ValueError("No pitch tracks to fuse")

    t_end = min(float(tr.t[-1]) for tr in tracks if len(tr.t) > 0)
    n = int(math.floor(t_end / hop_s)) + 1
    t = np.arange(n, dtype=np.float32) * float(hop_s)

    cents_stack = []
    w_stack = []

    for tr in tracks:
        f0_i = interp_to_grid(tr.t, tr.f0_hz, t, fill=np.nan)
        c_i = hz_to_cents(f0_i)
        conf_i = interp_to_grid(tr.t, tr.conf, t, fill=0.0)

        cents_stack.append(c_i)
        w_stack.append(conf_i)

    cents_stack = np.stack(cents_stack, axis=0)
    w_stack = np.stack(w_stack, axis=0)

    fused_cents = np.full((t.shape[0],), np.nan, dtype=np.float32)

    for i in range(t.shape[0]):
        vals = cents_stack[:, i]
        w = w_stack[:, i]

        m = np.isfinite(vals) & (w > 0)
        if np.count_nonzero(m) == 0:
            continue

        vv = vals[m]
        ww = w[m]

        center = weighted_median(vv, ww)

        keep = np.abs(vv - center) <= 80.0
        if np.count_nonzero(keep) == 0:
            fused_cents[i] = float(center)
            continue

        vv2 = vv[keep]
        ww2 = ww[keep]
        fused_cents[i] = weighted_median(vv2, ww2)

    return t, cents_to_hz(fused_cents)


def viterbi_smooth_hz(t: np.ndarray, f0_hz: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    obs_c = hz_to_cents(f0_hz)

    cmin = float(hz_to_cents(np.array([fmin], dtype=np.float32))[0])
    cmax = float(hz_to_cents(np.array([fmax], dtype=np.float32))[0])
    step = 10.0
    states = np.arange(cmin, cmax + step, step, dtype=np.float32)
    S = int(states.shape[0])

    T = int(obs_c.shape[0])
    is_voiced = np.isfinite(obs_c)

    out_c = np.full((T,), np.nan, dtype=np.float32)

    delta = np.abs(states.reshape(-1, 1) - states.reshape(1, -1))
    trans = (delta / 50.0).astype(np.float32)

    big = 1e9

    i = 0
    while i < T:
        if not is_voiced[i]:
            i += 1
            continue

        start = i
        while i < T and is_voiced[i]:
            i += 1
        end = i

        seg = obs_c[start:end]
        L = int(seg.shape[0])
        if L <= 0:
            continue

        dp = np.full((L, S), big, dtype=np.float32)
        back = np.full((L, S), -1, dtype=np.int32)

        emit0 = (np.abs(states - float(seg[0])) / 25.0).astype(np.float32)
        dp[0, :] = emit0

        for j in range(1, L):
            emit = (np.abs(states - float(seg[j])) / 25.0).astype(np.float32)
            prev = dp[j - 1, :]

            costs = prev.reshape(1, -1) + trans
            best_prev = np.argmin(costs, axis=1)
            dp[j, :] = emit + costs[np.arange(S), best_prev]
            back[j, :] = best_prev.astype(np.int32)

        s = int(np.argmin(dp[L - 1, :]))
        path = np.full((L,), -1, dtype=np.int32)
        for j in range(L - 1, -1, -1):
            path[j] = s
            s2 = int(back[j, s])
            if s2 < 0:
                break
            s = s2

        out_c[start:end] = states[path]

    return cents_to_hz(out_c)


def post_filter_energy_and_short_runs(
    y: np.ndarray,
    sr: int,
    hop_s: float,
    f0_hz: np.ndarray,
    *,
    energy_gate_db: float = -40.0,
    min_voiced_run_s: float = 0.15,
) -> np.ndarray:
    f0 = np.asarray(f0_hz, dtype=np.float32).copy()

    try:
        import librosa
    except Exception:
        return f0

    hop_length = max(1, int(round(hop_s * sr)))

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length, center=True).squeeze(0)
    rms = np.asarray(rms, dtype=np.float32)

    n = min(int(rms.shape[0]), int(f0.shape[0]))
    if n <= 0:
        return f0

    rms = rms[:n]
    f0 = f0[:n]

    db = librosa.amplitude_to_db(rms + 1e-12, ref=np.max)
    quiet = db < float(energy_gate_db)

    before_voiced = int(np.count_nonzero(np.isfinite(f0) & (f0 > 0)))
    f0[quiet] = np.nan
    after_energy_voiced = int(np.count_nonzero(np.isfinite(f0) & (f0 > 0)))

    min_run = max(1, int(round(float(min_voiced_run_s) / float(hop_s))))

    is_voiced = np.isfinite(f0) & (f0 > 0)
    i = 0
    removed_runs = 0
    while i < n:
        if not is_voiced[i]:
            i += 1
            continue
        j = i
        while j < n and is_voiced[j]:
            j += 1
        if (j - i) < min_run:
            f0[i:j] = np.nan
            removed_runs += 1
        i = j

    after_run_voiced = int(np.count_nonzero(np.isfinite(f0) & (f0 > 0)))

    if _dbg_enabled():
        print(
            "[pitch_guide_builder] post_filter: "
            f"energy_gate_db={energy_gate_db:.1f} min_run={min_run} "
            f"voiced_before={before_voiced} voiced_after_energy={after_energy_voiced} "
            f"voiced_after_runs={after_run_voiced} removed_runs={removed_runs}"
        )

    return f0


def trim_wrong_harmonic_onsets(
    f0_hz: np.ndarray,
    hop_s: float,
    *,
    onset_dev_cents: float = 250.0,
    stable_window_s: float = 0.30,
    onset_hold_frames: int = 5,
) -> np.ndarray:
    f0 = np.asarray(f0_hz, dtype=np.float32).copy()
    cents = hz_to_cents(f0)
    voiced = np.isfinite(cents)

    n = int(f0.shape[0])
    if n == 0:
        return f0

    stable_win = max(1, int(round(float(stable_window_s) / float(hop_s))))
    hold = max(1, int(onset_hold_frames))

    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue

        start = i
        j = i
        while j < n and voiced[j]:
            j += 1
        end = j

        L = end - start
        if L >= (stable_win + hold):
            tail_start = max(start, end - stable_win)
            tail = cents[tail_start:end]
            tail = tail[np.isfinite(tail)]
            if tail.size:
                stable_c = float(np.nanmedian(tail))

                k = start
                consec_ok = 0
                trim_limit = min(end, start + stable_win)
                while k < trim_limit:
                    if not np.isfinite(cents[k]):
                        consec_ok = 0
                        k += 1
                        continue

                    if abs(float(cents[k]) - stable_c) <= float(onset_dev_cents):
                        consec_ok += 1
                        if consec_ok >= hold:
                            break
                    else:
                        f0[k] = np.nan
                        cents[k] = np.nan
                        consec_ok = 0
                    k += 1

        i = end

    return f0
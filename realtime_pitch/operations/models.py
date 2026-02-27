from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from operations.aubio_yin import AubioYinPitcher
from operations.bacf import bacf_pitch
from operations.dsp import clamp, hann, rms
from operations.melodia import melodia_pitch
from operations.mpm import mpm_pitch
from operations.obp import obp_pitch
from operations.parselmouth_pitch import ParselmouthPitcher
from operations.pyworld_pitch import PyworldPitcher
from operations.swipe import swipe_pitch
from operations.yin import yin_pitch
from operations.yinfft import YinFftPitcher

ENGINE_REV = "realtime-pitch-2026-02-20-mpm-r5"
MPM_PIPELINE_REV = "mpm-isolated-dp-v3"
LAST_REVISION_MX = "2026-02-20 14:25:00 America/Mexico_City"


@dataclass
class PitchResult:
    ok: bool
    error: Optional[str]
    f0_hz: Optional[float]
    confidence: float
    debug: Optional[dict[str, float]] = None

    @property
    def note(self) -> Optional[str]:
        if not self.ok or self.f0_hz is None:
            return None
        return hz_to_note_name(self.f0_hz)


def hz_to_midi(f0_hz: float) -> float:
    return 69.0 + 12.0 * math.log2(max(f0_hz, 1e-12) / 440.0)


def midi_to_hz(m: float) -> float:
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


def midi_to_note_name(m: float) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    mi = int(round(m))
    name = names[mi % 12]
    octv = (mi // 12) - 1
    return f"{name}{octv}"


def hz_to_note_name(f0_hz: float) -> str:
    return midi_to_note_name(hz_to_midi(f0_hz))


def _pitch_distance_semitones(a_hz: float, b_hz: float) -> float:
    a = max(float(a_hz), 1e-6)
    b = max(float(b_hz), 1e-6)
    return abs(12.0 * math.log2(a / b))


def _closest_octave_candidate(raw_hz: float, prev_hz: float, lo_hz: float, hi_hz: float) -> float:
    candidates: list[float] = []
    for mul in (0.5, 1.0, 2.0):
        c = float(raw_hz) * mul
        if lo_hz <= c <= hi_hz:
            candidates.append(c)
    if not candidates:
        return float(raw_hz)

    prev = max(float(prev_hz), 1e-6)
    return min(candidates, key=lambda c: abs(math.log2(max(c, 1e-6) / prev)))


@dataclass
class PreprocessConfig:
    hp_cut_hz: float = 5.0
    lp_cut_hz: float = 5000.0
    enable_dc_block: bool = True
    enable_preemphasis: bool = False
    preemph: float = 0.97


class SimpleIIR:
    """Very small 1st-order IIR low/high pass for realtime use."""

    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.z = 0.0

    def reset(self) -> None:
        self.z = 0.0

    def lowpass(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x, dtype=np.float32)
        a = self.alpha
        z = self.z
        for i, xi in enumerate(x.astype(np.float32, copy=False)):
            z = a * xi + (1.0 - a) * z
            y[i] = z
        self.z = float(z)
        return y

    def highpass(self, x: np.ndarray) -> np.ndarray:
        lp = self.lowpass(x)
        return (x.astype(np.float32, copy=False) - lp).astype(np.float32, copy=False)


class Preprocessor:
    def __init__(self, sr: int, cfg: PreprocessConfig):
        self.sr = int(sr)
        self.cfg = cfg

        def alpha(fc: float) -> float:
            fc = float(max(0.0, fc))
            if fc <= 0.0:
                return 0.0
            return float(1.0 - math.exp(-2.0 * math.pi * fc / self.sr))

        self.hp = SimpleIIR(alpha(cfg.hp_cut_hz))
        self.lp = SimpleIIR(alpha(cfg.lp_cut_hz))
        self._prev = 0.0

    def reset(self) -> None:
        self.hp.reset()
        self.lp.reset()
        self._prev = 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)

        if self.cfg.enable_dc_block:
            r = 0.9995
            y = np.empty_like(x)
            prev_x = self._prev
            prev_y = 0.0
            for i, xi in enumerate(x):
                yi = xi - prev_x + r * prev_y
                y[i] = yi
                prev_x = float(xi)
                prev_y = float(yi)
            self._prev = prev_x
            x = y

        if self.cfg.hp_cut_hz and self.cfg.hp_cut_hz > 0:
            x = self.hp.highpass(x)
        x = self.lp.lowpass(x)

        if self.cfg.enable_preemphasis:
            a = float(self.cfg.preemph)
            y = np.empty_like(x)
            prev = 0.0
            for i, xi in enumerate(x):
                y[i] = xi - a * prev
                prev = float(xi)
            x = y

        return x.astype(np.float32, copy=False)


@dataclass
class YinPostProcessConfig:
    min_rms: float = 0.0025
    min_f0_hz: float = 40.0
    max_f0_hz: float = 2000.0
    median_win: int = 5
    ema_alpha: float = 0.30
    min_conf: float = 0.35
    hold_frames: int = 2
    octave_lock_conf: float = 0.80
    octave_switch_conf: float = 0.92
    octave_switch_frames: int = 2
    max_jump_semitones: float = 10.5
    jump_lock_conf: float = 0.90
    jitter_deadband_semitones: float = 0.18
    transition_min_semitones: float = 0.85
    transition_frames: int = 3


class YinPostProcessor:
    def __init__(self, cfg: YinPostProcessConfig):
        self.cfg = cfg
        self._hist: list[float] = []
        self._ema: Optional[float] = None
        self._last_good: Optional[float] = None
        self._hold_left: int = 0
        self._octave_pending: Optional[float] = None
        self._octave_pending_frames: int = 0
        self._transition_target: Optional[float] = None
        self._transition_left: int = 0

    def reset(self) -> None:
        self._hist.clear()
        self._ema = None
        self._last_good = None
        self._hold_left = 0
        self._octave_pending = None
        self._octave_pending_frames = 0
        self._transition_target = None
        self._transition_left = 0

    def _median(self) -> Optional[float]:
        if not self._hist:
            return None
        w = self.cfg.median_win
        xs = self._hist[-w:]
        xs = sorted(xs)
        mid = len(xs) // 2
        return float(xs[mid])

    def _stabilize_octave(self, f0_hz: float, conf: float) -> float:
        if self._last_good is None:
            return float(f0_hz)

        prev = float(self._last_good)
        cand = _closest_octave_candidate(
            raw_hz=float(f0_hz),
            prev_hz=prev,
            lo_hz=float(self.cfg.min_f0_hz),
            hi_hz=float(self.cfg.max_f0_hz),
        )
        jump_st = _pitch_distance_semitones(cand, prev)
        is_octave_jump = 9.5 <= jump_st <= 14.5

        if jump_st >= float(self.cfg.max_jump_semitones) and conf < float(self.cfg.jump_lock_conf):
            self._octave_pending = None
            self._octave_pending_frames = 0
            return prev

        if not is_octave_jump:
            self._octave_pending = None
            self._octave_pending_frames = 0
            return cand

        if conf < float(self.cfg.octave_lock_conf):
            self._octave_pending = None
            self._octave_pending_frames = 0
            return prev

        if self._octave_pending is not None and _pitch_distance_semitones(self._octave_pending, cand) < 0.5:
            self._octave_pending_frames += 1
        else:
            self._octave_pending = cand
            self._octave_pending_frames = 1

        if conf >= float(self.cfg.octave_switch_conf) or self._octave_pending_frames >= int(self.cfg.octave_switch_frames):
            self._octave_pending = None
            self._octave_pending_frames = 0
            return cand

        return prev

    def _apply_transition(self, candidate_hz: float) -> float:
        if self._last_good is None:
            self._transition_target = None
            self._transition_left = 0
            return float(candidate_hz)

        prev = float(self._last_good)
        jump_st = _pitch_distance_semitones(candidate_hz, prev)
        if jump_st < float(self.cfg.transition_min_semitones):
            self._transition_target = None
            self._transition_left = 0
            return float(candidate_hz)

        frames = max(1, int(self.cfg.transition_frames))
        if frames <= 1:
            self._transition_target = None
            self._transition_left = 0
            return float(candidate_hz)

        if (
            self._transition_target is None
            or self._transition_left <= 0
            or _pitch_distance_semitones(candidate_hz, self._transition_target) > 0.45
        ):
            self._transition_target = float(candidate_hz)
            self._transition_left = frames

        target = float(self._transition_target)
        left = max(1, int(self._transition_left))
        out = prev + (target - prev) / float(left)
        self._transition_left -= 1
        if self._transition_left <= 0:
            self._transition_target = None
            self._transition_left = 0
            return target
        return float(out)

    def process(self, raw_f0: Optional[float], conf: float, x: np.ndarray) -> PitchResult:
        if rms(x) < self.cfg.min_rms:
            self._hold_left = 0
            return PitchResult(ok=False, error="low_rms", f0_hz=None, confidence=0.0)

        if raw_f0 is None:
            if self._last_good is not None and self._hold_left > 0:
                self._hold_left -= 1
                return PitchResult(ok=True, error=None, f0_hz=float(self._last_good), confidence=float(conf))
            return PitchResult(ok=False, error="out_of_range", f0_hz=None, confidence=float(conf))

        if not (self.cfg.min_f0_hz <= raw_f0 <= self.cfg.max_f0_hz):
            return PitchResult(ok=False, error="out_of_range", f0_hz=None, confidence=float(conf))

        if conf < self.cfg.min_conf:
            if self._last_good is not None and self._hold_left > 0:
                self._hold_left -= 1
                return PitchResult(ok=True, error=None, f0_hz=float(self._last_good), confidence=float(conf))
            return PitchResult(ok=False, error="low_conf", f0_hz=None, confidence=float(conf))

        f0 = self._stabilize_octave(float(raw_f0), float(conf))

        self._hist.append(float(f0))
        med = self._median()
        if med is None:
            med = float(f0)

        if self._last_good is not None and _pitch_distance_semitones(float(med), float(self._last_good)) <= float(
            self.cfg.jitter_deadband_semitones
        ):
            med = float(self._last_good)

        jump_to_med = (
            _pitch_distance_semitones(float(med), float(self._last_good))
            if self._last_good is not None
            else 0.0
        )
        if jump_to_med < float(self.cfg.transition_min_semitones):
            if self._ema is None:
                self._ema = float(med)
            else:
                a = float(self.cfg.ema_alpha)
                self._ema = a * float(med) + (1.0 - a) * float(self._ema)
            candidate = float(self._ema)
        else:
            self._ema = float(med)
            candidate = float(med)

        out_f0 = self._apply_transition(candidate)
        self._last_good = out_f0
        self._hold_left = int(self.cfg.hold_frames)
        return PitchResult(ok=True, error=None, f0_hz=out_f0, confidence=float(conf))


@dataclass
class AdaptiveConf:
    enabled: bool = False
    win: int = 200


class ConfNormalizer:
    def __init__(self, cfg: AdaptiveConf):
        self.cfg = cfg
        self._buf: list[float] = []

    def reset(self) -> None:
        self._buf.clear()

    def update(self, conf: float) -> float:
        if not self.cfg.enabled:
            return float(conf)

        c = float(clamp(conf, 0.0, 0.999))
        self._buf.append(c)
        if len(self._buf) > self.cfg.win:
            self._buf = self._buf[-self.cfg.win :]

        xs = np.array(self._buf, dtype=np.float32)
        lo = float(np.quantile(xs, 0.05))
        hi = float(np.quantile(xs, 0.95))
        if hi - lo < 1e-6:
            return float(clamp(c, 0.0, 0.999))
        a = (c - lo) / (hi - lo)
        return float(clamp(a, 0.0, 0.999))


@dataclass
class MpmPostProcessConfig:
    min_rms: float = 0.0025
    min_f0_hz: float = 40.0
    max_f0_hz: float = 2000.0
    median_win: int = 5
    ema_alpha: float = 0.30
    min_conf: float = 0.25
    octave_lock_conf: float = 0.68
    octave_switch_conf: float = 0.82
    octave_switch_frames: int = 1
    max_jump_semitones: float = 10.5
    jump_lock_conf: float = 0.84
    hold_frames: int = 4
    jitter_deadband_semitones: float = 0.16
    transition_min_semitones: float = 0.80
    transition_frames: int = 3


@dataclass
class AuxPostProcessConfig:
    min_rms: float = 0.0025
    min_f0_hz: float = 40.0
    max_f0_hz: float = 2000.0
    median_win: int = 5
    ema_alpha: float = 0.30
    min_conf: float = 0.25
    octave_lock_conf: float = 0.68
    octave_switch_conf: float = 0.82
    octave_switch_frames: int = 1
    max_jump_semitones: float = 10.5
    jump_lock_conf: float = 0.84
    hold_frames: int = 4
    jitter_deadband_semitones: float = 0.16
    transition_min_semitones: float = 0.80
    transition_frames: int = 3


@dataclass
class MelodiaPostProcessConfig:
    min_rms: float = 0.0020
    min_f0_hz: float = 40.0
    max_f0_hz: float = 2000.0
    median_win: int = 7
    ema_alpha: float = 0.24
    min_conf: float = 0.24
    octave_lock_conf: float = 0.70
    octave_switch_conf: float = 0.82
    octave_switch_frames: int = 1
    max_jump_semitones: float = 10.0
    jump_lock_conf: float = 0.84
    hold_frames: int = 3
    jitter_deadband_semitones: float = 0.12
    transition_min_semitones: float = 0.70
    transition_frames: int = 2


@dataclass
class SwipePostProcessConfig:
    min_rms: float = 0.0022
    min_f0_hz: float = 40.0
    max_f0_hz: float = 2000.0
    median_win: int = 5
    ema_alpha: float = 0.28
    min_conf: float = 0.26
    octave_lock_conf: float = 0.68
    octave_switch_conf: float = 0.80
    octave_switch_frames: int = 1
    max_jump_semitones: float = 10.5
    jump_lock_conf: float = 0.84
    hold_frames: int = 3
    jitter_deadband_semitones: float = 0.16
    transition_min_semitones: float = 0.80
    transition_frames: int = 3


class _SharedPitchPostProcessor:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._hist: list[float] = []
        self._ema: Optional[float] = None
        self._last_good: Optional[float] = None
        self._hold_left: int = 0
        self._octave_pending: Optional[float] = None
        self._octave_pending_frames: int = 0
        self._transition_target: Optional[float] = None
        self._transition_left: int = 0

    def reset(self) -> None:
        self._hist.clear()
        self._ema = None
        self._last_good = None
        self._hold_left = 0
        self._octave_pending = None
        self._octave_pending_frames = 0
        self._transition_target = None
        self._transition_left = 0

    def _median(self) -> Optional[float]:
        if not self._hist:
            return None
        w = self.cfg.median_win
        xs = self._hist[-w:]
        xs = sorted(xs)
        return float(xs[len(xs) // 2])

    def _maybe_lock_octave(self, f0: float, conf: float) -> float:
        if self._last_good is None:
            return float(f0)

        prev = float(self._last_good)
        cand = _closest_octave_candidate(
            raw_hz=float(f0),
            prev_hz=prev,
            lo_hz=float(self.cfg.min_f0_hz),
            hi_hz=float(self.cfg.max_f0_hz),
        )
        jump_st = _pitch_distance_semitones(cand, prev)
        is_octave_jump = 9.5 <= jump_st <= 14.5

        if jump_st >= float(self.cfg.max_jump_semitones) and float(conf) < float(self.cfg.jump_lock_conf):
            self._octave_pending = None
            self._octave_pending_frames = 0
            return prev

        if not is_octave_jump:
            self._octave_pending = None
            self._octave_pending_frames = 0
            return cand

        if float(conf) < float(self.cfg.octave_lock_conf):
            self._octave_pending = None
            self._octave_pending_frames = 0
            return prev

        if self._octave_pending is not None and _pitch_distance_semitones(self._octave_pending, cand) < 0.5:
            self._octave_pending_frames += 1
        else:
            self._octave_pending = cand
            self._octave_pending_frames = 1

        if float(conf) >= float(self.cfg.octave_switch_conf) or self._octave_pending_frames >= int(self.cfg.octave_switch_frames):
            self._octave_pending = None
            self._octave_pending_frames = 0
            return cand

        return prev

    def _apply_transition(self, candidate_hz: float) -> float:
        if self._last_good is None:
            self._transition_target = None
            self._transition_left = 0
            return float(candidate_hz)

        prev = float(self._last_good)
        jump_st = _pitch_distance_semitones(candidate_hz, prev)
        if jump_st < float(self.cfg.transition_min_semitones):
            self._transition_target = None
            self._transition_left = 0
            return float(candidate_hz)

        frames = max(1, int(self.cfg.transition_frames))
        if frames <= 1:
            self._transition_target = None
            self._transition_left = 0
            return float(candidate_hz)

        if (
            self._transition_target is None
            or self._transition_left <= 0
            or _pitch_distance_semitones(candidate_hz, self._transition_target) > 0.45
        ):
            self._transition_target = float(candidate_hz)
            self._transition_left = frames

        target = float(self._transition_target)
        left = max(1, int(self._transition_left))
        out = prev + (target - prev) / float(left)
        self._transition_left -= 1
        if self._transition_left <= 0:
            self._transition_target = None
            self._transition_left = 0
            return target
        return float(out)

    def process(self, raw_f0: Optional[float], conf: float, x: np.ndarray) -> PitchResult:
        if rms(x) < self.cfg.min_rms:
            self._hold_left = 0
            return PitchResult(ok=False, error="low_rms", f0_hz=None, confidence=0.0)

        if raw_f0 is None:
            if self._last_good is not None and self._hold_left > 0:
                self._hold_left -= 1
                return PitchResult(ok=True, error=None, f0_hz=float(self._last_good), confidence=float(conf))
            return PitchResult(ok=False, error="out_of_range", f0_hz=None, confidence=float(conf))

        f0 = float(raw_f0)

        if not (self.cfg.min_f0_hz <= f0 <= self.cfg.max_f0_hz):
            return PitchResult(ok=False, error="out_of_range", f0_hz=None, confidence=float(conf))

        if float(conf) < float(self.cfg.min_conf):
            if self._last_good is not None and self._hold_left > 0:
                self._hold_left -= 1
                return PitchResult(ok=True, error=None, f0_hz=float(self._last_good), confidence=float(conf))
            return PitchResult(ok=False, error="low_conf", f0_hz=None, confidence=float(conf))

        f0 = self._maybe_lock_octave(f0, float(conf))

        self._hist.append(float(f0))
        med = self._median() or float(f0)

        if self._last_good is not None and _pitch_distance_semitones(float(med), float(self._last_good)) <= float(
            self.cfg.jitter_deadband_semitones
        ):
            med = float(self._last_good)

        jump_to_med = (
            _pitch_distance_semitones(float(med), float(self._last_good))
            if self._last_good is not None
            else 0.0
        )
        if jump_to_med < float(self.cfg.transition_min_semitones):
            if self._ema is None:
                self._ema = float(med)
            else:
                a = float(self.cfg.ema_alpha)
                self._ema = a * float(med) + (1.0 - a) * float(self._ema)
            candidate = float(self._ema)
        else:
            self._ema = float(med)
            candidate = float(med)

        out_f0 = self._apply_transition(candidate)
        self._last_good = out_f0
        self._hold_left = int(self.cfg.hold_frames)

        return PitchResult(ok=True, error=None, f0_hz=out_f0, confidence=float(conf))


class MpmIsolatedPostProcessor(_SharedPitchPostProcessor):
    pass


class AuxPostProcessor(_SharedPitchPostProcessor):
    pass


class MelodiaPostProcessor(_SharedPitchPostProcessor):
    pass


class SwipePostProcessor(_SharedPitchPostProcessor):
    pass


class AutoPostProcessor(_SharedPitchPostProcessor):
    pass


@dataclass
class VoicingConfig:
    enabled: bool = True
    min_conf_ratio: float = 0.76
    max_zcr: float = 0.22
    max_flatness: float = 0.55
    attack_frames: int = 2
    release_frames: int = 2


@dataclass
class TrackingConfig:
    enabled: bool = True
    max_jump_cents: float = 520.0
    high_conf: float = 0.80
    octave_jump_penalty: float = 180.0
    continuity_weight: float = 1.0
    confidence_bonus: float = 120.0
    smooth_cents: float = 120.0
    smooth_alpha: float = 0.30


@dataclass
class MpmOctaveFixConfig:
    enabled: bool = True
    hold_cents: float = 80.0
    confirm_frames: int = 3
    octave_confirm_frames: int = 4
    non_octave_confirm_frames: int = 2
    force_frames: int = 8
    high_conf: float = 0.70
    srh_margin: float = 1.28
    srh_margin_relaxed: float = 1.12
    octave_srh_margin: float = 1.45
    bias_ratio: float = 1.28
    bias_frames: int = 4
    stuck_frames: int = 12
    stuck_cents: float = 14.0
    stuck_alt_cents: float = 140.0
    relax_frames: int = 7
    dp_window_frames: int = 14
    dp_lookback_frames: int = 3
    dp_transition_sigma_cents: float = 140.0
    dp_octave_jump_penalty: float = 360.0
    dp_large_jump_penalty: float = 230.0
    dp_obs_srh_weight: float = 180.0
    dp_obs_raw_octave_penalty: float = 70.0
    non_octave_large_jump_semitones: float = 5.0
    non_octave_large_jump_confirm_frames: int = 3
    non_octave_large_jump_confirm_frames_high_conf: int = 2
    non_octave_large_jump_high_conf: float = 0.86


@dataclass
class EngineConfig:
    sr: int = 48000
    hop: int = 1024
    algo: str = "yin"
    fmin: float = 50.0
    fmax: float = 1000.0
    melodia_harmonics: int = 10
    melodia_fft_min: int = 8192
    pre: PreprocessConfig = field(default_factory=PreprocessConfig)
    mpm_pre: PreprocessConfig = field(default_factory=PreprocessConfig)
    yin_post: YinPostProcessConfig = field(default_factory=YinPostProcessConfig)
    yin_conf_norm: AdaptiveConf = field(default_factory=lambda: AdaptiveConf(enabled=False, win=200))
    mpm_post: MpmPostProcessConfig = field(default_factory=MpmPostProcessConfig)
    aux_post: AuxPostProcessConfig = field(default_factory=AuxPostProcessConfig)
    melodia_post: MelodiaPostProcessConfig = field(default_factory=MelodiaPostProcessConfig)
    swipe_post: SwipePostProcessConfig = field(default_factory=SwipePostProcessConfig)
    auto_post: SwipePostProcessConfig = field(default_factory=SwipePostProcessConfig)
    voicing: VoicingConfig = field(default_factory=VoicingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    mpm_octave_fix: MpmOctaveFixConfig = field(default_factory=MpmOctaveFixConfig)
    debug_octave: bool = False
    raw_pitch: bool = False
    octave_fix: bool = False
    octave_fix_confirm_frames: int = 2
    octave_fix_non_octave_confirm_frames: int = 2
    octave_fix_force_frames: int = 6
    octave_fix_high_conf: float = 0.66
    octave_fix_hold_cents: float = 90.0
    octave_fix_srh_margin: float = 1.35
    octave_fix_srh_margin_relaxed: float = 1.12
    octave_fix_srh_harmonics: int = 8
    octave_fix_srh_alpha: float = 0.5
    octave_fix_stuck_frames: int = 14
    octave_fix_stuck_cents: float = 15.0
    octave_fix_stuck_alt_cents: float = 150.0
    octave_fix_relax_frames: int = 8
    melodia_subharmonic_veto_ratio: float = 1.15
    melodia_subharmonic_veto_frames: int = 2
    melodia_conf_cap: float = 0.35
    melodia_conf_cap_ratio: float = 1.12


_SUPPORTED_ALGOS = {
    "yin",
    "yinfft",
    "aubio_yin",
    "obp",
    "bacf",
    "mpm",
    "pyworld",
    "parselmouth",
    "melodia",
    "swipe",
    "auto",
}
_STARTUP_UNLOCK_FRAMES: dict[str, int] = {
    "mpm": 12,
    "obp": 6,
    "bacf": 6,
    "pyworld": 0,
    "parselmouth": 0,
    "melodia": 0,
    "swipe": 12,
    "auto": 0,
    "yinfft": 0,
    "aubio_yin": 0,
}


class RealtimePitchState:
    """Realtime processing state for the websocket stream."""

    def __init__(self) -> None:
        self.cfg = EngineConfig()
        self.pre = Preprocessor(self.cfg.sr, self.cfg.pre)
        self.mpm_pre = Preprocessor(self.cfg.sr, self.cfg.mpm_pre)
        self.yin_post = YinPostProcessor(self.cfg.yin_post)
        self.yin_cn = ConfNormalizer(self.cfg.yin_conf_norm)
        self.mpm_post = MpmIsolatedPostProcessor(self.cfg.mpm_post)
        self.aux_post = AuxPostProcessor(self.cfg.aux_post)
        self.melodia_post = MelodiaPostProcessor(self.cfg.melodia_post)
        self.swipe_post = SwipePostProcessor(self.cfg.swipe_post)
        self.auto_post = AutoPostProcessor(self.cfg.auto_post)
        self._melodia_last_good: Optional[float] = None
        self._melodia_boot_candidate: Optional[float] = None
        self._melodia_boot_frames: int = 0
        self._melodia_jump_candidate: Optional[float] = None
        self._melodia_jump_frames: int = 0
        self._yinfft: Optional[YinFftPitcher] = None
        self._aubio_yin: Optional[AubioYinPitcher] = None
        self._pyworld: Optional[PyworldPitcher] = None
        self._parselmouth: Optional[ParselmouthPitcher] = None
        self._mpm_roll = np.zeros(0, dtype=np.float32)
        self._spec_roll = np.zeros(0, dtype=np.float32)
        self._mpm_spec_roll = np.zeros(0, dtype=np.float32)
        self._startup_unlock_left: dict[str, int] = dict(_STARTUP_UNLOCK_FRAMES)
        self._band_source: str = "manual"
        self._voiced_state: bool = False
        self._voiced_attack_count: int = 0
        self._voiced_release_count: int = 0
        self._track_prev_hz: Optional[float] = None
        self._rms_floor: float = 1e-4
        self._rms_floor_initialized: bool = False
        self._octave_prev_hz: Optional[float] = None
        self._octave_pending_hz: Optional[float] = None
        self._octave_pending_frames: int = 0
        self._octave_last_out_hz: Optional[float] = None
        self._octave_stuck_frames: int = 0
        self._octave_relax_left: int = 0
        self._octave_bias_up_frames: int = 0
        self._octave_bias_down_frames: int = 0
        self._melodia_subharm_up_frames: int = 0
        self._melodia_subharm_down_frames: int = 0
        self._raw_prev_hz: Optional[float] = None
        self._mpm_raw_prev_hz: Optional[float] = None
        self._mpm_octave_prev_hz: Optional[float] = None
        self._mpm_octave_pending_hz: Optional[float] = None
        self._mpm_octave_pending_frames: int = 0
        self._mpm_octave_last_out_hz: Optional[float] = None
        self._mpm_octave_stuck_frames: int = 0
        self._mpm_octave_relax_left: int = 0
        self._mpm_octave_bias_up_frames: int = 0
        self._mpm_octave_bias_down_frames: int = 0
        self._mpm_dp_candidates: list[list[float]] = []
        self._mpm_dp_costs: list[np.ndarray] = []
        self._mpm_dp_backptrs: list[np.ndarray] = []
        self._mpm_dp_srh_scores: list[list[float]] = []

    def _set_algo_min_conf(self, algo: str, value: float) -> None:
        if algo == "yin":
            self.cfg.yin_post.min_conf = value
        elif algo == "yinfft":
            self.cfg.yin_post.min_conf = value
        elif algo == "aubio_yin":
            self.cfg.yin_post.min_conf = value
        elif algo == "obp":
            self.cfg.aux_post.min_conf = value
        elif algo == "bacf":
            self.cfg.aux_post.min_conf = value
        elif algo == "mpm":
            self.cfg.mpm_post.min_conf = value
        elif algo == "pyworld":
            self.cfg.aux_post.min_conf = value
        elif algo == "parselmouth":
            self.cfg.aux_post.min_conf = value
        elif algo == "melodia":
            self.cfg.melodia_post.min_conf = value
        elif algo == "swipe":
            self.cfg.swipe_post.min_conf = value
        elif algo == "auto":
            self.cfg.auto_post.min_conf = value

    def _set_algo_smoothing(self, algo: str, smoothing: float) -> None:
        alpha = float(max(0.0, min(1.0, 1.0 - smoothing)))
        if algo == "yin":
            self.cfg.yin_post.ema_alpha = alpha
        elif algo == "yinfft":
            self.cfg.yin_post.ema_alpha = alpha
        elif algo == "aubio_yin":
            self.cfg.yin_post.ema_alpha = alpha
        elif algo == "obp":
            self.cfg.aux_post.ema_alpha = alpha
        elif algo == "bacf":
            self.cfg.aux_post.ema_alpha = alpha
        elif algo == "mpm":
            self.cfg.mpm_post.ema_alpha = alpha
        elif algo == "pyworld":
            self.cfg.aux_post.ema_alpha = alpha
        elif algo == "parselmouth":
            self.cfg.aux_post.ema_alpha = alpha
        elif algo == "melodia":
            self.cfg.melodia_post.ema_alpha = alpha
        elif algo == "swipe":
            self.cfg.swipe_post.ema_alpha = alpha
        elif algo == "auto":
            self.cfg.auto_post.ema_alpha = alpha

    def _update_post_ranges(self) -> None:
        self.cfg.yin_post.min_f0_hz = float(self.cfg.fmin)
        self.cfg.yin_post.max_f0_hz = float(self.cfg.fmax)
        self.cfg.mpm_post.min_f0_hz = float(self.cfg.fmin)
        self.cfg.mpm_post.max_f0_hz = float(self.cfg.fmax)
        self.cfg.aux_post.min_f0_hz = float(self.cfg.fmin)
        self.cfg.aux_post.max_f0_hz = float(self.cfg.fmax)
        self.cfg.melodia_post.min_f0_hz = float(self.cfg.fmin)
        self.cfg.melodia_post.max_f0_hz = float(self.cfg.fmax)
        self.cfg.swipe_post.min_f0_hz = float(self.cfg.fmin)
        self.cfg.swipe_post.max_f0_hz = float(self.cfg.fmax)
        self.cfg.auto_post.min_f0_hz = float(self.cfg.fmin)
        self.cfg.auto_post.max_f0_hz = float(self.cfg.fmax)

    def apply_config(self, cfg: dict[str, Any]) -> tuple[bool, Optional[str]]:
        try:
            if "method" in cfg:
                self.cfg.algo = str(cfg["method"])
            if "sample_rate" in cfg:
                self.cfg.sr = int(cfg["sample_rate"])
            if "fmin" in cfg:
                self.cfg.fmin = float(cfg["fmin"])
            if "fmax" in cfg:
                self.cfg.fmax = float(cfg["fmax"])
            if "melodia_harmonics" in cfg:
                self.cfg.melodia_harmonics = int(cfg["melodia_harmonics"])
            if "melodia_fft_min" in cfg:
                self.cfg.melodia_fft_min = int(cfg["melodia_fft_min"])

            if str(cfg.get("band_profile", "")).strip().lower() == "calibration":
                self.cfg.fmin = 90.0
                self.cfg.fmax = 220.0
                self._band_source = "calibration"
            else:
                voice_min = cfg.get("voice_type_min_hz")
                voice_max = cfg.get("voice_type_max_hz")
                if voice_min is not None and voice_max is not None:
                    vmin = float(voice_min)
                    vmax = float(voice_max)
                    if np.isfinite(vmin) and np.isfinite(vmax) and vmin > 0.0 and vmax > vmin:
                        self.cfg.fmin = vmin
                        self.cfg.fmax = vmax
                        self._band_source = "voice_type"
                elif "fmin" in cfg or "fmax" in cfg:
                    self._band_source = "manual"

            target_algo = self.cfg.algo
            melodia_voice_band = bool(cfg.get("melodia_voice_band", False))
            melodia_voice_kind = str(cfg.get("melodia_voice_kind", "")).strip().lower()
            voice_min = cfg.get("voice_type_min_hz")
            voice_max = cfg.get("voice_type_max_hz")
            if (
                melodia_voice_band
                and target_algo == "melodia"
                and voice_min is None
                and voice_max is None
            ):
                if melodia_voice_kind == "female":
                    self.cfg.fmin = 140.0
                    self.cfg.fmax = 1100.0
                    self._band_source = "melodia_voice"
                elif melodia_voice_kind == "male":
                    self.cfg.fmin = 80.0
                    self.cfg.fmax = 600.0
                    self._band_source = "melodia_voice"

            if "min_confidence" in cfg:
                self._set_algo_min_conf(target_algo, float(cfg["min_confidence"]))

            if "smoothing" in cfg:
                self._set_algo_smoothing(target_algo, float(cfg["smoothing"]))

            if "sr" in cfg:
                self.cfg.sr = int(cfg["sr"])
            if "hop" in cfg:
                self.cfg.hop = int(cfg["hop"])
            if "algo" in cfg:
                self.cfg.algo = str(cfg["algo"])
            if "debug_octave" in cfg:
                self.cfg.debug_octave = bool(cfg["debug_octave"])
            if "raw_pitch" in cfg:
                self.cfg.raw_pitch = bool(cfg["raw_pitch"])
            if "octave_fix" in cfg:
                self.cfg.octave_fix = bool(cfg["octave_fix"])
            # Baseline lock:
            # - MPM always runs with octave fix enabled.
            # - Raw bypass is disabled so MPM uses the calibrated full pipeline.
            # - On explicit method switches away from MPM, reset to neutral defaults.
            if self.cfg.algo == "mpm":
                self.cfg.raw_pitch = False
                self.cfg.octave_fix = True
            elif "method" in cfg or "algo" in cfg:
                self.cfg.raw_pitch = False
                self.cfg.octave_fix = False

            if "pre" in cfg and isinstance(cfg["pre"], dict):
                for k, v in cfg["pre"].items():
                    if hasattr(self.cfg.pre, k):
                        setattr(self.cfg.pre, k, v)

            if "mpm_pre" in cfg and isinstance(cfg["mpm_pre"], dict):
                for k, v in cfg["mpm_pre"].items():
                    if hasattr(self.cfg.mpm_pre, k):
                        setattr(self.cfg.mpm_pre, k, v)

            if "yin_post" in cfg and isinstance(cfg["yin_post"], dict):
                for k, v in cfg["yin_post"].items():
                    if hasattr(self.cfg.yin_post, k):
                        setattr(self.cfg.yin_post, k, v)

            if "yin_conf_norm" in cfg and isinstance(cfg["yin_conf_norm"], dict):
                for k, v in cfg["yin_conf_norm"].items():
                    if hasattr(self.cfg.yin_conf_norm, k):
                        setattr(self.cfg.yin_conf_norm, k, v)

            if "mpm_post" in cfg and isinstance(cfg["mpm_post"], dict):
                for k, v in cfg["mpm_post"].items():
                    if hasattr(self.cfg.mpm_post, k):
                        setattr(self.cfg.mpm_post, k, v)

            if "aux_post" in cfg and isinstance(cfg["aux_post"], dict):
                for k, v in cfg["aux_post"].items():
                    if hasattr(self.cfg.aux_post, k):
                        setattr(self.cfg.aux_post, k, v)

            if "melodia_post" in cfg and isinstance(cfg["melodia_post"], dict):
                for k, v in cfg["melodia_post"].items():
                    if hasattr(self.cfg.melodia_post, k):
                        setattr(self.cfg.melodia_post, k, v)

            if "swipe_post" in cfg and isinstance(cfg["swipe_post"], dict):
                for k, v in cfg["swipe_post"].items():
                    if hasattr(self.cfg.swipe_post, k):
                        setattr(self.cfg.swipe_post, k, v)

            if "auto_post" in cfg and isinstance(cfg["auto_post"], dict):
                for k, v in cfg["auto_post"].items():
                    if hasattr(self.cfg.auto_post, k):
                        setattr(self.cfg.auto_post, k, v)

            if "voicing" in cfg and isinstance(cfg["voicing"], dict):
                for k, v in cfg["voicing"].items():
                    if hasattr(self.cfg.voicing, k):
                        setattr(self.cfg.voicing, k, v)

            if "tracking" in cfg and isinstance(cfg["tracking"], dict):
                for k, v in cfg["tracking"].items():
                    if hasattr(self.cfg.tracking, k):
                        setattr(self.cfg.tracking, k, v)

            if "mpm_octave_fix" in cfg and isinstance(cfg["mpm_octave_fix"], dict):
                for k, v in cfg["mpm_octave_fix"].items():
                    if hasattr(self.cfg.mpm_octave_fix, k):
                        setattr(self.cfg.mpm_octave_fix, k, v)

            if self.cfg.algo not in _SUPPORTED_ALGOS:
                raise ValueError(f"Unsupported method '{self.cfg.algo}'. Allowed: {', '.join(sorted(_SUPPORTED_ALGOS))}")

            if self.cfg.algo == "pyworld":
                _ = PyworldPitcher(
                    sr=int(self.cfg.sr),
                    hop_size=int(max(64, self.cfg.hop)),
                    fmin=float(self.cfg.fmin),
                    fmax=float(self.cfg.fmax),
                )
                self._pyworld = None

            self._update_post_ranges()

        except Exception as e:
            return False, str(e)

        self.pre = Preprocessor(self.cfg.sr, self.cfg.pre)
        self.mpm_pre = Preprocessor(self.cfg.sr, self.cfg.mpm_pre)
        self.yin_post = YinPostProcessor(self.cfg.yin_post)
        self.yin_cn = ConfNormalizer(self.cfg.yin_conf_norm)
        self.mpm_post = MpmIsolatedPostProcessor(self.cfg.mpm_post)
        self.aux_post = AuxPostProcessor(self.cfg.aux_post)
        self.melodia_post = MelodiaPostProcessor(self.cfg.melodia_post)
        self.swipe_post = SwipePostProcessor(self.cfg.swipe_post)
        self.auto_post = AutoPostProcessor(self.cfg.auto_post)
        self._melodia_last_good = None
        self._melodia_boot_candidate = None
        self._melodia_boot_frames = 0
        self._melodia_jump_candidate = None
        self._melodia_jump_frames = 0
        self._yinfft = None
        self._aubio_yin = None
        self._pyworld = None
        self._parselmouth = None
        self._mpm_roll = np.zeros(0, dtype=np.float32)
        self._spec_roll = np.zeros(0, dtype=np.float32)
        self._mpm_spec_roll = np.zeros(0, dtype=np.float32)
        self._startup_unlock_left = dict(_STARTUP_UNLOCK_FRAMES)
        self._voiced_state = False
        self._voiced_attack_count = 0
        self._voiced_release_count = 0
        self._track_prev_hz = None
        self._rms_floor = 1e-4
        self._rms_floor_initialized = False
        self._octave_prev_hz = None
        self._octave_pending_hz = None
        self._octave_pending_frames = 0
        self._octave_last_out_hz = None
        self._octave_stuck_frames = 0
        self._octave_relax_left = 0
        self._octave_bias_up_frames = 0
        self._octave_bias_down_frames = 0
        self._melodia_subharm_up_frames = 0
        self._melodia_subharm_down_frames = 0
        self._raw_prev_hz = None
        self._mpm_raw_prev_hz = None
        self._mpm_octave_prev_hz = None
        self._mpm_octave_pending_hz = None
        self._mpm_octave_pending_frames = 0
        self._mpm_octave_last_out_hz = None
        self._mpm_octave_stuck_frames = 0
        self._mpm_octave_relax_left = 0
        self._mpm_octave_bias_up_frames = 0
        self._mpm_octave_bias_down_frames = 0
        self._mpm_dp_candidates = []
        self._mpm_dp_costs = []
        self._mpm_dp_backptrs = []
        self._mpm_dp_srh_scores = []

        return True, None

    def config_ack_payload(self) -> dict[str, Any]:
        return {
            "method": str(self.cfg.algo),
            "fmin": float(self.cfg.fmin),
            "fmax": float(self.cfg.fmax),
            "band_source": str(self._band_source),
            "raw_pitch": bool(self.cfg.raw_pitch),
            "octave_fix": bool(self.cfg.octave_fix),
            "mpm_baseline_locked": bool(self.cfg.algo == "mpm"),
            "engine_rev": ENGINE_REV,
            "mpm_pipeline_rev": MPM_PIPELINE_REV,
            "mpm_isolated": True,
            "last_revision": LAST_REVISION_MX,
        }

    def _use_startup_unlock(self, algo: str) -> bool:
        left = int(self._startup_unlock_left.get(algo, 0))
        if left <= 0:
            return False
        self._startup_unlock_left[algo] = left - 1
        return True

    def _startup_raw_result(self, f0: Optional[float], conf: float, min_conf: float) -> PitchResult:
        if f0 is None:
            return PitchResult(ok=False, error="out_of_range", f0_hz=None, confidence=float(conf))
        f = float(f0)
        if not (float(self.cfg.fmin) <= f <= float(self.cfg.fmax)):
            return PitchResult(ok=False, error="out_of_range", f0_hz=None, confidence=float(conf))
        conf_floor = max(0.05, float(min_conf) * 0.55)
        if float(conf) < conf_floor:
            return PitchResult(ok=False, error="low_conf", f0_hz=None, confidence=float(conf))
        return PitchResult(ok=True, error=None, f0_hz=f, confidence=float(conf))

    def _algo_post_cfg(self, algo: str) -> Any:
        if algo == "yin":
            return self.cfg.yin_post
        if algo == "yinfft":
            return self.cfg.yin_post
        if algo == "aubio_yin":
            return self.cfg.yin_post
        if algo == "mpm":
            return self.cfg.mpm_post
        if algo in {"obp", "bacf", "pyworld", "parselmouth"}:
            return self.cfg.aux_post
        if algo == "melodia":
            return self.cfg.melodia_post
        if algo == "swipe":
            return self.cfg.swipe_post
        return self.cfg.auto_post

    def _frame_voicing_metrics(self, x: np.ndarray) -> tuple[float, float, float]:
        frame_rms = float(rms(x))
        if x.size < 8:
            return frame_rms, 0.0, 1.0
        signs = x >= 0.0
        zcr = float(np.mean(signs[1:] != signs[:-1]))
        if frame_rms < 1e-8:
            return frame_rms, zcr, 1.0

        n = int(x.size)
        xw = np.asarray(x, dtype=np.float32) * hann(n)
        mag = np.abs(np.fft.rfft(xw, n=n)).astype(np.float32) + 1e-9
        freqs = np.fft.rfftfreq(n, d=1.0 / float(self.cfg.sr))
        band_hi = min(4000.0, float(self.cfg.sr) * 0.5)
        mask = (freqs >= 60.0) & (freqs <= band_hi)
        band = mag[mask] if np.any(mask) else mag
        geo = float(np.exp(np.mean(np.log(band))))
        ari = float(np.mean(band))
        flatness = geo / max(ari, 1e-9)
        return frame_rms, zcr, float(flatness)

    def _update_rms_floor(self, frame_rms: float, voiced_candidate: bool) -> None:
        val = float(max(frame_rms, 1e-7))
        if not self._rms_floor_initialized:
            self._rms_floor = val
            self._rms_floor_initialized = True
            return
        if voiced_candidate:
            # Keep floor stable while voiced so long notes do not raise the noise model.
            self._rms_floor = float(max(self._rms_floor * 0.999, 1e-7))
            return
        alpha = 0.03 if val > self._rms_floor else 0.01
        self._rms_floor = float((1.0 - alpha) * self._rms_floor + alpha * val)

    def _is_voiced_frame(self, x: np.ndarray, conf: float, algo: str) -> tuple[bool, dict[str, float]]:
        frame_rms, zcr, flatness = self._frame_voicing_metrics(x)
        post_cfg = self._algo_post_cfg(algo)
        min_rms = float(getattr(post_cfg, "min_rms", 0.0025))
        min_conf = float(getattr(post_cfg, "min_conf", 0.25))

        if not self.cfg.voicing.enabled:
            return True, {"rms": frame_rms, "zcr": zcr, "flatness": flatness}

        conf_ratio = float(self.cfg.voicing.min_conf_ratio)
        # Per-method confidence ratio: melodia is noisier in broadband vocal frames.
        ratio_by_algo = {
            "yin": max(0.72, conf_ratio),
            "yinfft": max(0.74, conf_ratio),
            "aubio_yin": max(0.74, conf_ratio),
            "obp": max(0.70, conf_ratio),
            "bacf": max(0.70, conf_ratio),
            "mpm": max(0.72, conf_ratio),
            "pyworld": max(0.68, conf_ratio),
            "parselmouth": max(0.70, conf_ratio),
            "swipe": max(0.78, conf_ratio),
            "melodia": max(0.84, conf_ratio),
            "auto": max(0.78, conf_ratio),
        }
        conf_ratio_eff = ratio_by_algo.get(algo, conf_ratio)
        conf_gate = float(conf) >= max(0.02, min_conf * conf_ratio_eff)
        rms_floor = float(self._rms_floor if self._rms_floor_initialized else frame_rms)
        if self._voiced_state:
            rms_gate = frame_rms >= max(min_rms * 0.72, rms_floor * 1.45)
            zcr_gate = zcr <= (float(self.cfg.voicing.max_zcr) * 1.2)
            flat_gate = flatness <= (float(self.cfg.voicing.max_flatness) * 1.18)
            conf_gate = float(conf) >= max(0.01, min_conf * conf_ratio_eff * 0.75)
        else:
            rms_gate = frame_rms >= max(min_rms, rms_floor * 2.15)
            zcr_gate = zcr <= float(self.cfg.voicing.max_zcr)
            flat_gate = flatness <= float(self.cfg.voicing.max_flatness)

        voiced_candidate = bool(rms_gate and zcr_gate and flat_gate and conf_gate)
        if voiced_candidate:
            self._voiced_attack_count += 1
            self._voiced_release_count = 0
        else:
            self._voiced_release_count += 1
            self._voiced_attack_count = 0

        if not self._voiced_state and self._voiced_attack_count >= max(1, int(self.cfg.voicing.attack_frames)):
            self._voiced_state = True
        if self._voiced_state and self._voiced_release_count >= max(1, int(self.cfg.voicing.release_frames)):
            self._voiced_state = False

        self._update_rms_floor(frame_rms, voiced_candidate)

        dbg = {
            "rms": frame_rms,
            "zcr": zcr,
            "flatness": flatness,
            "rms_floor": self._rms_floor,
            "voiced_candidate": 1.0 if voiced_candidate else 0.0,
            "voiced_state": 1.0 if self._voiced_state else 0.0,
        }
        return self._voiced_state, dbg

    def _track_pitch(self, f0_hz: float, conf: float) -> float:
        if not self.cfg.tracking.enabled:
            return float(f0_hz)

        f0 = float(f0_hz)
        lo = float(self.cfg.fmin)
        hi = float(self.cfg.fmax)
        prev = self._track_prev_hz

        cands: list[float] = []
        for mul in (0.5, 1.0, 2.0):
            cand = f0 * mul
            if lo <= cand <= hi:
                if not any(abs(12.0 * math.log2(max(cand, 1e-9) / max(x, 1e-9))) < 0.2 for x in cands):
                    cands.append(cand)
        if not cands:
            cands = [min(max(f0, lo), hi)]

        if prev is None:
            chosen = min(cands, key=lambda c: abs(12.0 * math.log2(max(c, 1e-9) / max(f0, 1e-9))))
            self._track_prev_hz = float(chosen)
            return float(chosen)

        prev_hz = float(prev)
        prev_cents = 1200.0 * math.log2(max(prev_hz, 1e-9))

        best_score = float("inf")
        best = cands[0]
        for cand in cands:
            cand_cents = 1200.0 * math.log2(max(cand, 1e-9))
            diff_prev = abs(cand_cents - prev_cents)
            continuity_cost = float(self.cfg.tracking.continuity_weight) * min(diff_prev, 900.0)
            octave_pen = float(self.cfg.tracking.octave_jump_penalty) if abs(diff_prev - 1200.0) < 180.0 else 0.0
            conf_bonus = float(self.cfg.tracking.confidence_bonus) * float(clamp(conf, 0.0, 1.0))
            score = continuity_cost + octave_pen - conf_bonus
            if score < best_score:
                best_score = score
                best = cand

        best_cents = 1200.0 * math.log2(max(best, 1e-9))
        diff = abs(best_cents - prev_cents)
        # Octave hops are almost always detector artifacts in this product context.
        if abs(diff - 1200.0) < 170.0 and float(conf) < max(0.88, float(self.cfg.tracking.high_conf) + 0.08):
            best = prev_hz
            best_cents = prev_cents
            diff = 0.0
        if diff > float(self.cfg.tracking.max_jump_cents) and float(conf) < float(self.cfg.tracking.high_conf):
            best = prev_hz
            best_cents = prev_cents
            diff = 0.0

        if diff < float(self.cfg.tracking.smooth_cents):
            alpha = float(clamp(self.cfg.tracking.smooth_alpha, 0.05, 0.95))
            smoothed_cents = (1.0 - alpha) * prev_cents + alpha * best_cents
            best = float(2.0 ** (smoothed_cents / 1200.0))

        self._track_prev_hz = float(best)
        return float(best)

    def _build_srh_bundle(self) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
        signal = self._spec_roll
        if signal.size < 256:
            return None
        n = int(signal.size)
        target_n = min(max(n, 1024), 8192)
        n_fft = 1
        while n_fft < target_n:
            n_fft <<= 1
        xw = signal[-n:].astype(np.float32, copy=False) * hann(n)
        spec = np.abs(np.fft.rfft(xw, n=n_fft)).astype(np.float32)
        if spec.size < 16:
            return None
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(self.cfg.sr)).astype(np.float32)
        nyq = float(freqs[-1])
        return freqs, spec, nyq

    def _build_mpm_srh_bundle(self) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
        signal = self._mpm_spec_roll
        if signal.size < 256:
            return None
        n = int(signal.size)
        target_n = min(max(n, 1024), 8192)
        n_fft = 1
        while n_fft < target_n:
            n_fft <<= 1
        xw = signal[-n:].astype(np.float32, copy=False) * hann(n)
        spec = np.abs(np.fft.rfft(xw, n=n_fft)).astype(np.float32)
        if spec.size < 16:
            return None
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(self.cfg.sr)).astype(np.float32)
        nyq = float(freqs[-1])
        return freqs, spec, nyq

    def _srh_score(self, bundle: Optional[tuple[np.ndarray, np.ndarray, float]], f0_hz: float) -> float:
        if bundle is None or not np.isfinite(f0_hz) or f0_hz <= 0.0:
            return 0.0
        freqs, spec, nyq = bundle
        f0 = float(f0_hz)
        if f0 <= float(freqs[1]) or f0 >= nyq:
            return 0.0
        score = 0.0
        k_max = max(1, int(self.cfg.octave_fix_srh_harmonics))
        alpha = float(clamp(self.cfg.octave_fix_srh_alpha, 0.0, 1.0))
        for k in range(1, k_max + 1):
            harmonic_hz = f0 * float(k)
            if harmonic_hz >= nyq:
                break
            weight = 1.0 / math.sqrt(float(k))
            harmonic_mag = float(np.interp(harmonic_hz, freqs, spec))
            between_hz = (float(k) + 0.5) * f0
            between_mag = float(np.interp(between_hz, freqs, spec)) if between_hz < nyq else 0.0
            score += weight * (harmonic_mag - alpha * between_mag)
        return float(max(score, 0.0))

    def _apply_melodia_safeguards(
        self,
        f0_hz: float,
        conf: float,
        lo: float,
        hi: float,
        bundle: Optional[tuple[np.ndarray, np.ndarray, float]],
    ) -> tuple[float, float, dict[str, float]]:
        raw = float(clamp(f0_hz, lo, hi))
        score_raw = self._srh_score(bundle, raw)
        up = raw * 2.0
        down = raw * 0.5
        score_up = self._srh_score(bundle, up) if up <= hi else 0.0
        score_down = self._srh_score(bundle, down) if down >= lo else 0.0

        ratio = float(max(1.01, self.cfg.melodia_subharmonic_veto_ratio))
        if up <= hi and score_up > ratio * max(score_raw, 1e-9):
            self._melodia_subharm_up_frames += 1
        else:
            self._melodia_subharm_up_frames = 0
        self._melodia_subharm_down_frames = 0

        promoted = 0.0
        demoted = 0.0
        veto_frames = max(1, int(self.cfg.melodia_subharmonic_veto_frames))
        if up <= hi and self._melodia_subharm_up_frames >= veto_frames:
            raw = up
            score_raw = score_up
            promoted = 1.0
            self._melodia_subharm_up_frames = 0
            self._melodia_subharm_down_frames = 0

        scores = [score_raw]
        if up <= hi:
            scores.append(score_up)
        if down >= lo:
            scores.append(score_down)
        capped = 0.0
        if len(scores) >= 2:
            sorted_scores = sorted(scores, reverse=True)
            top = float(sorted_scores[0])
            second = float(sorted_scores[1])
            if top <= float(self.cfg.melodia_conf_cap_ratio) * max(second, 1e-9):
                conf = min(float(conf), float(self.cfg.melodia_conf_cap))
                capped = 1.0

        debug = {
            "mel_srh_raw": float(score_raw),
            "mel_srh_up": float(score_up),
            "mel_srh_down": float(score_down),
            "mel_up_veto": promoted,
            "mel_down_veto": demoted,
            "mel_conf_capped": capped,
        }
        return float(raw), float(conf), debug

    def _mpm_dp_propose(
        self,
        raw_hz: float,
        conf: float,
        srh_bundle: Optional[tuple[np.ndarray, np.ndarray, float]],
    ) -> tuple[float, float, dict[str, float]]:
        cfg = self.cfg.mpm_octave_fix
        lo = float(self.cfg.fmin)
        hi = float(self.cfg.fmax)
        raw = float(clamp(raw_hz, lo, hi))
        raw_conf = float(clamp(conf, 0.0, 1.0))

        candidates: list[float] = []
        for mul in (0.5, 1.0, 2.0):
            candidate_hz = raw * mul
            if lo <= candidate_hz <= hi:
                if not any(abs(1200.0 * math.log2(max(candidate_hz, 1e-9) / max(x, 1e-9))) < 40.0 for x in candidates):
                    candidates.append(float(candidate_hz))
        if not candidates:
            candidates = [raw]

        scores = [float(self._srh_score(srh_bundle, cand)) for cand in candidates]
        max_score = max(scores) if scores else 0.0
        raw_cents = 1200.0 * math.log2(max(raw, 1e-9))

        obs = np.zeros(len(candidates), dtype=np.float64)
        srh_w = float(max(0.0, cfg.dp_obs_srh_weight))
        raw_oct_pen = float(max(0.0, cfg.dp_obs_raw_octave_penalty))
        for idx, cand in enumerate(candidates):
            cand_cents = 1200.0 * math.log2(max(cand, 1e-9))
            diff_raw = abs(cand_cents - raw_cents)
            srh_norm = float(scores[idx] / max(max_score, 1e-9)) if max_score > 0.0 else 0.0
            obs_cost = -srh_w * srh_norm
            if 840.0 <= diff_raw <= 1360.0:
                obs_cost += raw_oct_pen * raw_conf
            obs[idx] = obs_cost

        if not self._mpm_dp_candidates:
            curr_costs = obs.copy()
            back = np.arange(len(candidates), dtype=np.int32)
        else:
            prev_candidates = self._mpm_dp_candidates[-1]
            prev_costs = self._mpm_dp_costs[-1]
            sigma = float(max(40.0, cfg.dp_transition_sigma_cents))
            octave_pen = float(max(0.0, cfg.dp_octave_jump_penalty))
            large_pen = float(max(0.0, cfg.dp_large_jump_penalty))
            curr_costs = np.full(len(candidates), np.inf, dtype=np.float64)
            back = np.zeros(len(candidates), dtype=np.int32)
            for j, cand in enumerate(candidates):
                cand_cents = 1200.0 * math.log2(max(cand, 1e-9))
                best_cost = float("inf")
                best_i = 0
                for i, prev_cand in enumerate(prev_candidates):
                    prev_cents = 1200.0 * math.log2(max(float(prev_cand), 1e-9))
                    diff = abs(cand_cents - prev_cents)
                    transition_cost = ((diff / sigma) ** 2) * 24.0
                    if abs(diff - 1200.0) <= 240.0:
                        transition_cost += octave_pen
                    elif diff > 420.0:
                        transition_cost += large_pen
                    trial = float(prev_costs[i]) + float(transition_cost)
                    if trial < best_cost:
                        best_cost = trial
                        best_i = i
                curr_costs[j] = float(obs[j]) + best_cost
                back[j] = int(best_i)

        min_cost = float(np.min(curr_costs))
        if np.isfinite(min_cost):
            curr_costs = curr_costs - min_cost

        self._mpm_dp_candidates.append([float(c) for c in candidates])
        self._mpm_dp_costs.append(curr_costs.astype(np.float64, copy=False))
        self._mpm_dp_backptrs.append(back)
        self._mpm_dp_srh_scores.append([float(s) for s in scores])

        max_window = max(4, int(cfg.dp_window_frames))
        while len(self._mpm_dp_candidates) > max_window:
            self._mpm_dp_candidates.pop(0)
            self._mpm_dp_costs.pop(0)
            self._mpm_dp_backptrs.pop(0)
            self._mpm_dp_srh_scores.pop(0)

        lookback = int(max(0, cfg.dp_lookback_frames))
        lookback = min(lookback, len(self._mpm_dp_candidates) - 1)
        out_frame_idx = len(self._mpm_dp_candidates) - 1 - lookback

        best_state = int(np.argmin(self._mpm_dp_costs[-1]))
        for frame_idx in range(len(self._mpm_dp_candidates) - 1, out_frame_idx, -1):
            best_state = int(self._mpm_dp_backptrs[frame_idx][best_state])

        out_hz = float(self._mpm_dp_candidates[out_frame_idx][best_state])
        out_score = float(self._mpm_dp_srh_scores[out_frame_idx][best_state])

        latest_costs = self._mpm_dp_costs[-1]
        if latest_costs.size >= 2:
            sorted_idx = np.argsort(latest_costs)
            margin = float(latest_costs[sorted_idx[1]] - latest_costs[sorted_idx[0]])
        else:
            margin = 0.0

        debug = {
            "mpm_dp_states": float(len(candidates)),
            "mpm_dp_lookback": float(lookback),
            "mpm_dp_margin": float(margin),
            "mpm_dp_srh_norm": float(out_score / max(max_score, 1e-9)) if max_score > 0.0 else 0.0,
        }
        return out_hz, out_score, debug

    def _stabilize_octave_mpm(self, f0_hz: float, conf: float) -> tuple[float, dict[str, float]]:
        cfg = self.cfg.mpm_octave_fix
        lo = float(self.cfg.fmin)
        hi = float(self.cfg.fmax)
        raw = float(clamp(f0_hz, lo, hi))
        raw_conf = float(clamp(conf, 0.0, 1.0))
        prev = self._mpm_octave_prev_hz
        srh_bundle = self._build_mpm_srh_bundle()

        proposed, proposed_score, dp_dbg = self._mpm_dp_propose(raw, raw_conf, srh_bundle)

        if prev is None:
            self._mpm_octave_prev_hz = float(proposed)
            self._mpm_octave_pending_hz = None
            self._mpm_octave_pending_frames = 0
            self._mpm_octave_last_out_hz = float(proposed)
            self._mpm_octave_stuck_frames = 0
            self._mpm_octave_relax_left = 0
            self._mpm_octave_bias_up_frames = 0
            self._mpm_octave_bias_down_frames = 0
            out_dbg = {"mpm_oct_mode": 0.0, "mpm_oct_pending": 0.0, "mpm_oct_srh_ratio": 1.0}
            out_dbg.update(dp_dbg)
            return float(proposed), out_dbg

        prev_hz = float(prev)
        prev_score = float(self._srh_score(srh_bundle, prev_hz))
        proposed_cents = 1200.0 * math.log2(max(proposed, 1e-9))
        prev_cents = 1200.0 * math.log2(max(prev_hz, 1e-9))
        jump_cents = abs(proposed_cents - prev_cents)
        octave_jump = abs(jump_cents - 1200.0) <= 240.0

        hold_cents = float(max(30.0, cfg.hold_cents))
        if jump_cents <= hold_cents:
            self._mpm_octave_prev_hz = float(proposed)
            self._mpm_octave_pending_hz = None
            self._mpm_octave_pending_frames = 0
            self._mpm_octave_last_out_hz = float(proposed)
            self._mpm_octave_stuck_frames = 0
            out_dbg = {"mpm_oct_mode": 1.0, "mpm_oct_pending": 0.0, "mpm_oct_srh_ratio": 1.0}
            out_dbg.update(dp_dbg)
            return float(proposed), out_dbg

        if self._mpm_octave_pending_hz is not None:
            pending_cents = 1200.0 * math.log2(max(float(self._mpm_octave_pending_hz), 1e-9))
            if abs(pending_cents - proposed_cents) <= 70.0:
                self._mpm_octave_pending_frames += 1
            else:
                self._mpm_octave_pending_hz = float(proposed)
                self._mpm_octave_pending_frames = 1
        else:
            self._mpm_octave_pending_hz = float(proposed)
            self._mpm_octave_pending_frames = 1

        srh_ratio = (proposed_score + 1e-9) / (prev_score + 1e-9)
        srh_available = (proposed_score > 1e-8) or (prev_score > 1e-8)
        required_ratio = float(cfg.octave_srh_margin if octave_jump else cfg.srh_margin_relaxed)
        if not srh_available:
            required_ratio = 1.0

        required_frames = int(cfg.octave_confirm_frames if octave_jump else cfg.non_octave_confirm_frames)
        required_frames = max(1, required_frames)
        if octave_jump:
            required_conf = float(max(cfg.high_conf, 0.70))
        else:
            required_conf = float(max(0.50, cfg.high_conf - 0.16))

        jump_st = float(jump_cents / 100.0)
        large_non_octave_jump = (
            (not octave_jump) and jump_st >= float(max(0.0, cfg.non_octave_large_jump_semitones))
        )
        if large_non_octave_jump:
            high_conf_large = raw_conf >= float(cfg.non_octave_large_jump_high_conf)
            guard_frames = (
                int(cfg.non_octave_large_jump_confirm_frames_high_conf)
                if high_conf_large
                else int(cfg.non_octave_large_jump_confirm_frames)
            )
            required_frames = max(required_frames, max(2, guard_frames))

        can_accept = (
            self._mpm_octave_pending_frames >= required_frames
            and srh_ratio >= required_ratio
            and raw_conf >= required_conf
        )
        can_force = (
            self._mpm_octave_pending_frames >= max(required_frames + 2, int(cfg.force_frames))
            and raw_conf >= (required_conf * 0.92)
            and srh_ratio >= float(max(1.0, cfg.srh_margin_relaxed))
        )

        if can_accept or can_force:
            self._mpm_octave_prev_hz = float(proposed)
            self._mpm_octave_pending_hz = None
            self._mpm_octave_pending_frames = 0
            self._mpm_octave_last_out_hz = float(proposed)
            self._mpm_octave_stuck_frames = 0
            out_dbg = {
                "mpm_oct_mode": 7.0 if can_force else (2.0 if octave_jump else 4.0),
                "mpm_oct_pending": 0.0,
                "mpm_oct_srh_ratio": float(srh_ratio),
                "mpm_large_jump": 1.0 if large_non_octave_jump else 0.0,
                "mpm_large_jump_req": float(required_frames) if large_non_octave_jump else 0.0,
                "mpm_jump_st": float(jump_st),
            }
            out_dbg.update(dp_dbg)
            return float(proposed), out_dbg

        self._mpm_octave_prev_hz = float(prev_hz)
        self._mpm_octave_last_out_hz = float(prev_hz)
        out_dbg = {
            "mpm_oct_mode": 3.0 if octave_jump else 5.0,
            "mpm_oct_pending": float(self._mpm_octave_pending_frames),
            "mpm_oct_srh_ratio": float(srh_ratio),
            "mpm_large_jump": 1.0 if large_non_octave_jump else 0.0,
            "mpm_large_jump_req": float(required_frames) if large_non_octave_jump else 0.0,
            "mpm_jump_st": float(jump_st),
        }
        out_dbg.update(dp_dbg)
        return float(prev_hz), out_dbg

    def _stabilize_octave(
        self,
        algo: str,
        f0_hz: float,
        conf: float,
        octave_anchor_hz: Optional[float] = None,
    ) -> tuple[float, dict[str, float]]:
        lo = float(self.cfg.fmin)
        hi = float(self.cfg.fmax)
        raw = float(clamp(f0_hz, lo, hi))
        raw_conf = float(conf)
        prev = self._octave_prev_hz
        srh_bundle = self._build_srh_bundle()
        melodia_dbg: dict[str, float] = {}
        if algo == "melodia":
            raw, raw_conf, melodia_dbg = self._apply_melodia_safeguards(raw, raw_conf, lo, hi, srh_bundle)

        candidates: list[float] = []
        for mul in (0.5, 1.0, 2.0):
            candidate_hz = raw * mul
            if lo <= candidate_hz <= hi:
                if not any(abs(1200.0 * math.log2(max(candidate_hz, 1e-9) / max(x, 1e-9))) < 40.0 for x in candidates):
                    candidates.append(candidate_hz)
        if not candidates:
            candidates = [raw]

        if prev is None:
            chosen = min(candidates, key=lambda c: abs(1200.0 * math.log2(max(c, 1e-9) / max(raw, 1e-9))))
            if octave_anchor_hz is not None and np.isfinite(float(octave_anchor_hz)) and float(octave_anchor_hz) > 0.0:
                anchor = float(octave_anchor_hz)
                anchor_best = min(candidates, key=lambda c: abs(1200.0 * math.log2(max(c, 1e-9) / max(anchor, 1e-9))))
                anchor_dist = abs(1200.0 * math.log2(max(anchor_best, 1e-9) / max(anchor, 1e-9)))
                startup_anchor_cap = 320.0 if algo in {"yinfft", "aubio_yin"} else 240.0
                if anchor_dist <= startup_anchor_cap:
                    chosen = float(anchor_best)
            if len(candidates) > 1 and srh_bundle is not None:
                scored = [(cand, self._srh_score(srh_bundle, cand)) for cand in candidates]
                scored.sort(key=lambda item: item[1], reverse=True)
                best_cand, best_score = scored[0]
                second_score = scored[1][1] if len(scored) > 1 else 0.0
                startup_ratio = (best_score + 1e-9) / (second_score + 1e-9)
                if algo in {"yinfft", "aubio_yin"}:
                    startup_margin = 1.08
                elif algo in {"yin", "swipe"}:
                    startup_margin = 1.22
                else:
                    startup_margin = 1.08
                if best_score > 0.0 and startup_ratio >= startup_margin:
                    chosen = float(best_cand)
                    if algo in {"yinfft", "aubio_yin"} and len(scored) > 1:
                        hi_cand, hi_score = max(scored, key=lambda item: item[0])
                        if hi_score >= (0.93 * best_score):
                            chosen = float(hi_cand)
                elif algo == "melodia":
                    band_center_hz = math.sqrt(max(lo, 1e-9) * max(hi, 1e-9))
                    chosen = min(candidates, key=lambda c: abs(1200.0 * math.log2(max(c, 1e-9) / max(band_center_hz, 1e-9))))
            self._octave_prev_hz = float(chosen)
            self._octave_pending_hz = None
            self._octave_pending_frames = 0
            self._octave_last_out_hz = float(chosen)
            out_dbg = {"oct_mode": 0.0, "oct_pending": 0.0}
            out_dbg.update(melodia_dbg)
            return float(chosen), out_dbg

        prev_cents = 1200.0 * math.log2(max(prev, 1e-9))
        chosen = min(candidates, key=lambda c: abs((1200.0 * math.log2(max(c, 1e-9))) - prev_cents))
        chosen_cents = 1200.0 * math.log2(max(chosen, 1e-9))
        jump_cents = abs(chosen_cents - prev_cents)

        anchor_dist_for_jump: Optional[float] = None
        if octave_anchor_hz is not None and np.isfinite(float(octave_anchor_hz)) and float(octave_anchor_hz) > 0.0:
            anchor = float(octave_anchor_hz)
            anchor_best = min(candidates, key=lambda c: abs(1200.0 * math.log2(max(c, 1e-9) / max(anchor, 1e-9))))
            anchor_dist = abs(1200.0 * math.log2(max(anchor_best, 1e-9) / max(anchor, 1e-9)))
            chosen_dist = abs(1200.0 * math.log2(max(chosen, 1e-9) / max(anchor, 1e-9)))
            anchor_dist_for_jump = float(anchor_dist)
            # Use anchor only for true octave disambiguation; avoid note-to-note "freezing".
            if chosen_dist >= 700.0 and anchor_dist <= 220.0 and (chosen_dist - anchor_dist) >= 500.0:
                chosen = float(anchor_best)
                chosen_cents = 1200.0 * math.log2(max(chosen, 1e-9))
                jump_cents = abs(chosen_cents - prev_cents)
        prev_score = self._srh_score(srh_bundle, float(prev))
        chosen_score = self._srh_score(srh_bundle, float(chosen))

        hold_cents = float(max(30.0, self.cfg.octave_fix_hold_cents))
        octave_jump = abs(jump_cents - 1200.0) <= 240.0
        high_conf = float(raw_conf) >= float(self.cfg.octave_fix_high_conf)
        confirm_frames = max(1, int(self.cfg.octave_fix_confirm_frames))
        non_oct_confirm_frames = max(1, int(self.cfg.octave_fix_non_octave_confirm_frames))
        force_frames = max(confirm_frames + 1, int(self.cfg.octave_fix_force_frames))
        if algo in {"yin", "swipe"}:
            hold_cents = max(hold_cents, 120.0)
            confirm_frames += 2
            non_oct_confirm_frames += 2
            force_frames = max(force_frames, confirm_frames + 6)
        if algo in {"yinfft", "aubio_yin"}:
            hold_cents = max(hold_cents, 130.0)
            confirm_frames += 1
            non_oct_confirm_frames += 1
            force_frames = max(force_frames, confirm_frames + 4)
        margin = (
            float(self.cfg.octave_fix_srh_margin_relaxed)
            if self._octave_relax_left > 0
            else float(self.cfg.octave_fix_srh_margin)
        )
        if algo in {"yin", "swipe"}:
            margin = max(margin, 1.45)
        if algo in {"yinfft", "aubio_yin"}:
            margin = max(margin, 1.30)
        relaxed_margin = float(self.cfg.octave_fix_srh_margin_relaxed)
        if algo in {"yin", "swipe"}:
            relaxed_margin = max(relaxed_margin, 1.30)
        if algo in {"yinfft", "aubio_yin"}:
            relaxed_margin = max(relaxed_margin, 1.16)
        srh_ratio = (chosen_score + 1e-9) / (prev_score + 1e-9)
        srh_ok = srh_ratio >= margin
        if (
            algo in {"yinfft", "aubio_yin"}
            and octave_jump
            and anchor_dist_for_jump is not None
            and anchor_dist_for_jump <= 220.0
        ):
            srh_ok = True
            confirm_frames = max(1, confirm_frames - 1)

        anchor_jump_ok = True
        if octave_jump and octave_anchor_hz is not None and np.isfinite(float(octave_anchor_hz)) and float(octave_anchor_hz) > 0.0:
            anchor_cents = 1200.0 * math.log2(max(float(octave_anchor_hz), 1e-9))
            prev_anchor_dist = abs(prev_cents - anchor_cents)
            chosen_anchor_dist = abs(chosen_cents - anchor_cents)
            # For octave jumps, require clear agreement with the anchor estimate.
            if not (chosen_anchor_dist <= 220.0 or (prev_anchor_dist - chosen_anchor_dist) >= 280.0):
                anchor_jump_ok = False

        if jump_cents <= hold_cents:
            self._octave_prev_hz = float(chosen)
            self._octave_pending_hz = None
            self._octave_pending_frames = 0
            self._octave_last_out_hz = float(chosen)
            self._octave_stuck_frames = 0
            self._octave_relax_left = max(0, self._octave_relax_left - 1)
            self._octave_bias_up_frames = max(0, self._octave_bias_up_frames - 1)
            self._octave_bias_down_frames = max(0, self._octave_bias_down_frames - 1)
            out_dbg = {
                "oct_mode": 1.0,
                "oct_pending": 0.0,
                "oct_srh_ratio": float(srh_ratio),
                "oct_srh_prev": float(prev_score),
                "oct_srh_chosen": float(chosen_score),
            }
            if octave_anchor_hz is not None and np.isfinite(float(octave_anchor_hz)):
                out_dbg["oct_anchor_hz"] = float(octave_anchor_hz)
            out_dbg.update(melodia_dbg)
            return float(chosen), out_dbg

        # Persistent wrong-octave recovery:
        # if x2 or x0.5 keeps stronger SRH evidence for several frames,
        # force an octave migration even when continuity prefers the current octave.
        score_up = self._srh_score(srh_bundle, chosen * 2.0) if (chosen * 2.0) <= hi else 0.0
        score_down = self._srh_score(srh_bundle, chosen * 0.5) if (chosen * 0.5) >= lo else 0.0
        if algo in {"yinfft", "aubio_yin"}:
            bias_ratio = 1.12
            bias_frames = 3
        elif algo == "yin":
            bias_ratio = 1.45
            bias_frames = 10
        elif algo == "swipe":
            bias_ratio = 1.45
            bias_frames = 10
        elif algo == "melodia":
            bias_ratio = 1.40
            bias_frames = 9
        elif algo == "mpm":
            bias_ratio = 1.55
            bias_frames = 12
        else:  # auto
            bias_ratio = 1.50
            bias_frames = 10

        if score_up > bias_ratio * max(chosen_score, 1e-9):
            self._octave_bias_up_frames += 1
        else:
            self._octave_bias_up_frames = max(0, self._octave_bias_up_frames - 1)
        if score_down > bias_ratio * max(chosen_score, 1e-9):
            self._octave_bias_down_frames += 1
        else:
            self._octave_bias_down_frames = max(0, self._octave_bias_down_frames - 1)

        if self._octave_bias_up_frames >= bias_frames and (chosen * 2.0) <= hi:
            forced = float(chosen * 2.0)
            self._octave_prev_hz = forced
            self._octave_pending_hz = None
            self._octave_pending_frames = 0
            self._octave_last_out_hz = forced
            self._octave_stuck_frames = 0
            self._octave_relax_left = 0
            self._octave_bias_up_frames = 0
            self._octave_bias_down_frames = 0
            out_dbg = {
                "oct_mode": 8.0,
                "oct_pending": 0.0,
                "oct_srh_ratio": float((score_up + 1e-9) / max(chosen_score, 1e-9)),
                "oct_srh_prev": float(chosen_score),
                "oct_srh_chosen": float(score_up),
            }
            out_dbg.update(melodia_dbg)
            return forced, out_dbg

        if self._octave_bias_down_frames >= bias_frames and (chosen * 0.5) >= lo:
            forced = float(chosen * 0.5)
            self._octave_prev_hz = forced
            self._octave_pending_hz = None
            self._octave_pending_frames = 0
            self._octave_last_out_hz = forced
            self._octave_stuck_frames = 0
            self._octave_relax_left = 0
            self._octave_bias_up_frames = 0
            self._octave_bias_down_frames = 0
            out_dbg = {
                "oct_mode": 9.0,
                "oct_pending": 0.0,
                "oct_srh_ratio": float((score_down + 1e-9) / max(chosen_score, 1e-9)),
                "oct_srh_prev": float(chosen_score),
                "oct_srh_chosen": float(score_down),
            }
            out_dbg.update(melodia_dbg)
            return forced, out_dbg

        # Anti-freeze: if output stays locked while an octave alternative has better SRH, relax and try switching.
        if self._octave_last_out_hz is not None:
            last_out_cents = 1200.0 * math.log2(max(float(self._octave_last_out_hz), 1e-9))
            if abs(last_out_cents - chosen_cents) <= float(max(1.0, self.cfg.octave_fix_stuck_cents)):
                self._octave_stuck_frames += 1
            else:
                self._octave_stuck_frames = 0
        else:
            self._octave_stuck_frames = 0

        allow_stuck_relax = algo in {
            "obp",
            "bacf",
            "mpm",
            "pyworld",
            "parselmouth",
            "melodia",
            "auto",
            "yinfft",
            "aubio_yin",
        }
        if allow_stuck_relax and self._octave_stuck_frames >= max(2, int(self.cfg.octave_fix_stuck_frames)):
            min_alt = float(max(60.0, self.cfg.octave_fix_stuck_alt_cents))
            alt = None
            alt_score = -1.0
            for cand in candidates:
                cand_cents = 1200.0 * math.log2(max(cand, 1e-9))
                if abs(cand_cents - prev_cents) < min_alt:
                    continue
                sc = self._srh_score(srh_bundle, cand)
                if sc > alt_score:
                    alt = cand
                    alt_score = sc
            if alt is not None:
                alt_ratio = (alt_score + 1e-9) / (prev_score + 1e-9)
                if alt_ratio >= relaxed_margin:
                    chosen = float(alt)
                    chosen_cents = 1200.0 * math.log2(max(chosen, 1e-9))
                    jump_cents = abs(chosen_cents - prev_cents)
                    octave_jump = abs(jump_cents - 1200.0) <= 240.0
                    chosen_score = float(alt_score)
                    srh_ratio = float(alt_ratio)
                    srh_ok = True
                    self._octave_relax_left = max(self._octave_relax_left, int(self.cfg.octave_fix_relax_frames))
                    self._octave_stuck_frames = 0

        if octave_jump:
            if self._octave_pending_hz is not None:
                pending_cents = 1200.0 * math.log2(max(float(self._octave_pending_hz), 1e-9))
                if abs(pending_cents - chosen_cents) <= 70.0:
                    self._octave_pending_frames += 1
                else:
                    self._octave_pending_hz = float(chosen)
                    self._octave_pending_frames = 1
            else:
                self._octave_pending_hz = float(chosen)
                self._octave_pending_frames = 1

            if self._octave_pending_frames >= confirm_frames and srh_ok and anchor_jump_ok and (high_conf or self._octave_pending_frames >= force_frames):
                self._octave_prev_hz = float(chosen)
                self._octave_pending_hz = None
                self._octave_pending_frames = 0
                self._octave_last_out_hz = float(chosen)
                self._octave_relax_left = max(0, self._octave_relax_left - 1)
                self._octave_bias_up_frames = 0
                self._octave_bias_down_frames = 0
                out_dbg = {
                    "oct_mode": 2.0,
                    "oct_pending": 0.0,
                    "oct_srh_ratio": float(srh_ratio),
                    "oct_srh_prev": float(prev_score),
                    "oct_srh_chosen": float(chosen_score),
                }
                if octave_anchor_hz is not None and np.isfinite(float(octave_anchor_hz)):
                    out_dbg["oct_anchor_hz"] = float(octave_anchor_hz)
                out_dbg.update(melodia_dbg)
                return float(chosen), out_dbg

            if self._octave_pending_frames >= (force_frames + 2) and srh_ratio >= relaxed_margin and anchor_jump_ok:
                self._octave_prev_hz = float(chosen)
                self._octave_pending_hz = None
                self._octave_pending_frames = 0
                self._octave_last_out_hz = float(chosen)
                self._octave_relax_left = max(0, self._octave_relax_left - 1)
                self._octave_bias_up_frames = 0
                self._octave_bias_down_frames = 0
                out_dbg = {
                    "oct_mode": 7.0,
                    "oct_pending": 0.0,
                    "oct_srh_ratio": float(srh_ratio),
                    "oct_srh_prev": float(prev_score),
                    "oct_srh_chosen": float(chosen_score),
                }
                if octave_anchor_hz is not None and np.isfinite(float(octave_anchor_hz)):
                    out_dbg["oct_anchor_hz"] = float(octave_anchor_hz)
                out_dbg.update(melodia_dbg)
                return float(chosen), out_dbg

            self._octave_last_out_hz = float(prev)
            self._octave_relax_left = max(0, self._octave_relax_left - 1)
            out_dbg = {
                "oct_mode": 3.0,
                "oct_pending": float(self._octave_pending_frames),
                "oct_srh_ratio": float(srh_ratio),
                "oct_srh_prev": float(prev_score),
                "oct_srh_chosen": float(chosen_score),
            }
            if octave_anchor_hz is not None and np.isfinite(float(octave_anchor_hz)):
                out_dbg["oct_anchor_hz"] = float(octave_anchor_hz)
            out_dbg.update(melodia_dbg)
            return float(prev), out_dbg

        # Non-octave large jumps need short confirmation, then allow.
        if self._octave_pending_hz is not None:
            pending_cents = 1200.0 * math.log2(max(float(self._octave_pending_hz), 1e-9))
            if abs(pending_cents - chosen_cents) <= 70.0:
                self._octave_pending_frames += 1
            else:
                self._octave_pending_hz = float(chosen)
                self._octave_pending_frames = 1
        else:
            self._octave_pending_hz = float(chosen)
            self._octave_pending_frames = 1

        if (
            self._octave_pending_frames >= non_oct_confirm_frames
            and float(raw_conf) >= max(0.40, float(self.cfg.octave_fix_high_conf) - 0.24)
            and (srh_ok or float(raw_conf) >= float(self.cfg.octave_fix_high_conf))
        ):
            self._octave_prev_hz = float(chosen)
            self._octave_pending_hz = None
            self._octave_pending_frames = 0
            self._octave_last_out_hz = float(chosen)
            self._octave_relax_left = max(0, self._octave_relax_left - 1)
            self._octave_bias_up_frames = 0
            self._octave_bias_down_frames = 0
            out_dbg = {
                "oct_mode": 4.0,
                "oct_pending": 0.0,
                "oct_srh_ratio": float(srh_ratio),
                "oct_srh_prev": float(prev_score),
                "oct_srh_chosen": float(chosen_score),
            }
            if octave_anchor_hz is not None and np.isfinite(float(octave_anchor_hz)):
                out_dbg["oct_anchor_hz"] = float(octave_anchor_hz)
            out_dbg.update(melodia_dbg)
            return float(chosen), out_dbg
        if (
            self._octave_pending_frames >= force_frames + 1
            and float(raw_conf) >= 0.32
            and srh_ratio >= relaxed_margin
        ):
            self._octave_prev_hz = float(chosen)
            self._octave_pending_hz = None
            self._octave_pending_frames = 0
            self._octave_last_out_hz = float(chosen)
            self._octave_relax_left = max(0, self._octave_relax_left - 1)
            self._octave_bias_up_frames = 0
            self._octave_bias_down_frames = 0
            out_dbg = {
                "oct_mode": 6.0,
                "oct_pending": 0.0,
                "oct_srh_ratio": float(srh_ratio),
                "oct_srh_prev": float(prev_score),
                "oct_srh_chosen": float(chosen_score),
            }
            if octave_anchor_hz is not None and np.isfinite(float(octave_anchor_hz)):
                out_dbg["oct_anchor_hz"] = float(octave_anchor_hz)
            out_dbg.update(melodia_dbg)
            return float(chosen), out_dbg

        self._octave_last_out_hz = float(prev)
        self._octave_relax_left = max(0, self._octave_relax_left - 1)
        out_dbg = {
            "oct_mode": 5.0,
            "oct_pending": float(self._octave_pending_frames),
            "oct_srh_ratio": float(srh_ratio),
            "oct_srh_prev": float(prev_score),
            "oct_srh_chosen": float(chosen_score),
        }
        if octave_anchor_hz is not None and np.isfinite(float(octave_anchor_hz)):
            out_dbg["oct_anchor_hz"] = float(octave_anchor_hz)
        out_dbg.update(melodia_dbg)
        return float(prev), out_dbg

    def _finalize_result(
        self,
        algo: str,
        x: np.ndarray,
        result: PitchResult,
        raw_conf: float,
        raw_debug: Optional[dict[str, float]] = None,
        octave_anchor_hz: Optional[float] = None,
    ) -> PitchResult:
        voiced, vo_dbg = self._is_voiced_frame(x, raw_conf, algo)
        debug_payload: Optional[dict[str, float]] = None
        if self.cfg.debug_octave:
            debug_payload = {}
            if raw_debug:
                debug_payload.update(raw_debug)
            debug_payload.update(vo_dbg)

        if not voiced:
            self._raw_prev_hz = None
            self._octave_bias_up_frames = 0
            self._octave_bias_down_frames = 0
            if algo == "mpm":
                self._mpm_raw_prev_hz = None
                self._mpm_octave_pending_hz = None
                self._mpm_octave_pending_frames = 0
                self._mpm_octave_stuck_frames = 0
                self._mpm_octave_bias_up_frames = 0
                self._mpm_octave_bias_down_frames = 0
                self._mpm_dp_candidates = []
                self._mpm_dp_costs = []
                self._mpm_dp_backptrs = []
                self._mpm_dp_srh_scores = []
            return PitchResult(ok=False, error="unvoiced", f0_hz=None, confidence=float(raw_conf), debug=debug_payload)

        if not result.ok or result.f0_hz is None:
            if debug_payload is not None and result.debug:
                debug_payload.update(result.debug)
            return PitchResult(
                ok=result.ok,
                error=result.error,
                f0_hz=result.f0_hz,
                confidence=float(result.confidence),
                debug=debug_payload if debug_payload else result.debug,
            )

        if self.cfg.octave_fix:
            if self.cfg.raw_pitch and algo in {"melodia", "yin"}:
                fixed = float(result.f0_hz)
                self._octave_prev_hz = fixed
                self._octave_pending_hz = None
                self._octave_pending_frames = 0
                self._octave_last_out_hz = fixed
                self._octave_stuck_frames = 0
                self._octave_relax_left = 0
                self._octave_bias_up_frames = 0
                self._octave_bias_down_frames = 0
                oct_dbg = {"oct_mode": 10.0, "oct_pending": 0.0}
            elif algo == "mpm" and self.cfg.mpm_octave_fix.enabled:
                fixed, oct_dbg = self._stabilize_octave_mpm(float(result.f0_hz), float(result.confidence))
            elif algo == "mpm":
                fixed = float(result.f0_hz)
                self._mpm_octave_prev_hz = fixed
                self._mpm_octave_pending_hz = None
                self._mpm_octave_pending_frames = 0
                self._mpm_octave_last_out_hz = fixed
                self._mpm_octave_stuck_frames = 0
                self._mpm_octave_relax_left = 0
                self._mpm_octave_bias_up_frames = 0
                self._mpm_octave_bias_down_frames = 0
                oct_dbg = {"mpm_oct_mode": 10.0, "mpm_oct_pending": 0.0}
            else:
                fixed, oct_dbg = self._stabilize_octave(algo, float(result.f0_hz), float(result.confidence), octave_anchor_hz)
            result = PitchResult(
                ok=True,
                error=None,
                f0_hz=float(fixed),
                confidence=float(result.confidence),
                debug=debug_payload if debug_payload else result.debug,
            )
            if debug_payload is not None:
                debug_payload.update(oct_dbg)

        if self.cfg.raw_pitch:
            if self.cfg.octave_fix and result.f0_hz is not None:
                current = float(result.f0_hz)
                previous = self._mpm_raw_prev_hz if algo == "mpm" else self._raw_prev_hz
                if previous is None:
                    stabilized = current
                else:
                    prev_cents = 1200.0 * math.log2(max(previous, 1e-9))
                    curr_cents = 1200.0 * math.log2(max(current, 1e-9))
                    diff_cents = abs(curr_cents - prev_cents)
                    if algo == "mpm" and 850.0 <= diff_cents <= 1350.0 and float(result.confidence) < 0.96:
                        stabilized = previous
                    elif diff_cents <= 22.0:
                        stabilized = previous
                    elif diff_cents <= 180.0 and float(result.confidence) < 0.88:
                        stabilized = (0.30 * current) + (0.70 * previous)
                    else:
                        stabilized = current
                if algo == "mpm":
                    self._mpm_raw_prev_hz = float(stabilized)
                else:
                    self._raw_prev_hz = float(stabilized)
                result = PitchResult(
                    ok=True,
                    error=None,
                    f0_hz=float(stabilized),
                    confidence=float(result.confidence),
                    debug=debug_payload if debug_payload else result.debug,
                )
            return PitchResult(
                ok=True,
                error=None,
                f0_hz=float(result.f0_hz),
                confidence=float(result.confidence),
                debug=debug_payload if debug_payload else result.debug,
            )

        if algo == "mpm" and self.cfg.octave_fix:
            # Keep MPM on its dedicated octave tracker output and avoid the
            # generic cross-algorithm tracker here.
            return PitchResult(
                ok=True,
                error=None,
                f0_hz=float(result.f0_hz),
                confidence=float(result.confidence),
                debug=debug_payload if debug_payload else result.debug,
            )

        tracked_hz = self._track_pitch(float(result.f0_hz), float(result.confidence))
        if debug_payload is not None:
            debug_payload["tracked_hz"] = float(tracked_hz)
            if result.debug:
                debug_payload.update(result.debug)

        return PitchResult(
            ok=True,
            error=None,
            f0_hz=float(tracked_hz),
            confidence=float(result.confidence),
            debug=debug_payload if debug_payload else result.debug,
        )

    def _append_mpm_roll(self, x: np.ndarray) -> None:
        target_n = int(max(len(x) * 4, self.cfg.sr * 0.12))
        self._mpm_roll = np.concatenate([self._mpm_roll, x], dtype=np.float32)
        if self._mpm_roll.size > target_n:
            self._mpm_roll = self._mpm_roll[-target_n:]

    def _append_mpm_spec_roll(self, x: np.ndarray) -> None:
        target_n = int(max(len(x) * 6, self.cfg.sr * 0.18))
        self._mpm_spec_roll = np.concatenate([self._mpm_spec_roll, x], dtype=np.float32)
        if self._mpm_spec_roll.size > target_n:
            self._mpm_spec_roll = self._mpm_spec_roll[-target_n:]

    def _yinfft_pitch(self, x: np.ndarray) -> tuple[Optional[float], float]:
        if self._yinfft is None:
            self._yinfft = YinFftPitcher(
                sr=int(self.cfg.sr),
                hop_size=int(max(64, x.size)),
                fmin=float(self.cfg.fmin),
                fmax=float(self.cfg.fmax),
            )
        else:
            self._yinfft.reconfigure(
                sr=int(self.cfg.sr),
                hop_size=int(max(64, x.size)),
                fmin=float(self.cfg.fmin),
                fmax=float(self.cfg.fmax),
            )
        return self._yinfft.process(x)

    def _aubio_yin_pitch(self, x: np.ndarray) -> tuple[Optional[float], float]:
        if self._aubio_yin is None:
            self._aubio_yin = AubioYinPitcher(
                sr=int(self.cfg.sr),
                hop_size=int(max(64, x.size)),
                fmin=float(self.cfg.fmin),
                fmax=float(self.cfg.fmax),
            )
        else:
            self._aubio_yin.reconfigure(
                sr=int(self.cfg.sr),
                hop_size=int(max(64, x.size)),
                fmin=float(self.cfg.fmin),
                fmax=float(self.cfg.fmax),
            )
        return self._aubio_yin.process(x)

    def _pyworld_pitch(self, x: np.ndarray) -> tuple[Optional[float], float]:
        if self._pyworld is None:
            self._pyworld = PyworldPitcher(
                sr=int(self.cfg.sr),
                hop_size=int(max(64, x.size)),
                fmin=float(self.cfg.fmin),
                fmax=float(self.cfg.fmax),
            )
        else:
            self._pyworld.reconfigure(
                sr=int(self.cfg.sr),
                hop_size=int(max(64, x.size)),
                fmin=float(self.cfg.fmin),
                fmax=float(self.cfg.fmax),
            )
        return self._pyworld.process(x)

    def _parselmouth_pitch(self, x: np.ndarray) -> tuple[Optional[float], float]:
        if self._parselmouth is None:
            self._parselmouth = ParselmouthPitcher(
                sr=int(self.cfg.sr),
                hop_size=int(max(64, x.size)),
                fmin=float(self.cfg.fmin),
                fmax=float(self.cfg.fmax),
            )
        else:
            self._parselmouth.reconfigure(
                sr=int(self.cfg.sr),
                hop_size=int(max(64, x.size)),
                fmin=float(self.cfg.fmin),
                fmax=float(self.cfg.fmax),
            )
        return self._parselmouth.process(x)

    def _cluster_auto_estimates(self, estimates: list[tuple[str, float, float]]) -> tuple[Optional[float], float]:
        if not estimates:
            return None, 0.0
        if len(estimates) == 1:
            _, f0, conf = estimates[0]
            return float(f0), float(clamp(conf, 0.0, 0.999))

        # Favor the methods that currently show best octave robustness on our benchmark set.
        method_weight = {
            "yin": 0.75,
            "yinfft": 0.95,
            "aubio_yin": 1.10,
            "obp": 0.95,
            "bacf": 0.90,
            "mpm": 1.40,
            "pyworld": 1.05,
            "parselmouth": 1.10,
            "melodia": 1.25,
            "swipe": 0.85,
        }
        clusters: list[dict[str, float]] = []
        for method, f0, conf in estimates:
            cents = 1200.0 * math.log2(max(float(f0), 1e-9))
            w = float(clamp(conf, 0.0, 1.0)) * method_weight.get(method, 1.0)
            assigned = False
            for cluster in clusters:
                if abs(cents - cluster["center_cents"]) <= 45.0:
                    sum_w = cluster["sum_w"] + w
                    if sum_w > 1e-9:
                        cluster["center_cents"] = (cluster["center_cents"] * cluster["sum_w"] + cents * w) / sum_w
                        cluster["sum_w"] = sum_w
                        cluster["sum_conf"] += float(clamp(conf, 0.0, 1.0))
                        cluster["sum_wc"] += float(clamp(conf, 0.0, 1.0)) * w
                        cluster["count"] += 1.0
                    assigned = True
                    break
            if not assigned:
                c = float(clamp(conf, 0.0, 1.0))
                clusters.append({"center_cents": cents, "sum_w": w, "sum_conf": c, "sum_wc": c * w, "count": 1.0})

        prev_cents = 1200.0 * math.log2(max(self._track_prev_hz, 1e-9)) if self._track_prev_hz is not None else None
        best = clusters[0]
        best_score = -1e12
        for cluster in clusters:
            score = float(cluster["sum_wc"])
            if prev_cents is not None:
                score -= 0.0016 * abs(cluster["center_cents"] - prev_cents)
            if score > best_score:
                best_score = score
                best = cluster

        out_hz = float(2.0 ** (best["center_cents"] / 1200.0))
        out_conf = float(clamp(best["sum_wc"] / max(best["sum_w"], 1e-9), 0.0, 0.999))
        return out_hz, out_conf

    def process_chunk(self, x: np.ndarray) -> PitchResult:
        x_raw = np.asarray(x, dtype=np.float32)
        x_shared = self.pre(x_raw)
        x_mpm = self.mpm_pre(x_raw)

        # Shared spectral buffer for non-MPM algorithms.
        target_spec_n = int(max(len(x_shared) * 6, self.cfg.sr * 0.18))
        self._spec_roll = np.concatenate([self._spec_roll, x_shared], dtype=np.float32)
        if self._spec_roll.size > target_spec_n:
            self._spec_roll = self._spec_roll[-target_spec_n:]

        if self.cfg.algo == "yin":
            f0, conf = yin_pitch(x_shared, self.cfg.sr, fmin=self.cfg.fmin, fmax=self.cfg.fmax)
            conf = self.yin_cn.update(conf)
            anchor = None
            if self.cfg.raw_pitch:
                raw = self._startup_raw_result(f0, float(conf), self.cfg.yin_post.min_conf)
                return self._finalize_result("yin", x_shared, raw, float(conf), octave_anchor_hz=anchor)
            result = self.yin_post.process(f0, conf, x_shared)
            return self._finalize_result("yin", x_shared, result, float(conf), octave_anchor_hz=anchor)

        if self.cfg.algo == "yinfft":
            f0, conf = self._yinfft_pitch(x_shared)
            conf = float(clamp(conf, 0.0, 0.999))
            anchor = None
            if self.cfg.raw_pitch:
                raw = self._startup_raw_result(f0, float(conf), self.cfg.yin_post.min_conf)
                return self._finalize_result("yinfft", x_shared, raw, float(conf), octave_anchor_hz=anchor)
            result = self.yin_post.process(f0, conf, x_shared)
            return self._finalize_result("yinfft", x_shared, result, float(conf), octave_anchor_hz=anchor)

        if self.cfg.algo == "aubio_yin":
            f0, conf = self._aubio_yin_pitch(x_shared)
            conf = float(clamp(conf, 0.0, 0.999))
            anchor = None
            if self.cfg.raw_pitch:
                raw = self._startup_raw_result(f0, float(conf), self.cfg.yin_post.min_conf)
                return self._finalize_result("aubio_yin", x_shared, raw, float(conf), octave_anchor_hz=anchor)
            result = self.yin_post.process(f0, conf, x_shared)
            return self._finalize_result("aubio_yin", x_shared, result, float(conf), octave_anchor_hz=anchor)

        if self.cfg.algo == "obp":
            f0, conf = obp_pitch(self._spec_roll, self.cfg.sr, fmin=self.cfg.fmin, fmax=self.cfg.fmax)
            conf = float(clamp(conf, 0.0, 0.999))
            anchor = None
            if self.cfg.raw_pitch:
                raw = self._startup_raw_result(f0, float(conf), self.cfg.aux_post.min_conf)
                return self._finalize_result("obp", x_shared, raw, float(conf), octave_anchor_hz=anchor)
            if self._use_startup_unlock("obp"):
                raw = self._startup_raw_result(f0, float(conf), self.cfg.aux_post.min_conf)
                if raw.ok:
                    return self._finalize_result("obp", x_shared, raw, float(conf), octave_anchor_hz=anchor)
            result = self.aux_post.process(f0, float(conf), x_shared)
            return self._finalize_result("obp", x_shared, result, float(conf), octave_anchor_hz=anchor)

        if self.cfg.algo == "bacf":
            f0, conf = bacf_pitch(self._spec_roll, self.cfg.sr, fmin=self.cfg.fmin, fmax=self.cfg.fmax)
            conf = float(clamp(conf, 0.0, 0.999))
            anchor = None
            if self.cfg.raw_pitch:
                raw = self._startup_raw_result(f0, float(conf), self.cfg.aux_post.min_conf)
                return self._finalize_result("bacf", x_shared, raw, float(conf), octave_anchor_hz=anchor)
            if self._use_startup_unlock("bacf"):
                raw = self._startup_raw_result(f0, float(conf), self.cfg.aux_post.min_conf)
                if raw.ok:
                    return self._finalize_result("bacf", x_shared, raw, float(conf), octave_anchor_hz=anchor)
            result = self.aux_post.process(f0, float(conf), x_shared)
            return self._finalize_result("bacf", x_shared, result, float(conf), octave_anchor_hz=anchor)

        if self.cfg.algo == "pyworld":
            try:
                f0, conf = self._pyworld_pitch(x_shared)
            except RuntimeError as exc:
                return PitchResult(ok=False, error=str(exc), f0_hz=None, confidence=0.0)
            conf = float(clamp(conf, 0.0, 0.999))
            anchor = None
            if self.cfg.raw_pitch:
                raw = self._startup_raw_result(f0, float(conf), self.cfg.aux_post.min_conf)
                return self._finalize_result("pyworld", x_shared, raw, float(conf), octave_anchor_hz=anchor)
            result = self.aux_post.process(f0, float(conf), x_shared)
            return self._finalize_result("pyworld", x_shared, result, float(conf), octave_anchor_hz=anchor)

        if self.cfg.algo == "parselmouth":
            try:
                f0, conf = self._parselmouth_pitch(x_shared)
            except RuntimeError as exc:
                return PitchResult(ok=False, error=str(exc), f0_hz=None, confidence=0.0)
            conf = float(clamp(conf, 0.0, 0.999))
            anchor = None
            if self.cfg.raw_pitch:
                raw = self._startup_raw_result(f0, float(conf), self.cfg.aux_post.min_conf)
                return self._finalize_result("parselmouth", x_shared, raw, float(conf), octave_anchor_hz=anchor)
            result = self.aux_post.process(f0, float(conf), x_shared)
            return self._finalize_result("parselmouth", x_shared, result, float(conf), octave_anchor_hz=anchor)

        if self.cfg.algo == "mpm":
            self._append_mpm_roll(x_mpm)
            self._append_mpm_spec_roll(x_mpm)
            f0, conf = mpm_pitch(self._mpm_roll, self.cfg.sr, fmin=self.cfg.fmin, fmax=self.cfg.fmax)
            # MPM refinement is now centered on the raw detector + dedicated
            # MPM octave tracker. The generic MPM post path can reintroduce
            # octave artifacts on real voice, so octave_fix uses raw input.
            if self.cfg.raw_pitch or self.cfg.octave_fix:
                raw = self._startup_raw_result(f0, float(conf), self.cfg.mpm_post.min_conf)
                return self._finalize_result("mpm", x_mpm, raw, float(conf))
            if self._use_startup_unlock("mpm"):
                raw = self._startup_raw_result(f0, float(conf), self.cfg.mpm_post.min_conf)
                if raw.ok:
                    return self._finalize_result("mpm", x_mpm, raw, float(conf))
            result = self.mpm_post.process(f0, float(conf), x_mpm)
            return self._finalize_result("mpm", x_mpm, result, float(conf))

        if self.cfg.algo == "melodia":
            f0, conf, debug_payload = melodia_pitch(
                self._spec_roll,
                self.cfg.sr,
                fmin=self.cfg.fmin,
                fmax=self.cfg.fmax,
                harmonics=int(self.cfg.melodia_harmonics),
                n_fft_min=int(self.cfg.melodia_fft_min),
                debug=self.cfg.debug_octave,
            )
            conf = float(clamp(float(conf), 0.0, 0.999))
            anchor = None
            if self.cfg.raw_pitch:
                raw = self._startup_raw_result(f0, conf, self.cfg.melodia_post.min_conf)
                return self._finalize_result("melodia", x_shared, raw, conf, debug_payload, octave_anchor_hz=anchor)
            result = self.melodia_post.process(f0, conf, x_shared)
            return self._finalize_result("melodia", x_shared, result, conf, debug_payload, octave_anchor_hz=anchor)

        if self.cfg.algo == "swipe":
            f0, conf = swipe_pitch(self._spec_roll, self.cfg.sr, fmin=self.cfg.fmin, fmax=self.cfg.fmax)
            anchor = None
            if self.cfg.raw_pitch:
                raw = self._startup_raw_result(f0, float(conf), self.cfg.swipe_post.min_conf)
                return self._finalize_result("swipe", x_shared, raw, float(conf), octave_anchor_hz=anchor)
            if self._use_startup_unlock("swipe"):
                raw = self._startup_raw_result(f0, float(conf), self.cfg.swipe_post.min_conf)
                if raw.ok:
                    return self._finalize_result("swipe", x_shared, raw, float(conf), octave_anchor_hz=anchor)
            result = self.swipe_post.process(f0, float(conf), x_shared)
            return self._finalize_result("swipe", x_shared, result, float(conf), octave_anchor_hz=anchor)

        # auto ensemble mode
        self._append_mpm_roll(x_shared)
        estimates: list[tuple[str, float, float]] = []

        yin_f0, yin_conf = yin_pitch(x_shared, self.cfg.sr, fmin=self.cfg.fmin, fmax=self.cfg.fmax)
        yin_conf = self.yin_cn.update(yin_conf)
        yin_conf = float(clamp(yin_conf, 0.0, 0.999))
        if yin_f0 is not None and float(self.cfg.fmin) <= float(yin_f0) <= float(self.cfg.fmax):
            if yin_conf >= max(0.05, float(self.cfg.yin_post.min_conf) * 0.55):
                estimates.append(("yin", float(yin_f0), yin_conf))

        yinfft_f0, yinfft_conf = self._yinfft_pitch(x_shared)
        yinfft_conf = float(clamp(yinfft_conf, 0.0, 0.999))
        if yinfft_f0 is not None and float(self.cfg.fmin) <= float(yinfft_f0) <= float(self.cfg.fmax):
            if yinfft_conf >= max(0.05, float(self.cfg.yin_post.min_conf) * 0.55):
                estimates.append(("yinfft", float(yinfft_f0), yinfft_conf))

        aubio_yin_f0, aubio_yin_conf = self._aubio_yin_pitch(x_shared)
        aubio_yin_conf = float(clamp(aubio_yin_conf, 0.0, 0.999))
        if aubio_yin_f0 is not None and float(self.cfg.fmin) <= float(aubio_yin_f0) <= float(self.cfg.fmax):
            if aubio_yin_conf >= max(0.05, float(self.cfg.yin_post.min_conf) * 0.55):
                estimates.append(("aubio_yin", float(aubio_yin_f0), aubio_yin_conf))

        obp_f0, obp_conf = obp_pitch(self._spec_roll, self.cfg.sr, fmin=self.cfg.fmin, fmax=self.cfg.fmax)
        obp_conf = float(clamp(obp_conf, 0.0, 0.999))
        if obp_f0 is not None and float(self.cfg.fmin) <= float(obp_f0) <= float(self.cfg.fmax):
            if obp_conf >= max(0.05, float(self.cfg.aux_post.min_conf) * 0.55):
                estimates.append(("obp", float(obp_f0), obp_conf))

        bacf_f0, bacf_conf = bacf_pitch(self._spec_roll, self.cfg.sr, fmin=self.cfg.fmin, fmax=self.cfg.fmax)
        bacf_conf = float(clamp(bacf_conf, 0.0, 0.999))
        if bacf_f0 is not None and float(self.cfg.fmin) <= float(bacf_f0) <= float(self.cfg.fmax):
            if bacf_conf >= max(0.05, float(self.cfg.aux_post.min_conf) * 0.55):
                estimates.append(("bacf", float(bacf_f0), bacf_conf))

        mpm_f0, mpm_conf = mpm_pitch(self._mpm_roll, self.cfg.sr, fmin=self.cfg.fmin, fmax=self.cfg.fmax)
        mpm_conf = float(clamp(mpm_conf, 0.0, 0.999))
        if mpm_f0 is not None and float(self.cfg.fmin) <= float(mpm_f0) <= float(self.cfg.fmax):
            if mpm_conf >= max(0.05, float(self.cfg.mpm_post.min_conf) * 0.55):
                estimates.append(("mpm", float(mpm_f0), mpm_conf))

        mel_f0, mel_conf, mel_dbg = melodia_pitch(
            self._spec_roll,
            self.cfg.sr,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
            harmonics=int(self.cfg.melodia_harmonics),
            n_fft_min=int(self.cfg.melodia_fft_min),
            debug=self.cfg.debug_octave,
        )
        mel_conf = float(clamp(float(mel_conf), 0.0, 0.999))
        if mel_f0 is not None and float(self.cfg.fmin) <= float(mel_f0) <= float(self.cfg.fmax):
            if mel_conf >= max(0.10, float(self.cfg.melodia_post.min_conf) * 0.85):
                estimates.append(("melodia", float(mel_f0), mel_conf))

        swp_f0, swp_conf = swipe_pitch(self._spec_roll, self.cfg.sr, fmin=self.cfg.fmin, fmax=self.cfg.fmax)
        swp_conf = float(clamp(swp_conf, 0.0, 0.999))
        if swp_f0 is not None and float(self.cfg.fmin) <= float(swp_f0) <= float(self.cfg.fmax):
            if swp_conf >= max(0.05, float(self.cfg.swipe_post.min_conf) * 0.55):
                estimates.append(("swipe", float(swp_f0), swp_conf))

        auto_f0, auto_conf = self._cluster_auto_estimates(estimates)
        if self.cfg.raw_pitch:
            raw = self._startup_raw_result(auto_f0, float(auto_conf), self.cfg.auto_post.min_conf)
            dbg = None
            if self.cfg.debug_octave:
                dbg = {"ensemble_count": float(len(estimates))}
                if mel_dbg:
                    dbg.update({f"mel_{k}": v for k, v in mel_dbg.items()})
            return self._finalize_result("auto", x_shared, raw, float(auto_conf), dbg)
        result = self.auto_post.process(auto_f0, float(auto_conf), x_shared)
        dbg = None
        if self.cfg.debug_octave:
            dbg = {"ensemble_count": float(len(estimates))}
            if mel_dbg:
                dbg.update({f"mel_{k}": v for k, v in mel_dbg.items()})
        return self._finalize_result("auto", x_shared, result, float(auto_conf), dbg)

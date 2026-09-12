"""Unit test cho Praat params."""
import numpy as np
import parselmouth
from parselmouth.praat import call


PRAAT_CONFIG = {
    'pitch_floor': 75.0,
    'pitch_ceiling': 500.0,
    'time_step': 0.01,
}


def _load_dummy_sound(duration=0.5, fs=16000, f0=150.0):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    samples = 0.5 * np.sin(2 * np.pi * f0 * t)
    samples = samples + 0.01 * np.random.RandomState(0).randn(len(samples))
    return parselmouth.Sound(samples, sampling_frequency=fs)


def test_pitch_params():
    sound = _load_dummy_sound(duration=1.0)
    pitch = call(sound, "To Pitch (cc)",
                 PRAAT_CONFIG['time_step'],
                 PRAAT_CONFIG['pitch_floor'],
                 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14,
                 PRAAT_CONFIG['pitch_ceiling'])
    assert pitch is not None
    n_frames = pitch.get_number_of_frames()
    assert abs(n_frames - 101) <= 5
    f0_values = pitch.selected_array['frequency']
    f0_voiced = f0_values[f0_values > 0]
    assert len(f0_voiced) > 0
    assert 140 < f0_voiced.mean() < 160


def test_hnr_params():
    sound = _load_dummy_sound(duration=0.5)
    harmonicity = call(sound, "To Harmonicity (cc)",
                       PRAAT_CONFIG['time_step'],
                       PRAAT_CONFIG['pitch_floor'],
                       0.1, 1.0)
    assert harmonicity is not None
    assert 45 <= harmonicity.get_number_of_frames() <= 55


def test_jitter_shimmer_variants():
    sound = _load_dummy_sound(duration=1.0)
    pp = call(sound, "To PointProcess (periodic, cc)",
              PRAAT_CONFIG['pitch_floor'], PRAAT_CONFIG['pitch_ceiling'])
    jitter_local = call(pp, "Get jitter (local)",
                        0.0, 0.0, 0.0001, 0.02, 1.3)
    shimmer_local = call([sound, pp], "Get shimmer (local)",
                         0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    assert jitter_local is not None
    assert shimmer_local is not None

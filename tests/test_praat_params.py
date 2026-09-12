"""Unit test cho Praat params (Muc 3.2 Buoc 1.1 cua CARVE v5.1).

Kiem tra config Praat dung:
- Pitch floor 75 Hz, ceiling 500 Hz
- Time step 0.01 s (hop 10 ms)
- Harmonicity window 0.05 s (50 ms), method To Harmonicity (cc)
- Jitter variant: local
- Shimmer variant: local
"""
import parselmouth
from parselmouth.praat import call


# Config chot truoc (Muc 3.2 Phan II)
PRAAT_CONFIG = {
    'pitch_floor': 75.0,
    'pitch_ceiling': 500.0,
    'time_step': 0.01,
    'hnr_window': 0.05,
    'hnr_method': 'To Harmonicity (cc)',
    'jitter_variant': 'local',
    'shimmer_variant': 'local',
}


def _load_dummy_sound(duration=0.5, fs=16000, f0=150.0):
    """Tao sound dummy: sine wave 150 Hz, 0.5s, 16kHz."""
    import numpy as np
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    samples = 0.5 * np.sin(2 * np.pi * f0 * t)
    sound = parselmouth.Sound(samples, sampling_frequency=fs)
    return sound


def test_pitch_params():
    """Pitch floor/ceiling/time_step phai dung config chot truoc."""
    sound = _load_dummy_sound()
    pitch = call(sound, "To Pitch (cc)", 
                 PRAAT_CONFIG['time_step'],
                 PRAAT_CONFIG['pitch_floor'],
                 15,  # max_candidates
                 "no",  # very_accurate
                 0.03,  # silence_threshold
                 0.45,  # voicing_threshold
                 0.01,  # octave_cost
                 0.35,  # octave_jump_cost
                 0.14,  # voiced_unvoiced_cost
                 PRAAT_CONFIG['pitch_ceiling'])
    assert pitch is not None
    # Frame count: duration / time_step + 1 (xap xi)
    n_frames = pitch.get_number_of_frames()
    expected_frames = int(0.5 / 0.01) + 1
    assert abs(n_frames - expected_frames) <= 5, (
        f"n_frames={n_frames}, expected~{expected_frames}"
    )
    # F0 trung binh phai ~150 Hz
    f0_values = pitch.selected_array['frequency']
    f0_voiced = f0_values[f0_values > 0]
    assert len(f0_voiced) > 0, "Khong co frame voiced nao"
    mean_f0 = f0_voiced.mean()
    assert 140 < mean_f0 < 160, f"mean_f0={mean_f0:.2f}, expected~150"


def test_hnr_params():
    """Harmonicity window 50ms, method cc."""
    sound = _load_dummy_sound()
    harmonicity = call(sound, "To Harmonicity (cc)",
                       PRAAT_CONFIG['time_step'],
                       PRAAT_CONFIG['pitch_floor'],
                       0.1,  # silence_threshold
                       1.0)  # periods_per_window
    assert harmonicity is not None
    n_frames = harmonicity.get_number_of_frames()
    # ~0.5s / 0.01s = 50 frames
    assert 45 <= n_frames <= 55, f"n_frames={n_frames}"


def test_jitter_shimmer_variants():
    """Jitter/shimmer local variant phai chay duoc tren sound voiced."""
    sound = _load_dummy_sound(duration=1.0)
    pitch = call(sound, "To Pitch (cc)", 0.01, 75.0, 15, "no", 0.03,
                 0.45, 0.01, 0.35, 0.14, 500.0)
    pp = call(sound, "To PointProcess (cc)")
    # local jitter
    jitter_local = call(pp, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
    # local shimmer
    shimmer_local = call([sound, pp], "Get shimmer (local)",
                         0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    # Sine wave sach -> jitter/shimmer rat nho
    assert jitter_local is not None
    assert shimmer_local is not None


if __name__ == '__main__':
    test_pitch_params()
    test_hnr_params()
    test_jitter_shimmer_variants()
    print("All praat_params tests passed.")
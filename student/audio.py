"""Versioned causal 40-band frontend; no per-clip statistics or future samples."""
import numpy as np

CONTRACT = {'version': 'causal_logmel40_v1', 'sample_rate': 16000, 'fps': 16,
            'window_samples': 400, 'n_fft': 512, 'bands': 40, 'fmin': 50., 'fmax': 7600.,
            'window': 'numpy_hanning_symmetric', 'power_divisor': 512,
            'filter_normalization': 'sum', 'log_floor': 1e-8,
            'normalization': 'clip((log10(power)+4)/4,-2,2)',
            'feature_end': 'video_frame_pts', 'history': 8, 'left_padding': 'repeat_first_feature'}


def filterbank():
    frequencies = np.fft.rfftfreq(512, d=1/16000)
    lo, hi = 2595*np.log10(1+50/700), 2595*np.log10(1+7600/700)
    edges = 700*(10**(np.linspace(lo, hi, 42)/2595)-1)
    weights = []
    for left, center, right in zip(edges[:-2], edges[1:-1], edges[2:]):
        triangle = np.maximum(0, np.minimum((frequencies-left)/(center-left), (right-frequencies)/(right-center)))
        weights.append(triangle/triangle.sum())
    return np.asarray(weights, dtype=np.float32)


def features(waveform, times):
    waveform = np.asarray(waveform, dtype=np.float32)
    times = np.asarray(times, dtype=np.float64)
    if waveform.ndim != 1 or times.ndim != 1 or not np.isfinite(waveform).all() or not np.isfinite(times).all():
        raise ValueError('invalid audio/timestamp array')
    if np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError('video times must be nonnegative and strictly increasing')
    bank, window, result = filterbank(), np.hanning(400), []
    for time in times:
        end = round(float(time)*16000)
        if end > len(waveform):
            raise ValueError('audio does not cover video feature timestamp')
        segment = np.zeros(400, dtype=np.float32)
        start = max(0, end-400)
        if end > start:
            segment[-(end-start):] = waveform[start:end]
        power = np.abs(np.fft.rfft(segment*window, n=512))**2/512
        mel = bank @ power
        result.append(np.clip((np.log10(np.maximum(mel, 1e-8))+4)/4, -2, 2))
    return np.asarray(result, dtype=np.float32)


def histories(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 40 or len(values) == 0:
        raise ValueError('expected nonempty [frames,40] features')
    padded = np.concatenate([np.repeat(values[:1], 7, axis=0), values])
    return np.stack([padded[i:i+8] for i in range(len(values))])

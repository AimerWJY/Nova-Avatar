package com.nova.avatar.preview;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

final class AudioFrontend {
    static final int RATE = 16000;
    static final int SAMPLES_PER_FRAME = 1000;
    static final int MAX_SAMPLES = RATE * 10;
    private static final int FFT = 512;
    private static final int WINDOW = 400;
    private static final int BANDS = 40;

    static short[] decodeWav(byte[] bytes) throws IOException {
        ByteBuffer in = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        if (bytes.length < 12 || in.getInt(0) != 0x46464952 || in.getInt(8) != 0x45564157
                || Integer.toUnsignedLong(in.getInt(4)) != bytes.length - 8) {
            throw new IOException("需要标准 RIFF WAV 文件");
        }
        boolean format = false;
        short[] samples = null;
        int offset = 12;
        while (offset + 8 <= bytes.length) {
            int type = in.getInt(offset);
            long length = Integer.toUnsignedLong(in.getInt(offset + 4));
            int start = offset + 8;
            long next = start + length + (length & 1);
            if (next > bytes.length) throw new IOException("WAV 数据不完整");
            if (type == 0x20746d66) {
                if (format || length < 16 || in.getShort(start) != 1 || in.getShort(start + 2) != 1
                        || in.getInt(start + 4) != RATE || in.getInt(start + 8) != RATE * 2
                        || in.getShort(start + 12) != 2 || in.getShort(start + 14) != 16) {
                    throw new IOException("音频需为 16 kHz 单声道 PCM 16-bit WAV");
                }
                format = true;
            } else if (type == 0x61746164) {
                if (samples != null || length % 2 != 0 || length < SAMPLES_PER_FRAME * 2
                        || length > MAX_SAMPLES * 2) {
                    throw new IOException("音频需为 0.0625 至 10 秒");
                }
                samples = new short[(int) length / 2];
                in.position(start);
                in.asShortBuffer().get(samples);
            }
            offset = (int) next;
        }
        if (offset != bytes.length || !format || samples == null) throw new IOException("WAV 缺少音频或格式信息");
        return samples;
    }

    private final double[] window = new double[WINDOW];
    private final float[][] bank = new float[BANDS][FFT / 2 + 1];
    private final double[] cos = new double[FFT / 2];
    private final double[] sin = new double[FFT / 2];
    private final double[] real = new double[FFT];
    private final double[] imag = new double[FFT];
    private final double[] power = new double[FFT / 2 + 1];

    AudioFrontend() {
        for (int i = 0; i < WINDOW; ++i) window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (WINDOW - 1));
        for (int i = 0; i < cos.length; ++i) {
            cos[i] = Math.cos(-2 * Math.PI * i / FFT);
            sin[i] = Math.sin(-2 * Math.PI * i / FFT);
        }
        double lo = 2595 * Math.log10(1 + 50.0 / 700);
        double hi = 2595 * Math.log10(1 + 7600.0 / 700);
        double[] edges = new double[BANDS + 2];
        for (int i = 0; i < edges.length; ++i) {
            edges[i] = 700 * (Math.pow(10, (lo + (hi - lo) * i / (edges.length - 1)) / 2595) - 1);
        }
        for (int band = 0; band < BANDS; ++band) {
            double left = edges[band], center = edges[band + 1], right = edges[band + 2];
            double[] weights = new double[power.length];
            double sum = 0;
            for (int bin = 0; bin < weights.length; ++bin) {
                double hz = (double) bin * RATE / FFT;
                weights[bin] = Math.max(0, Math.min((hz - left) / (center - left), (right - hz) / (right - center)));
                sum += weights[bin];
            }
            for (int bin = 0; bin < weights.length; ++bin) bank[band][bin] = (float) (weights[bin] / sum);
        }
    }

    void compute(short[] samples, int frame, float[] history) {
        int end = frame * SAMPLES_PER_FRAME;
        for (int i = 0; i < FFT; ++i) {
            int reversed = Integer.reverse(i) >>> (32 - 9);
            int sample = end - WINDOW + i;
            real[reversed] = i < WINDOW && sample >= 0 ? samples[sample] / 32768.0 * window[i] : 0;
            imag[reversed] = 0;
        }
        for (int size = 2; size <= FFT; size *= 2) {
            int half = size / 2;
            for (int base = 0; base < FFT; base += size) {
                for (int i = 0; i < half; ++i) {
                    int a = base + i, b = a + half, angle = i * FFT / size;
                    double r = real[b] * cos[angle] - imag[b] * sin[angle];
                    double j = real[b] * sin[angle] + imag[b] * cos[angle];
                    real[b] = real[a] - r;
                    imag[b] = imag[a] - j;
                    real[a] += r;
                    imag[a] += j;
                }
            }
        }
        for (int i = 0; i < power.length; ++i) power[i] = (real[i] * real[i] + imag[i] * imag[i]) / FFT;
        System.arraycopy(history, BANDS, history, 0, history.length - BANDS);
        int current = history.length - BANDS;
        for (int band = 0; band < BANDS; ++band) {
            double mel = 0;
            for (int bin = 0; bin < power.length; ++bin) mel += bank[band][bin] * power[bin];
            history[current + band] = (float) Math.max(-2, Math.min(2, (Math.log10(Math.max(mel, 1e-8)) + 4) / 4));
        }
        if (frame == 0) {
            for (int i = 0; i < current; i += BANDS) System.arraycopy(history, current, history, i, BANDS);
        }
    }
}

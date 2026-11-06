package com.nova.avatar.preview;

import android.app.Activity;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTrack;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;

public final class MainActivity extends Activity {
    private static final String TAG = "NovaAvatarPreview";
    private static final int SIDE = 256;
    private static final int PHOTO = 1, AUDIO = 2;
    private static native long openModel(AssetManager assets, int[] image);
    private static native void generateFrame(long model, float[] history, int[] output);
    private static native void closeModel(long model);

    private Uri photoUri, audioUri;
    private TextView status;
    private Button generate;
    private Preview view;
    private Playback active;

    @Override public void onCreate(Bundle saved) {
        super.onCreate(saved);
        setVolumeControlStream(android.media.AudioManager.STREAM_MUSIC);
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        Button photo = button(layout, "选择参考原图", () -> choose(PHOTO, "image/*"));
        Button audio = button(layout, "选择 WAV 音频", () -> choose(AUDIO, "audio/*"));
        generate = button(layout, "生成并播放", () -> start(false));
        generate.setEnabled(false);
        status = new TextView(this);
        status.setText("试播版支持已验证参考图，音频为 16 kHz 单声道 PCM WAV，最长 10 秒");
        layout.addView(status);
        view = new Preview();
        layout.addView(view, new LinearLayout.LayoutParams(-1, 0, 1));
        setContentView(layout);
        photo.setContentDescription("选择参考原图");
        audio.setContentDescription("选择 WAV 音频");
        if (getIntent().getBooleanExtra("benchmark", false)) view.post(() -> start(true));
    }

    private Button button(LinearLayout layout, String label, Runnable action) {
        Button button = new Button(this);
        button.setText(label);
        button.setOnClickListener(v -> action.run());
        layout.addView(button);
        return button;
    }

    private void choose(int request, String type) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType(type);
        startActivityForResult(intent, request);
    }

    @Override protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (result != RESULT_OK || data == null || data.getData() == null) return;
        if (request == PHOTO) photoUri = data.getData();
        if (request == AUDIO) audioUri = data.getData();
        generate.setEnabled(active == null && photoUri != null && audioUri != null);
        status.setText((photoUri == null ? "请选择图片" : "图片已选") + " · "
                + (audioUri == null ? "请选择音频" : "音频已选"));
    }

    private static byte[] read(InputStream input, int limit) throws IOException {
        if (input == null) throw new IOException("文件无法打开");
        try (InputStream stream = input; ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] chunk = new byte[8192];
            int length;
            while ((length = stream.read(chunk)) != -1) {
                if (out.size() + length > limit) throw new IOException("文件超过试播版大小限制");
                out.write(chunk, 0, length);
            }
            return out.toByteArray();
        }
    }

    private int[] decodePhoto(byte[] bytes) throws Exception {
        String hash = new String(read(getAssets().open("reference.sha256"), 64), StandardCharsets.US_ASCII);
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
        StringBuilder hex = new StringBuilder(64);
        for (byte value : digest) hex.append(Character.forDigit((value & 255) >>> 4, 16))
                .append(Character.forDigit(value & 15, 16));
        if (!hash.contentEquals(hex)) throw new IOException("请选择当前模型对应的参考原图");
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap original = BitmapFactory.decodeByteArray(bytes, 0, bytes.length, options);
        if (original == null || original.getWidth() * 4L != original.getHeight() * 3L) {
            throw new IOException("参考原图无法按当前构图读取");
        }
        Bitmap square = Bitmap.createBitmap(SIDE, SIDE, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(square);
        canvas.drawColor(0xff000000);
        canvas.drawBitmap(original, null, new Rect(32, 0, 224, SIDE), new Paint(Paint.FILTER_BITMAP_FLAG));
        int[] image = new int[SIDE * SIDE];
        square.getPixels(image, 0, SIDE, 0, 0, SIDE, SIDE);
        return image;
    }

    private void start(boolean benchmark) {
        if (active != null || (!benchmark && (photoUri == null || audioUri == null))) return;
        Playback run = new Playback(benchmark, photoUri, audioUri);
        active = run;
        view.show(null);
        generate.setEnabled(false);
        status.setText("正在读取并生成");
        new Thread(run::produce, "NovaAvatarCPU").start();
    }

    private void finish(Playback run, String message) {
        if (active != run) return;
        active = null;
        run.cancelled = true;
        if (run.track != null) run.track.release();
        generate.setEnabled(photoUri != null && audioUri != null);
        status.setText(message);
    }

    @Override protected void onStop() {
        super.onStop();
        if (active != null) finish(active, "播放已停止");
    }

    private static double now() {
        return SystemClock.elapsedRealtimeNanos() / 1_000_000.0;
    }

    private static final class Frame {
        final int index;
        final Bitmap image;
        Frame(int index, Bitmap image) { this.index = index; this.image = image; }
    }

    private final class Playback {
        final boolean benchmark;
        final Uri photo, audio;
        final double request = now();
        final ArrayBlockingQueue<Frame> queue = new ArrayBlockingQueue<>(6);
        volatile boolean cancelled;
        short[] samples;
        AudioTrack track;
        int frames, lastDrawn = -1, drawn, dropped;
        double decodedMs, readyMs, firstRgbMs, drawMs, audioMs = -1, playRequested;

        Playback(boolean benchmark, Uri photo, Uri audio) {
            this.benchmark = benchmark;
            this.photo = photo;
            this.audio = audio;
        }

        void produce() {
            long model = 0;
            try {
                byte[] imageBytes = read(benchmark ? new FileInputStream(getFileStreamPath("reference.png"))
                        : getContentResolver().openInputStream(photo), 16 * 1024 * 1024);
                int[] image = decodePhoto(imageBytes);
                byte[] wav = read(benchmark ? new FileInputStream(getFileStreamPath("input.wav"))
                        : getContentResolver().openInputStream(audio), 1024 * 1024);
                samples = AudioFrontend.decodeWav(wav);
                frames = (samples.length + AudioFrontend.SAMPLES_PER_FRAME - 1) / AudioFrontend.SAMPLES_PER_FRAME;
                AudioFrontend frontend = new AudioFrontend();
                decodedMs = now() - request;
                if (cancelled) return;
                System.loadLibrary("nova_avatar");
                model = openModel(getAssets(), image);
                readyMs = now() - request;
                float[] history = new float[320];
                int[] pixels = new int[SIDE * SIDE];
                for (int i = 0; i < frames && !cancelled; ++i) {
                    frontend.compute(samples, i, history);
                    generateFrame(model, history, pixels);
                    Frame frame = new Frame(i, Bitmap.createBitmap(pixels, SIDE, SIDE, Bitmap.Config.ARGB_8888));
                    if (i == 0) {
                        firstRgbMs = now() - request;
                        view.post(() -> preparePlayback(frame));
                    } else {
                        while (!cancelled && !queue.offer(frame, 50, TimeUnit.MILLISECONDS)) {
                            // Bound memory while the producer waits for playback.
                        }
                    }
                }
            } catch (Exception | LinkageError error) {
                view.post(() -> {
                    if (active == this) {
                        Log.e(TAG, "preview_failed", error);
                        finish(this, "试播失败 · " + error.getMessage());
                    }
                });
            } finally {
                if (model != 0) closeModel(model);
            }
        }

        void preparePlayback(Frame first) {
            if (active != this) return;
            try {
                track = new AudioTrack.Builder()
                        .setAudioAttributes(new AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                        .setAudioFormat(new AudioFormat.Builder().setSampleRate(AudioFrontend.RATE)
                                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                        .setBufferSizeInBytes(samples.length * 2)
                        .setTransferMode(AudioTrack.MODE_STATIC).build();
                if (track.write(samples, 0, samples.length) != samples.length
                        || track.getState() != AudioTrack.STATE_INITIALIZED) {
                    throw new IllegalStateException("无法写入音频");
                }
                view.show(first);
            } catch (RuntimeException error) {
                Log.e(TAG, "audio_setup_failed", error);
                finish(this, "音频播放初始化失败");
            }
        }

        void didDraw(int index) {
            if (index == lastDrawn) return;
            lastDrawn = index;
            ++drawn;
            if (index == 0) {
                drawMs = now() - request;
                view.post(() -> {
                    if (active != this) return;
                    try {
                        track.play();
                        playRequested = now();
                        status.setText("正在播放");
                        view.postOnAnimation(this::advance);
                    } catch (RuntimeException error) {
                        Log.e(TAG, "audio_start_failed", error);
                        finish(this, "无法播放音频");
                    }
                });
            }
        }

        void advance() {
            if (active != this) return;
            int head = track.getPlaybackHeadPosition();
            if (audioMs < 0 && head > 0) audioMs = now() - request;
            int target = Math.min(frames - 1, head / AudioFrontend.SAMPLES_PER_FRAME);
            Frame latest = null;
            while (queue.peek() != null && queue.peek().index <= target) {
                if (latest != null) ++dropped;
                latest = queue.poll();
            }
            if (latest != null) {
                if (view.frame != null && view.frame.index > lastDrawn) ++dropped;
                view.show(latest);
            }
            if (head >= samples.length && lastDrawn == frames - 1) {
                Log.i(TAG, "preview_complete frames=" + frames + " drawn=" + drawn + " dropped=" + dropped
                        + " decode_ms=" + decodedMs + " model_ready_ms=" + readyMs + " first_rgb_ms=" + firstRgbMs
                        + " first_draw_ms=" + drawMs + " audio_head_ms=" + audioMs
                        + " startup_ms=" + Math.max(drawMs, audioMs) + " playback_wall_ms=" + (now() - playRequested));
                finish(this, "播放完成 · " + drawn + "/" + frames + " 帧");
            } else if (now() - playRequested > samples.length / 16.0 + 3000) {
                Log.e(TAG, "playback_timeout head=" + head + " last_drawn=" + lastDrawn);
                finish(this, "播放超时");
            } else {
                view.postOnAnimation(this::advance);
            }
        }
    }

    private final class Preview extends View {
        private final Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);
        private final Rect destination = new Rect();
        Frame frame;
        Preview() { super(MainActivity.this); }

        void show(Frame value) {
            frame = value;
            invalidate();
        }

        @Override protected void onDraw(Canvas canvas) {
            canvas.drawColor(0xff000000);
            if (frame == null) return;
            int side = Math.min(getWidth(), getHeight());
            int x = (getWidth() - side) / 2, y = (getHeight() - side) / 2;
            destination.set(x, y, x + side, y + side);
            canvas.drawBitmap(frame.image, null, destination, paint);
            if (active != null) active.didDraw(frame.index);
        }
    }
}

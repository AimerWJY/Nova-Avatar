#include <android/asset_manager_jni.h>
#include <jni.h>
#include <net.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int side = 256;
constexpr int pixels = side * side;

void check(int code, const char* action) {
    if (code != 0) throw std::runtime_error(action);
}

void load(ncnn::Net& net, AAssetManager* assets, const std::string& name) {
    net.opt.num_threads = 2;
    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_fp16_uniform = false;
    check(net.load_param(assets, (name + ".ncnn.param").c_str()), "读取模型结构失败");
    check(net.load_model(assets, (name + ".ncnn.bin").c_str()), "读取模型权重失败");
}

struct Session {
    ncnn::Net encoder, decoder;
    ncnn::Mat image{side, side, 3}, spatial, identity, state{128};

    Session(AAssetManager* assets, const std::vector<jint>& argb) {
        if (image.empty() || state.empty()) throw std::bad_alloc();
        load(encoder, assets, "encoder");
        load(decoder, assets, "frame");
        for (int c = 0; c < 3; ++c) {
            float* values = image.channel(c);
            for (int i = 0; i < pixels; ++i) values[i] = ((argb[i] >> (16 - 8 * c)) & 255) / 255.f;
        }
        state.fill(0.f);
        auto ex = encoder.create_extractor();
        check(ex.input("in0", image), "输入图片失败");
        check(ex.extract("out0", spatial), "提取空间特征失败");
        check(ex.extract("out1", identity), "提取人物特征失败");
        // Extractor type 0 converts packed CPU outputs to scalar float32.
        spatial = spatial.reshape(64, 64, 32);
        identity = identity.reshape(128);
        if (spatial.empty() || identity.empty()) throw std::runtime_error("图片特征形状错误");
    }

    void frame(const ncnn::Mat& audio, std::vector<jint>& argb) {
        auto ex = decoder.create_extractor();
        check(ex.input("in0", image), "帧图片输入失败");
        check(ex.input("in1", spatial), "帧空间特征输入失败");
        check(ex.input("in2", identity), "帧人物特征输入失败");
        check(ex.input("in3", audio), "帧音频特征输入失败");
        check(ex.input("in4", state), "帧状态输入失败");
        ncnn::Mat rgb, next;
        check(ex.extract("out0", rgb), "生成画面失败");
        check(ex.extract("out1", next), "生成状态失败");
        if (rgb.w != side || rgb.h != side || rgb.c != 3 || rgb.elempack != 1)
            throw std::runtime_error("画面形状错误");
        state = next.reshape(128);
        if (state.empty() || state.elempack != 1) throw std::runtime_error("状态形状错误");
        for (int i = 0; i < 128; ++i) {
            if (!std::isfinite(state[i])) throw std::runtime_error("状态数值错误");
        }
        std::fill(argb.begin(), argb.end(), static_cast<jint>(0xff000000u));
        for (int c = 0; c < 3; ++c) {
            const float* values = rgb.channel(c);
            for (int i = 0; i < pixels; ++i) {
                if (!std::isfinite(values[i])) throw std::runtime_error("画面数值错误");
                argb[i] |= static_cast<jint>(std::lround(std::clamp(values[i], 0.f, 1.f) * 255)) << (16 - 8 * c);
            }
        }
    }
};

void fail(JNIEnv* env, const std::exception& error) {
    jclass type = env->FindClass("java/lang/IllegalStateException");
    if (type) env->ThrowNew(type, error.what());
}
}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_nova_avatar_preview_MainActivity_openModel(JNIEnv* env, jclass, jobject manager, jintArray image) {
    try {
        AAssetManager* assets = AAssetManager_fromJava(env, manager);
        if (!assets || env->GetArrayLength(image) != pixels) throw std::runtime_error("模型或图片输入错误");
        std::vector<jint> argb(pixels);
        env->GetIntArrayRegion(image, 0, pixels, argb.data());
        if (env->ExceptionCheck()) return 0;
        auto session = std::make_unique<Session>(assets, argb);
        return reinterpret_cast<jlong>(session.release());
    } catch (const std::exception& error) {
        fail(env, error);
        return 0;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nova_avatar_preview_MainActivity_generateFrame(JNIEnv* env, jclass, jlong handle,
                                                        jfloatArray history, jintArray output) {
    try {
        if (!handle || env->GetArrayLength(history) != 320 || env->GetArrayLength(output) != pixels)
            throw std::runtime_error("帧输入错误");
        ncnn::Mat audio(40, 8);
        if (audio.empty()) throw std::bad_alloc();
        env->GetFloatArrayRegion(history, 0, 320, audio);
        if (env->ExceptionCheck()) return;
        std::vector<jint> argb(pixels);
        reinterpret_cast<Session*>(handle)->frame(audio, argb);
        env->SetIntArrayRegion(output, 0, pixels, argb.data());
    } catch (const std::exception& error) {
        fail(env, error);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nova_avatar_preview_MainActivity_closeModel(JNIEnv*, jclass, jlong handle) {
    delete reinterpret_cast<Session*>(handle);
}

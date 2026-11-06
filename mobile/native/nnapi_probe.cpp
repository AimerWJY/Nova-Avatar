// Small operator/driver probe, NOT a Wan/DiT/VAE model benchmark.
#include <android/NeuralNetworks.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

using Clock = std::chrono::steady_clock;
using Model = std::unique_ptr<ANeuralNetworksModel, decltype(&ANeuralNetworksModel_free)>;
using Compilation = std::unique_ptr<ANeuralNetworksCompilation, decltype(&ANeuralNetworksCompilation_free)>;
using Execution = std::unique_ptr<ANeuralNetworksExecution, decltype(&ANeuralNetworksExecution_free)>;

void check(int code, const char* action) {
    if (code != ANEURALNETWORKS_NO_ERROR)
        throw std::runtime_error(std::string(action) + ": NNAPI error " + std::to_string(code));
}

std::string quote(const std::string& text) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << c;
    }
    out << '"';
    return out.str();
}

double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

long rss_kib() {
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream in(line.substr(6));
            long value = 0;
            in >> value;
            return value;
        }
    }
    return -1;
}

struct Workload {
    Model model{nullptr, ANeuralNetworksModel_free};
    std::string name;
    std::vector<float> input, output, w1, w2, b1, b2;
    uint32_t next = 0;
    int operations = 0;
    uint64_t weight_bytes = 0;

    uint32_t operand(std::vector<uint32_t> dims, int type = ANEURALNETWORKS_TENSOR_FLOAT32) {
        ANeuralNetworksOperandType desc{type, uint32_t(dims.size()), dims.data(), 0, 0};
        check(ANeuralNetworksModel_addOperand(model.get(), &desc), "addOperand");
        return next++;
    }
    uint32_t scalar(int32_t value) {
        auto id = operand({}, ANEURALNETWORKS_INT32);
        check(ANeuralNetworksModel_setOperandValue(model.get(), id, &value, sizeof(value)), "set scalar");
        return id;
    }
    uint32_t constant(std::vector<uint32_t> dims, const std::vector<float>& values) {
        auto id = operand(dims);
        check(ANeuralNetworksModel_setOperandValue(model.get(), id, values.data(), values.size()*sizeof(float)), "set weights");
        return id;
    }
    void operation(int type, std::vector<uint32_t> inputs, uint32_t output) {
        check(ANeuralNetworksModel_addOperation(model.get(), type, inputs.size(), inputs.data(), 1, &output), "addOperation");
        ++operations;
    }
};

void randomize(std::vector<float>& values, std::mt19937& rng) {
    std::uniform_real_distribution<float> distribution(-0.03f, 0.03f);
    for (auto& value : values) value = distribution(rng);
}

Workload make_workload(bool convolution, bool relaxed) {
    Workload work;
    work.name = convolution ? "conv2d_relu_64x64x32" : "fc_relu_fc_tokens128_width512_hidden2048";
    ANeuralNetworksModel* raw = nullptr;
    check(ANeuralNetworksModel_create(&raw), "createModel");
    work.model.reset(raw);
    std::mt19937 rng(42);
    uint32_t input, output;
    if (convolution) {
        work.input.resize(64*64*32);
        work.output.resize(work.input.size());
        work.w1.resize(32*3*3*32);
        work.b1.assign(32, 0);
        randomize(work.input, rng);
        randomize(work.w1, rng);
        input = work.operand({1, 64, 64, 32});
        auto weights = work.constant({32, 3, 3, 32}, work.w1);
        auto bias = work.constant({32}, work.b1);
        auto padding = work.scalar(ANEURALNETWORKS_PADDING_SAME);
        auto stride = work.scalar(1);
        auto relu = work.scalar(ANEURALNETWORKS_FUSED_RELU);
        output = work.operand({1, 64, 64, 32});
        work.operation(ANEURALNETWORKS_CONV_2D, {input, weights, bias, padding, stride, stride, relu}, output);
    } else {
        work.input.resize(128*512);
        work.output.resize(work.input.size());
        work.w1.resize(2048*512);
        work.w2.resize(512*2048);
        work.b1.assign(2048, 0);
        work.b2.assign(512, 0);
        randomize(work.input, rng);
        randomize(work.w1, rng);
        randomize(work.w2, rng);
        input = work.operand({128, 512});
        auto w1 = work.constant({2048, 512}, work.w1);
        auto b1 = work.constant({2048}, work.b1);
        auto relu = work.scalar(ANEURALNETWORKS_FUSED_RELU);
        auto hidden = work.operand({128, 2048});
        work.operation(ANEURALNETWORKS_FULLY_CONNECTED, {input, w1, b1, relu}, hidden);
        auto w2 = work.constant({512, 2048}, work.w2);
        auto b2 = work.constant({512}, work.b2);
        auto none = work.scalar(ANEURALNETWORKS_FUSED_NONE);
        output = work.operand({128, 512});
        work.operation(ANEURALNETWORKS_FULLY_CONNECTED, {hidden, w2, b2, none}, output);
    }
    work.weight_bytes = (work.w1.size()+work.w2.size()+work.b1.size()+work.b2.size())*sizeof(float);
    check(ANeuralNetworksModel_identifyInputsAndOutputs(work.model.get(), 1, &input, 1, &output), "identifyIO");
    check(ANeuralNetworksModel_relaxComputationFloat32toFloat16(work.model.get(), relaxed), "setRelaxed");
    check(ANeuralNetworksModel_finish(work.model.get()), "finishModel");
    return work;
}

double validate(const Workload& work, bool convolution) {
    double error = 0;
    if (convolution) {
        for (int y : {0, 31, 63}) for (int x : {0, 31, 63}) for (int out : {0, 15, 31}) {
            float value = 0;
            for (int ky=0; ky<3; ++ky) for (int kx=0; kx<3; ++kx) {
                int iy = y+ky-1, ix = x+kx-1;
                if (iy<0 || iy>=64 || ix<0 || ix>=64) continue;
                for (int in=0; in<32; ++in)
                    value += work.input[(iy*64+ix)*32+in] * work.w1[((out*3+ky)*3+kx)*32+in];
            }
            float actual = work.output[(y*64+x)*32+out];
            if (!std::isfinite(actual)) throw std::runtime_error("non-finite result");
            error = std::max(error, double(std::abs(std::max(0.f, value)-actual)));
        }
    } else {
        for (int token : {0, 127}) {
            std::vector<float> hidden(2048);
            for (int h=0; h<2048; ++h) {
                float value = 0;
                for (int k=0; k<512; ++k) value += work.input[token*512+k] * work.w1[h*512+k];
                hidden[h] = std::max(0.f, value);
            }
            for (int out : {0, 127, 511}) {
                float value = 0;
                for (int h=0; h<2048; ++h) value += hidden[h]*work.w2[out*2048+h];
                float actual = work.output[token*512+out];
                if (!std::isfinite(actual)) throw std::runtime_error("non-finite result");
                error = std::max(error, double(std::abs(value-actual)));
            }
        }
    }
    return error;
}

std::string benchmark(const ANeuralNetworksDevice* device, bool convolution,
                      bool relaxed, int warmup, int iterations) {
    auto work = make_workload(convolution, relaxed);
    bool supported[2] = {false, false};
    check(ANeuralNetworksModel_getSupportedOperationsForDevices(work.model.get(), &device, 1, supported), "supportedOperations");
    for (int i=0; i<work.operations; ++i)
        if (!supported[i]) return "{\"workload\":" + quote(work.name) + ",\"status\":\"unsupported\",\"all_ops_supported\":false}";
    auto start = Clock::now();
    ANeuralNetworksCompilation* raw = nullptr;
    check(ANeuralNetworksCompilation_createForDevices(work.model.get(), &device, 1, &raw), "compileForDevice");
    Compilation compilation(raw, ANeuralNetworksCompilation_free);
    check(ANeuralNetworksCompilation_setPreference(raw, ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER), "preference");
    check(ANeuralNetworksCompilation_finish(raw), "finishCompilation");
    double compile_ms = milliseconds(start);
    std::vector<double> samples;
    for (int i=0; i<warmup+iterations; ++i) {
        ANeuralNetworksExecution* raw_execution = nullptr;
        check(ANeuralNetworksExecution_create(raw, &raw_execution), "createExecution");
        Execution execution(raw_execution, ANeuralNetworksExecution_free);
        auto begin = Clock::now();
        check(ANeuralNetworksExecution_setInput(raw_execution, 0, nullptr, work.input.data(), work.input.size()*sizeof(float)), "setInput");
        check(ANeuralNetworksExecution_setOutput(raw_execution, 0, nullptr, work.output.data(), work.output.size()*sizeof(float)), "setOutput");
        check(ANeuralNetworksExecution_compute(raw_execution), "compute");
        double elapsed = milliseconds(begin);
        if (i>=warmup) samples.push_back(elapsed);
    }
    double error = validate(work, convolution);
    std::sort(samples.begin(), samples.end());
    double mean = std::accumulate(samples.begin(), samples.end(), 0.0)/samples.size();
    double p50 = (samples[(samples.size()-1)/2] + samples[samples.size()/2])/2;
    double p95 = samples[size_t(std::ceil(samples.size()*0.95))-1];
    std::ostringstream out;
    out << "{\"workload\":" << quote(work.name)
        << ",\"status\":" << quote(error<=0.002 ? "success" : "numerical_check_failed")
        << ",\"all_ops_supported\":true,\"precision\":" << quote(relaxed ? "fp32_allow_fp16" : "fp32")
        << ",\"compile_ms\":" << compile_ms << ",\"mean_ms\":" << mean
        << ",\"p50_ms\":" << p50 << ",\"p95_ms\":" << p95
        << ",\"iterations\":" << iterations << ",\"warmup\":" << warmup
        << ",\"constant_weight_bytes\":" << work.weight_bytes
        << ",\"max_abs_error_sampled\":" << error << ",\"rss_kib\":" << rss_kib() << "}";
    return out.str();
}

int main(int argc, char** argv) {
    try {
        int iterations=20, warmup=3;
        bool include_cpu=false, relaxed=true;
        for (int i=1; i<argc; ++i) {
            std::string arg=argv[i];
            if (arg=="--include-cpu") include_cpu=true;
            else if (arg=="--fp32") relaxed=false;
            else if ((arg=="--iterations" || arg=="--warmup") && i+1<argc) {
                int value=std::stoi(argv[++i]);
                if (arg=="--iterations") iterations=value; else warmup=value;
            } else throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
        if (iterations<1 || iterations>100 || warmup<0 || warmup>20) throw std::runtime_error("invalid run count");
        uint32_t count=0;
        check(ANeuralNetworks_getDeviceCount(&count), "getDeviceCount");
        std::cout << "{\"schema_version\":1,\"api\":\"NNAPI_deprecated_diagnostic_only\","
                  << "\"scope\":\"small random-weight FC/ReLU and Conv2D probes; not DiT, VAE, audio or video generation\","
                  << "\"cpu_included\":" << (include_cpu ? "true" : "false") << ",\"devices\":[";
        for (uint32_t i=0; i<count; ++i) {
            ANeuralNetworksDevice* device=nullptr;
            check(ANeuralNetworks_getDevice(i, &device), "getDevice");
            const char *name=nullptr, *version=nullptr;
            int32_t type=0;
            int64_t level=0;
            check(ANeuralNetworksDevice_getName(device, &name), "deviceName");
            check(ANeuralNetworksDevice_getVersion(device, &version), "deviceVersion");
            check(ANeuralNetworksDevice_getType(device, &type), "deviceType");
            check(ANeuralNetworksDevice_getFeatureLevel(device, &level), "featureLevel");
            bool hardware=type==ANEURALNETWORKS_DEVICE_GPU || type==ANEURALNETWORKS_DEVICE_ACCELERATOR;
            if (i) std::cout << ',';
            std::cout << "{\"name\":" << quote(name) << ",\"version\":" << quote(version)
                      << ",\"type\":" << type << ",\"feature_level\":" << level
                      << ",\"hardware_driver\":" << (hardware ? "true" : "false") << ",\"results\":[";
            if (hardware || (include_cpu && type==ANEURALNETWORKS_DEVICE_CPU)) {
                for (int test=0; test<2; ++test) {
                    if (test) std::cout << ',';
                    try { std::cout << benchmark(device, test==1, relaxed, warmup, iterations); }
                    catch (const std::exception& error) {
                        std::cout << "{\"status\":\"failed\",\"workload_index\":" << test
                                  << ",\"error\":" << quote(error.what()) << '}';
                    }
                }
            }
            std::cout << "]}";
        }
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        std::cout << "],\"process_peak_rss_kib\":" << usage.ru_maxrss << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

#pragma once
// ncnn Vulkan FP32/FP16 operator-chain validation. No trained video model.
#include <net.h>
#include <layer.h>
#include <gpu.h>
#include <command.h>
#include <allocator.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

using Clock=std::chrono::steady_clock;
const char* precision_name(int mode) {
    return mode==0 ? "fp32" : mode==1 ? "fp16_requested" : "fp16_storage_fp32_arithmetic_requested";
}
void check(int result, const char* what) { if (result) throw std::runtime_error(std::string(what)+": "+std::to_string(result)); }
std::string quote(const std::string& value) {
    std::string result="\"";
    for (char c:value) { if (c=='"'||c=='\\') result+='\\'; result+= c=='\n' ? ' ' : c; }
    return result+'"';
}
double elapsed(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
double percentile(std::vector<double> values, double q) {
    std::sort(values.begin(),values.end());
    if (q==.5) return (values[(values.size()-1)/2]+values[values.size()/2])/2;
    return values[size_t(std::ceil(q*values.size()))-1];
}

struct Definition {
    std::string type;
    ncnn::ParamDict parameters;
    std::vector<ncnn::Mat> weights;
};
struct Case {
    std::string name;
    ncnn::Mat input;
    std::vector<Definition> layers;
};
ncnn::Mat random_mat(int count, std::mt19937& rng) {
    ncnn::Mat values(count);
    std::uniform_real_distribution<float> distribution(-.03f,.03f);
    for (int i=0;i<count;++i) values[i]=distribution(rng);
    return values;
}
Definition projection(int input, int output, std::mt19937& rng) {
    Definition result; result.type="InnerProduct";
    result.parameters.set(0,output); result.parameters.set(1,0); result.parameters.set(2,input*output);
    result.weights.push_back(random_mat(input*output,rng)); return result;
}
Definition convolution(int channels, std::mt19937& rng) {
    Definition result; result.type="Convolution";
    result.parameters.set(0,channels); result.parameters.set(1,3);
    result.parameters.set(2,1); result.parameters.set(3,1); result.parameters.set(4,1);
    result.parameters.set(5,0); result.parameters.set(6,channels*channels*9);
    result.weights.push_back(random_mat(channels*channels*9,rng)); return result;
}
Case make_case(int id) {
    std::mt19937 rng(42);
    Case result;
    if (id==0) {
        result.name="ffn_tokens128_width512_hidden2048_gelu_approx";
        result.input=random_mat(128*512,rng).reshape(512,128);
        result.layers.push_back(projection(512,2048,rng));
        Definition gelu; gelu.type="GELU"; gelu.parameters.set(0,1);
        result.layers.push_back(gelu);
        result.layers.push_back(projection(2048,512,rng));
    } else if (id==1) {
        result.name="self_attention_tokens128_width256_heads4";
        result.input=random_mat(128*256,rng).reshape(256,128);
        Definition attention; attention.type="MultiHeadAttention";
        attention.parameters.set(0,256); attention.parameters.set(1,4);
        attention.parameters.set(2,256*256); attention.parameters.set(3,256); attention.parameters.set(4,256);
        attention.parameters.set(5,0);
        for (int i=0;i<4;++i) {
            attention.weights.push_back(random_mat(256*256,rng));
            ncnn::Mat bias(256); bias.fill(0.f); attention.weights.push_back(bias);
        }
        result.layers.push_back(attention);
    } else {
        result.name="decoder_2d_proxy_64x64x32_conv_swish_conv_not_wan_vae";
        result.input=random_mat(64*64*32,rng).reshape(64,64,32);
        result.layers.push_back(convolution(32,rng));
        Definition swish; swish.type="Swish"; result.layers.push_back(swish);
        result.layers.push_back(convolution(32,rng));
    }
    return result;
}

struct GpuOptions {
    const ncnn::VulkanDevice* device;
    ncnn::VkAllocator* blob;
    ncnn::VkAllocator* staging;
    ncnn::Option option;
    GpuOptions(const ncnn::VulkanDevice* gpu,int mode): device(gpu) {
        bool half=mode!=0;
        blob=device->acquire_blob_allocator(); staging=device->acquire_staging_allocator();
        option.use_vulkan_compute=true; option.use_packing_layout=true; option.num_threads=1;
        option.use_fp16_packed=half && device->info.support_fp16_packed();
        option.use_fp16_storage=half && device->info.support_fp16_storage();
        option.use_fp16_uniform=half && device->info.support_fp16_uniform();
        option.use_fp16_arithmetic=mode==1 && device->info.support_fp16_arithmetic();
        option.blob_vkallocator=blob; option.workspace_vkallocator=blob; option.staging_vkallocator=staging;
    }
    ~GpuOptions() { device->reclaim_blob_allocator(blob); device->reclaim_staging_allocator(staging); }
};
struct Layer {
    ncnn::Layer* layer=nullptr;
    ncnn::Option option;
    bool created=false;
    Layer(const Definition& definition,const ncnn::Option& opt,const ncnn::VulkanDevice* device):option(opt) {
        layer=device ? ncnn::create_layer_vulkan(definition.type.c_str()) : ncnn::create_layer_cpu(definition.type.c_str());
        if (!layer) throw std::runtime_error("unsupported layer: "+definition.type);
        try {
            check(layer->load_param(definition.parameters),"load_param");
            if (device && !layer->support_vulkan) throw std::runtime_error("no GPU support: "+definition.type);
            if (device) layer->vkdev=device;
            if (!definition.weights.empty()) {
                ncnn::ModelBinFromMatArray model(definition.weights.data());
                check(layer->load_model(model),"load_model");
            }
            check(layer->create_pipeline(option),"create_pipeline"); created=true;
            if (device) { ncnn::VkTransfer upload(device); check(layer->upload_model(upload,option),"upload_model"); check(upload.submit_and_wait(),"weight submit"); }
        } catch (...) { if (created) layer->destroy_pipeline(option); delete layer; layer=nullptr; throw; }
    }
    ~Layer() { if (layer) { if (created) layer->destroy_pipeline(option); delete layer; } }
};
ncnn::Mat cpu_reference(const Case& definition) {
    ncnn::Option option;
    option.use_vulkan_compute=false; option.use_packing_layout=false;
    option.use_fp16_packed=false; option.use_fp16_storage=false; option.use_fp16_arithmetic=false;
    option.num_threads=2;
    ncnn::Mat input=definition.input.clone();
    for (const auto& item:definition.layers) {
        Layer operation(item,option,nullptr);
        if (operation.layer->one_blob_only) {
            if (operation.layer->support_inplace) check(operation.layer->forward_inplace(input,option),"CPU inplace");
            else { ncnn::Mat output; check(operation.layer->forward(input,output,option),"CPU forward"); input=output; }
        } else {
            std::vector<ncnn::Mat> output(1);
            check(operation.layer->forward(std::vector<ncnn::Mat>{input},output,option),"CPU vector forward"); input=output[0];
        }
    }
    return input;
}
ncnn::Mat vanilla(ncnn::Mat input) {
    ncnn::Option option; option.use_vulkan_compute=false;
    if (input.elempack!=1) { ncnn::Mat output; ncnn::convert_packing(input,output,1,option); input=output; }
    if (input.elembits()==16) { ncnn::Mat output; ncnn::cast_float16_to_float32(input,output,option); input=output; }
    if (input.elembits()!=32) throw std::runtime_error("unexpected result scalar type");
    return input;
}


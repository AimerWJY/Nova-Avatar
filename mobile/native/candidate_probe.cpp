// Connected architecture feasibility and PyTorch weight parity; no pretrained video model.
#include "ncnn_probe_common.hpp"
#include <fstream>
#include <tuple>

constexpr int WIDTH=128, HISTORY=8, MEL=40, SIDE=256;
enum Node { IMAGE1, IMAGE_ACT1, IMAGE2, IMAGE_ACT2, POOL, IDENTITY,
    AUDIO, AUDIO_ACT, ATTENTION, FLATTEN, AUDIO_CONTEXT, ADD_STATE, ADD_IDENTITY,
    MOTION, MOTION_ACT, MODULATE, RESHAPE, ADD_SPATIAL, DECODER1, DECODER_ACT1,
    UPSAMPLE1, DECODER2, DECODER_ACT2, UPSAMPLE2, RGB, SIGMOID };

Definition simple(const char* type) { Definition d; d.type=type; return d; }
Definition conv(int input,int output,int stride,std::mt19937& rng) {
    auto d=simple("Convolution");
    d.parameters.set(0,output); d.parameters.set(1,3); d.parameters.set(2,1);
    d.parameters.set(3,stride); d.parameters.set(4,1); d.parameters.set(5,0);
    d.parameters.set(6,input*output*9); d.weights.push_back(random_mat(input*output*9,rng));
    return d;
}
Definition resize(int side) {
    auto d=simple("Interp"); d.parameters.set(0,2); // bilinear
    d.parameters.set(3,side); d.parameters.set(4,side); return d;
}
std::vector<Definition> architecture() {
    std::mt19937 rng(47);
    std::vector<Definition> d;
    d.push_back(conv(3,16,2,rng)); d.push_back(simple("Swish"));
    d.push_back(conv(16,32,2,rng)); d.push_back(simple("Swish"));
    auto pool=simple("Pooling"); pool.parameters.set(0,1); pool.parameters.set(4,1);
    d.push_back(pool); d.push_back(projection(32,WIDTH,rng));
    d.push_back(projection(MEL,WIDTH,rng)); d.push_back(simple("Swish"));
    auto attention=simple("MultiHeadAttention");
    attention.parameters.set(0,WIDTH); attention.parameters.set(1,4);
    attention.parameters.set(2,WIDTH*WIDTH); attention.parameters.set(3,WIDTH);
    attention.parameters.set(4,WIDTH); attention.parameters.set(5,0);
    for (int i=0;i<4;++i) {
        attention.weights.push_back(random_mat(WIDTH*WIDTH,rng));
        ncnn::Mat bias(WIDTH); bias.fill(0.f); attention.weights.push_back(bias);
    }
    d.push_back(attention); d.push_back(simple("Flatten"));
    d.push_back(projection(HISTORY*WIDTH,WIDTH,rng));
    auto add=simple("BinaryOp"); add.parameters.set(0,0); add.parameters.set(1,0);
    d.push_back(add); d.push_back(add);
    d.push_back(projection(WIDTH,WIDTH,rng)); d.push_back(simple("Swish"));
    d.push_back(projection(WIDTH,32,rng));
    auto reshape=simple("Reshape"); reshape.parameters.set(0,1);
    reshape.parameters.set(1,1); reshape.parameters.set(2,32);
    d.push_back(reshape); d.push_back(add);
    d.push_back(conv(32,32,1,rng)); d.push_back(simple("Swish"));
    d.push_back(resize(SIDE/2)); d.push_back(conv(32,16,1,rng));
    d.push_back(simple("Swish")); d.push_back(resize(SIDE));
    d.push_back(conv(16,3,1,rng)); d.push_back(simple("Sigmoid"));
    return d;
}
#include "candidate_fixture.hpp"

struct Graph {
    std::vector<std::unique_ptr<Layer>> operations;
    Graph(const std::vector<Definition>& d,const ncnn::Option& opt,const ncnn::VulkanDevice* gpu) {
        for (const auto& item:d) operations.push_back(std::make_unique<Layer>(item,opt,gpu));
    }
};
struct Cpu {
    Graph& graph; const ncnn::Option& option;
    ncnn::Mat operator()(int id,std::vector<ncnn::Mat> input) {
        auto* layer=graph.operations[id]->layer;
        if (layer->one_blob_only) {
            if (layer->support_inplace) { check(layer->forward_inplace(input[0],option),"CPU inplace"); return input[0]; }
            ncnn::Mat output; check(layer->forward(input[0],output,option),"CPU forward"); return output;
        }
        std::vector<ncnn::Mat> output(1);
        check(layer->forward(input,output,option),"CPU multi-input"); return output[0];
    }
};
struct Gpu {
    Graph& graph; const ncnn::Option& option; ncnn::VkCompute& command;
    ncnn::VkMat operator()(int id,std::vector<ncnn::VkMat> input) {
        auto* layer=graph.operations[id]->layer;
        if (layer->one_blob_only) {
            if (layer->support_inplace) { check(layer->forward_inplace(input[0],command,option),"GPU inplace"); return input[0]; }
            ncnn::VkMat output; check(layer->forward(input[0],output,command,option),"GPU forward"); return output;
        }
        std::vector<ncnn::VkMat> output(1);
        check(layer->forward(input,output,command,option),"GPU multi-input"); return output[0];
    }
};
template<class Tensor> struct IdentityCache { Tensor spatial,context; };
template<class Tensor,class Execute>
IdentityCache<Tensor> encode(Tensor image,Execute& run) {
    auto spatial=run(IMAGE_ACT1,{run(IMAGE1,{image})});
    spatial=run(IMAGE_ACT2,{run(IMAGE2,{spatial})});
    auto context=run(IDENTITY,{run(POOL,{spatial})});
    return {spatial,context};
}
template<class Tensor,class Execute>
std::pair<Tensor,Tensor> frame(Tensor audio,Tensor state,const IdentityCache<Tensor>& identity,Execute& run) {
    auto tokens=run(AUDIO_ACT,{run(AUDIO,{audio})});
    auto context=run(AUDIO_CONTEXT,{run(FLATTEN,{run(ATTENTION,{tokens})})});
    auto motion=run(ADD_IDENTITY,{run(ADD_STATE,{context,state}),identity.context});
    auto next=run(MOTION_ACT,{run(MOTION,{motion})});
    auto modulation=run(RESHAPE,{run(MODULATE,{next})});
    auto fused=run(ADD_SPATIAL,{identity.spatial,modulation});
    auto decoded=run(DECODER_ACT1,{run(DECODER1,{fused})});
    decoded=run(DECODER_ACT2,{run(DECODER2,{run(UPSAMPLE1,{decoded})})});
    auto rgb=run(SIGMOID,{run(RGB,{run(UPSAMPLE2,{decoded})})});
    return {rgb,next};
}
ncnn::VkMat upload(const ncnn::Mat& cpu,const ncnn::VulkanDevice* gpu,ncnn::VkCompute& command,const ncnn::Option& option) {
    ncnn::VkMat value; command.record_upload(cpu,value,option);
    int dimension=cpu.dims==1 ? cpu.w : cpu.dims==2 ? cpu.h : cpu.c;
    int pack=dimension%4==0 ? 4 : 1;
    if (value.elempack!=pack) { ncnn::VkMat packed; gpu->convert_packing(value,packed,pack,command,option); value=packed; }
    return value;
}
void finite(const ncnn::Mat& value) {
    if (value.empty()) throw std::runtime_error("empty output");
    for (int c=0;c<value.c;++c) for (int y=0;y<value.h;++y) {
        const float* row=value.channel(c).row(y);
        for (int x=0;x<value.w;++x) if (!std::isfinite(row[x])) throw std::runtime_error("non-finite graph output");
    }
}
struct Error {
    double max_abs=0,max_scaled=0; size_t elements=0;
    double absolute=1e-4,relative=.02;
    std::string largest_error_stage="none";
    void compare(const ncnn::Mat& actual,const ncnn::Mat& expected,const std::string& stage="unspecified") {
        finite(actual); finite(expected);
        if (actual.dims!=expected.dims || actual.w!=expected.w || actual.h!=expected.h || actual.c!=expected.c)
            throw std::runtime_error("CPU/GPU shape mismatch");
        for (int c=0;c<actual.c;++c) for (int y=0;y<actual.h;++y) {
            const float* a=actual.channel(c).row(y); const float* b=expected.channel(c).row(y);
            for (int x=0;x<actual.w;++x) {
                double error=std::abs(double(a[x])-b[x]);
                max_abs=std::max(max_abs,error);
                double scaled=error/(absolute+relative*std::abs(b[x]));
                if (scaled>max_scaled) { max_scaled=scaled; largest_error_stage=stage; }
                ++elements;
            }
        }
    }
};
long rss_kib() {
    std::ifstream file("/proc/self/status"); std::string key,line;
    while (file>>key) { if (key=="VmRSS:") { long value; file>>value; return value; } std::getline(file,line); }
    return -1;
}
std::string run_candidate(int frames,int warmup,const std::string& bundle,bool teacher_trained,int precision_mode) {
    auto process_start=Clock::now();
    check(ncnn::create_gpu_instance(),"create_gpu_instance");
    if (ncnn::get_gpu_count()<1) throw std::runtime_error("no Vulkan GPU");
    const auto* gpu=ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    if (gpu->info.type()!=0 && gpu->info.type()!=1) throw std::runtime_error("CPU/virtual Vulkan device rejected");
    if (precision_mode==2 && !gpu->info.support_fp16_storage()) throw std::runtime_error("FP16 storage unavailable");
    std::mt19937 rng(51);
    auto image=random_mat(SIDE*SIDE*3,rng).reshape(SIDE,SIDE,3);
    auto history=random_mat((frames+HISTORY-1)*MEL,rng);
    ncnn::Mat zero(WIDTH); zero.fill(0.f);
    auto definitions=architecture(); size_t parameters=0;
    std::unique_ptr<Fixture> fixture;
    if (!bundle.empty()) {
        fixture=std::make_unique<Fixture>(load_fixture(bundle,definitions));
        if (frames>fixture->frames) throw std::runtime_error("requested more frames than bundle contains");
        image=fixture->image; history=fixture->history;
    }
    for (const auto& layer:definitions) for (const auto& weights:layer.weights) parameters+=weights.total();
    GpuOptions options(gpu,precision_mode);
    auto setup_begin=Clock::now(); Graph graph(definitions,options.option,gpu);
    double setup_ms=elapsed(setup_begin);
    IdentityCache<ncnn::VkMat> identity;
    ncnn::Mat gpu_spatial,gpu_context;
    int image_bits=0;
    auto encode_begin=Clock::now();
    {
        ncnn::VkCompute command(gpu); Gpu run{graph,options.option,command};
        auto input=upload(image,gpu,command,options.option);
        image_bits=input.elembits(); identity=encode(input,run);
        if (fixture) {
            command.record_download(identity.spatial,gpu_spatial,options.option);
            command.record_download(identity.context,gpu_context,options.option);
        }
        check(command.submit_and_wait(),"identity encode");
    }
    double encode_ms=elapsed(encode_begin);
    auto initial_state=[&]() {
        ncnn::VkCompute command(gpu); auto state=upload(zero,gpu,command,options.option);
        check(command.submit_and_wait(),"state reset"); return state;
    };
    auto state=initial_state();
    int audio_bits=0,output_bits=0;
    auto infer=[&](int index) {
        auto begin=Clock::now(); ncnn::VkCompute command(gpu); Gpu run{graph,options.option,command};
        ncnn::Mat audio(MEL,HISTORY);
        std::copy((const float*)history+index*MEL,(const float*)history+(index+HISTORY)*MEL,(float*)audio);
        auto input=upload(audio,gpu,command,options.option); audio_bits=input.elembits();
        auto output=frame(input,state,identity,run); state=output.second; output_bits=output.first.elembits();
        ncnn::Mat rgb,next;
        command.record_download(output.first,rgb,options.option);
        command.record_download(state,next,options.option);
        check(command.submit_and_wait(),"frame inference");
        double ms=elapsed(begin);
        return std::make_tuple(vanilla(rgb),vanilla(next),ms);
    };
    // First output is measured before warmup; no CPU reference work precedes it.
    auto cold=infer(0); double cold_ms=elapsed(process_start);
    finite(std::get<0>(cold)); finite(std::get<1>(cold));
    if (std::get<0>(cold).w!=SIDE || std::get<0>(cold).h!=SIDE || std::get<0>(cold).c!=3)
        throw std::runtime_error("unexpected RGB output shape");
    for (int i=0;i<warmup;++i) { auto output=infer(0); finite(std::get<0>(output)); finite(std::get<1>(output)); }
    state=initial_state();
    std::vector<double> times; std::vector<ncnn::Mat> rgb_reference_targets,state_reference_targets;
    std::vector<long> rss_samples; auto stream_begin=Clock::now();
    int checked=std::min(frames,8);
    for (int i=0;i<frames;++i) {
        auto output=infer(i); times.push_back(std::get<2>(output));
        finite(std::get<0>(output)); finite(std::get<1>(output));
        if (i<checked) { rgb_reference_targets.push_back(std::get<0>(output)); state_reference_targets.push_back(std::get<1>(output)); }
        if (i%16==0 || i==frames-1) rss_samples.push_back(rss_kib());
    }
    double stream_ms=elapsed(stream_begin); rusage gpu_usage{}; getrusage(RUSAGE_SELF,&gpu_usage);
    // Replay reset checks that no hidden graph cache overrides the supplied recurrent state.
    state=initial_state(); auto reset=infer(0); Error reset_error;
    reset_error.compare(std::get<0>(reset),rgb_reference_targets[0]);
    reset_error.compare(std::get<1>(reset),state_reference_targets[0]);
    ncnn::Option cpu_option; cpu_option.use_vulkan_compute=false; cpu_option.num_threads=2;
    cpu_option.use_packing_layout=false; cpu_option.use_fp16_storage=false;
    cpu_option.use_fp16_packed=false; cpu_option.use_fp16_arithmetic=false;
    Graph cpu_graph(definitions,cpu_option,nullptr); Cpu cpu{cpu_graph,cpu_option};
    auto cpu_identity=encode(image.clone(),cpu); ncnn::Mat cpu_state=zero.clone(); Error error;
    Error torch_cpu_error,torch_gpu_error;
    torch_cpu_error.absolute=1e-5; torch_cpu_error.relative=1e-4;
    if (precision_mode==0) { torch_gpu_error.absolute=1e-5; torch_gpu_error.relative=1e-4; }
    if (fixture) {
        torch_cpu_error.compare(cpu_identity.spatial,fixture->spatial,"identity_spatial");
        torch_cpu_error.compare(cpu_identity.context,fixture->context,"identity_context");
        torch_gpu_error.compare(vanilla(gpu_spatial),fixture->spatial,"identity_spatial");
        torch_gpu_error.compare(vanilla(gpu_context),fixture->context,"identity_context");
    }
    for (int i=0;i<checked;++i) {
        ncnn::Mat audio(MEL,HISTORY);
        std::copy((const float*)history+i*MEL,(const float*)history+(i+HISTORY)*MEL,(float*)audio);
        auto output=frame(audio,cpu_state,cpu_identity,cpu); cpu_state=output.second;
        error.compare(rgb_reference_targets[i],output.first); error.compare(state_reference_targets[i],cpu_state);
        if (fixture) {
            torch_cpu_error.compare(output.first,fixture->rgb[i],"rgb_frame_"+std::to_string(i));
            torch_cpu_error.compare(cpu_state,fixture->state[i],"state_frame_"+std::to_string(i));
            torch_gpu_error.compare(rgb_reference_targets[i],fixture->rgb[i],"rgb_frame_"+std::to_string(i));
            torch_gpu_error.compare(state_reference_targets[i],fixture->state[i],"state_frame_"+std::to_string(i));
        }
    }
    int expected_bits=precision_mode==0 ? 32 : 16;
    bool precision=image_bits==expected_bits && audio_bits==expected_bits && output_bits==expected_bits;
    bool external_valid=!fixture || (torch_cpu_error.max_scaled<=1 && torch_gpu_error.max_scaled<=1);
    bool valid=error.max_scaled<=1 && reset_error.max_scaled<=1 && precision && external_valid;
    std::ostringstream out;
    out << "{\"schema_version\":1,\"api\":\"ncnn_Vulkan\",\"candidate\":\"recurrent_conditioned_rgb_v0\","
        << "\"trained\":" << (teacher_trained ? "true" : "false")
        << ",\"synthetic_inputs\":true,\"has_audio_frontend\":false,\"has_player\":false,"
        << "\"external_pytorch_reference\":{\"available\":" << (fixture ? "true" : "false")
        << ",\"passed\":" << (fixture && external_valid ? "true" : "false")
        << ",\"cpu_max_abs_error\":" << torch_cpu_error.max_abs
        << ",\"cpu_max_scaled_error\":" << torch_cpu_error.max_scaled
        << ",\"gpu_max_abs_error\":" << torch_gpu_error.max_abs
        << ",\"gpu_max_scaled_error\":" << torch_gpu_error.max_scaled
        << ",\"cpu_largest_error_stage\":" << quote(torch_cpu_error.largest_error_stage)
        << ",\"gpu_largest_error_stage\":" << quote(torch_gpu_error.largest_error_stage)
        << ",\"checked_elements_per_backend\":" << torch_cpu_error.elements << "},"
        << "\"parameter_count\":" << parameters << ",\"output_width\":" << SIDE << ",\"output_height\":" << SIDE
        << ",\"audio_history_chunks\":" << HISTORY << ",\"audio_features_per_chunk\":" << MEL
        << ",\"state_elements\":" << WIDTH << ",\"gpu_phase_peak_rss_kib\":" << gpu_usage.ru_maxrss
        << ",\"rss_samples_kib\":[";
    for (size_t i=0;i<rss_samples.size();++i) { if (i) out<<','; out<<rss_samples[i]; }
    out << "],\"devices\":[{\"name\":" << quote(gpu->info.device_name())
        << ",\"hardware_driver\":true,\"results\":[{\"workload\":\"connected_candidate_rgb_with_recurrent_state\","
        << "\"precision\":" << quote(precision_name(precision_mode)) << ",\"fp16_arithmetic_option\":false,"
        << "\"status\":" << quote(valid ? "success" : "numerical_or_precision_check_failed")
        << ",\"observed_gpu_image_bits\":" << image_bits << ",\"observed_gpu_audio_bits\":" << audio_bits
        << ",\"observed_gpu_output_bits\":" << output_bits
        << ",\"frames\":" << frames << ",\"warmup\":" << warmup << ",\"target_fps\":16"
        << ",\"setup_and_weight_upload_ms\":" << setup_ms << ",\"identity_encode_ms\":" << encode_ms
        << ",\"cold_first_rgb_ms\":" << cold_ms << ",\"warm_first_rgb_ms\":" << times[0]
        << ",\"p50_ms\":" << percentile(times,.5) << ",\"p95_ms\":" << percentile(times,.95)
        << ",\"max_ms\":" << *std::max_element(times.begin(),times.end()) << ",\"stream_wall_ms\":" << stream_ms
        << ",\"all_frames_finite\":true,\"cpu_reference_frames\":" << checked
        << ",\"cpu_reference_checked_elements\":" << error.elements << ",\"max_abs_error\":" << error.max_abs
        << ",\"max_scaled_error\":" << error.max_scaled
        << ",\"state_reset_max_scaled_error\":" << reset_error.max_scaled << "}]}]}";
    return out.str();
}
int main(int argc,char** argv) {
    try {
        int frames=160,warmup=3,teacher_trained=0,precision_mode=2; std::string bundle;
        for (int i=1;i<argc;++i) {
            std::string arg=argv[i]; if (i+1>=argc) throw std::runtime_error("missing argument value");
            if (arg=="--iterations") frames=std::stoi(argv[++i]);
            else if (arg=="--warmup") warmup=std::stoi(argv[++i]);
            else if (arg=="--bundle") bundle=argv[++i];
            else if (arg=="--teacher-trained") teacher_trained=std::stoi(argv[++i]);
            else if (arg=="--precision-mode") precision_mode=std::stoi(argv[++i]);
            else throw std::runtime_error("unknown argument");
        }
        if (frames<8 || frames>160 || warmup<0 || warmup>20) throw std::runtime_error("frames must be 8..160");
        if (teacher_trained<0 || teacher_trained>1 || (teacher_trained && bundle.empty()))
            throw std::runtime_error("teacher-trained flag requires a checkpoint bundle");
        if (precision_mode!=0 && precision_mode!=2) throw std::runtime_error("precision mode must be 0 or 2");
        std::cout << run_candidate(frames,warmup,bundle,teacher_trained!=0,precision_mode) << '\n';
        ncnn::destroy_gpu_instance(); return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

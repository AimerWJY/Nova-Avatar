// Actual converted v2 Net graphs. Vulkan may include CPU fallback; report capability explicitly.
#include "ncnn_probe_common.hpp"
#include <pipelinecache.h>
#include <fstream>
#include <tuple>

uint32_t read_u32(std::ifstream& file) {
    unsigned char b[4]; if (!file.read(reinterpret_cast<char*>(b),4)) throw std::runtime_error("truncated fixture integer");
    return uint32_t(b[0])|uint32_t(b[1])<<8|uint32_t(b[2])<<16|uint32_t(b[3])<<24;
}
ncnn::Mat read_floats(std::ifstream& file,int count) {
    ncnn::Mat value(count);
    if (!file.read(reinterpret_cast<char*>((float*)value),size_t(count)*4)) throw std::runtime_error("truncated fixture floats");
    for (int i=0;i<count;++i) if (!std::isfinite(value[i])) throw std::runtime_error("non-finite fixture");
    return value;
}
struct Fixture {
    ncnn::Mat image,audio,spatial,identity;
    std::vector<ncnn::Mat> rgb,state;
    int count;
    explicit Fixture(const std::string& path) {
        static_assert(__BYTE_ORDER__==__ORDER_LITTLE_ENDIAN__,"little endian required");
        std::ifstream file(path,std::ios::binary|std::ios::ate);
        if (!file || file.tellg()>128*1024*1024) throw std::runtime_error("missing or oversized fixture");
        file.seekg(0); char magic[8];
        if (!file.read(magic,8) || std::string(magic,8)!="NOVAV202" || read_u32(file)!=2) throw std::runtime_error("fixture schema mismatch");
        count=int(read_u32(file)); if (count<8 || count>160) throw std::runtime_error("fixture frame bounds");
        image=read_floats(file,3*256*256).reshape(256,256,3);
        audio=read_floats(file,(count+7)*40);
        spatial=read_floats(file,32*64*64).reshape(64,64,32); identity=read_floats(file,128);
        for (int i=0;i<count;++i) { rgb.push_back(read_floats(file,3*256*256).reshape(256,256,3)); state.push_back(read_floats(file,128)); }
        if (file.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("trailing fixture data");
    }
};
struct Errors {
    double max_abs=0,max_scaled=0,largest_actual=0,largest_expected=0; size_t count=0; std::string stage="none";
    int largest_c=0,largest_y=0,largest_x=0;
    void compare(ncnn::Mat actual,ncnn::Mat expected,const std::string& label) {
        actual=vanilla(actual); expected=vanilla(expected);
        if (actual.w!=expected.w || actual.h!=expected.h || actual.c!=expected.c || actual.d!=expected.d)
            throw std::runtime_error("output shape mismatch: "+label);
        for (int c=0;c<actual.c;++c) for (int y=0;y<actual.h;++y) {
            const float* a=actual.channel(c).row(y); const float* e=expected.channel(c).row(y);
            for (int x=0;x<actual.w;++x) {
                if (!std::isfinite(a[x])) throw std::runtime_error("non-finite output: "+label);
                double difference=std::abs(double(a[x])-e[x]), scaled=difference/(1e-5+1e-4*std::abs(e[x]));
                max_abs=std::max(max_abs,difference);
                if (scaled>max_scaled) { max_scaled=scaled; stage=label; largest_actual=a[x]; largest_expected=e[x]; largest_c=c; largest_y=y; largest_x=x; }
                ++count;
            }
        }
    }
};
void dump_intermediates(ncnn::Net& net,ncnn::Extractor& ex,const std::string& directory) {
    std::ostringstream metadata; metadata<<"["; bool first=true; int number=0;
    for (const auto* layer:net.layers()) {
        const auto& kind=layer->type;
        if (kind!="Input" && kind!="Gemm" && kind!="InnerProduct" && kind!="Convolution" && kind!="GridSample"
            && kind!="Eltwise" && kind!="TanH" && kind!="MatMul" && kind!="Interp" && kind!="Clip") continue;
        for (int index:layer->tops) {
            const auto& blob=net.blobs()[index]; ncnn::Mat value;
            check(ex.extract(blob.name.c_str(),value),"diagnostic extraction"); value=vanilla(value);
            std::string name="node_"+std::to_string(number++)+".f32";
            std::ofstream file(directory+"/"+name,std::ios::binary);
            if (!file) throw std::runtime_error("cannot open diagnostic output");
            for (int c=0;c<value.c;++c) for (int z=0;z<value.d;++z) for (int y=0;y<value.h;++y)
                file.write(reinterpret_cast<const char*>(value.channel(c).depth(z).row(y)),size_t(value.w)*4);
            if (!file) throw std::runtime_error("cannot write diagnostic output");
            if (!first) metadata<<','; first=false;
            metadata<<"{\"layer\":"<<quote(layer->name)<<",\"type\":"<<quote(kind)<<",\"blob\":"<<quote(blob.name)
                    <<",\"file\":"<<quote(name)<<",\"shape\":["<<value.c<<','<<value.d<<','<<value.h<<','<<value.w<<"]}";
        }
    }
    metadata<<"]"; std::ofstream file(directory+"/nodes.json"); file<<metadata.str();
    if (!file) throw std::runtime_error("cannot write diagnostic manifest");
}
void load_net(ncnn::Net& net,const std::string& prefix,const ncnn::VulkanDevice* gpu) {
    net.opt.num_threads=2; net.opt.use_vulkan_compute=gpu!=nullptr;
    net.opt.use_fp16_packed=false; net.opt.use_fp16_storage=false;
    net.opt.use_fp16_uniform=false; net.opt.use_fp16_arithmetic=false;
    if (gpu) net.set_vulkan_device(gpu);
    check(net.load_param((prefix+".ncnn.param").c_str()),"load ncnn param");
    check(net.load_model((prefix+".ncnn.bin").c_str()),"load ncnn weights");
}
std::string unsupported(ncnn::Net& net,const std::string& graph,int& compute_count) {
    std::ostringstream out; bool first=true;
    for (const auto* layer:net.layers()) {
        if (!layer->support_vulkan) {
            bool structural=layer->type=="Input" || layer->type=="Split" || layer->type=="MemoryData";
            if (!structural) ++compute_count;
            if (!first) out<<','; first=false;
            out<<"{\"graph\":"<<quote(graph)<<",\"name\":"<<quote(layer->name)<<",\"type\":"<<quote(layer->type)
               <<",\"structural_only\":"<<(structural ? "true" : "false")<<'}';
        }
    }
    return out.str();
}
std::string run(const std::string& directory,const std::string& fixture_path,int frames,int warmup,bool use_gpu,
                int diagnostic_frame,const std::string& diagnostic_dir,const std::string& cache_mode) {
    auto begin=Clock::now();
    Fixture fixture(fixture_path); if (frames==0) frames=fixture.count;
    const double fixture_read_ms=elapsed(begin);
    if (frames>fixture.count || frames<8) throw std::runtime_error("frames outside fixture bounds");
    if (diagnostic_frame>=frames) throw std::runtime_error("diagnostic frame outside sequence");
    const ncnn::VulkanDevice* gpu=nullptr;
    const ncnn::PipelineCache* cache=nullptr;
    double cache_load_ms=0;
    const std::string cache_path=directory+"/pipeline.cache";
    if (use_gpu) {
        check(ncnn::create_gpu_instance(),"GPU init");
        if (ncnn::get_gpu_count()<1) throw std::runtime_error("no Vulkan GPU");
        gpu=ncnn::get_gpu_device(ncnn::get_default_gpu_index());
        if (gpu->info.type()!=0 && gpu->info.type()!=1) throw std::runtime_error("virtual/CPU Vulkan device rejected");
        if (!cache_mode.empty()) {
            cache=gpu->get_pipeline_cache();
            if (!cache) throw std::runtime_error("ncnn pipeline cache unavailable");
            if (cache_mode=="restore") {
                auto start=Clock::now();
                check(cache->load_cache(cache_path.c_str()),"restore ncnn pipeline cache");
                cache_load_ms=elapsed(start);
            }
        }
    }
    ncnn::Net encoder,decoder;
    auto setup=Clock::now(); load_net(encoder,directory+"/encoder",gpu); load_net(decoder,directory+"/frame",gpu);
    double setup_ms=elapsed(setup);
    int no_gpu_compute=0;
    std::string enc_unsupported=unsupported(encoder,"encoder",no_gpu_compute);
    std::string dec_unsupported=unsupported(decoder,"frame",no_gpu_compute);
    Errors errors; ncnn::Mat spatial,identity;
    auto encode=[&]() {
        auto ex=encoder.create_extractor(); check(ex.input("in0",fixture.image),"encoder input");
        check(ex.extract("out0",spatial),"spatial extraction"); check(ex.extract("out1",identity),"identity extraction");
        spatial=vanilla(spatial).reshape(64,64,32); identity=vanilla(identity).reshape(128);
    };
    auto enc_start=Clock::now(); encode();
    double encode_ms=elapsed(enc_start);
    errors.compare(spatial,fixture.spatial,"identity_spatial"); errors.compare(identity,fixture.identity,"identity_context");
    auto infer=[&](int index,const ncnn::Mat& state) {
        auto clock=Clock::now(); ncnn::Mat audio(40,8);
        std::copy((const float*)fixture.audio+index*40,(const float*)fixture.audio+(index+8)*40,(float*)audio);
        auto ex=decoder.create_extractor();
        if (index==diagnostic_frame) ex.set_light_mode(false);
        check(ex.input("in0",fixture.image),"image input"); check(ex.input("in1",spatial),"spatial input");
        check(ex.input("in2",identity),"identity input"); check(ex.input("in3",audio),"audio input"); check(ex.input("in4",state),"state input");
        ncnn::Mat rgb,next; check(ex.extract("out0",rgb),"RGB extraction"); check(ex.extract("out1",next),"state extraction");
        double ms=elapsed(clock);
        if (index==diagnostic_frame) dump_intermediates(decoder,ex,diagnostic_dir);
        return std::make_tuple(vanilla(rgb),vanilla(next).reshape(128),ms);
    };
    ncnn::Mat zero(128); zero.fill(0.f);
    auto first=infer(0,zero); double cold_ms=elapsed(begin);
    errors.compare(std::get<0>(first),fixture.rgb[0],"first_rgb");
    for (int i=0;i<warmup;++i) infer(0,zero);
    ncnn::Mat state=zero.clone(); std::vector<double> durations; auto stream_start=Clock::now();
    for (int i=0;i<frames;++i) {
        auto result=infer(i,state); state=std::get<1>(result); durations.push_back(std::get<2>(result));
        errors.compare(std::get<0>(result),fixture.rgb[i],"rgb_"+std::to_string(i));
        errors.compare(state,fixture.state[i],"state_"+std::to_string(i));
    }
    double stream_ms=elapsed(stream_start);
    auto replay=infer(0,zero); Errors reset;
    reset.compare(std::get<0>(replay),std::get<0>(first),"reset_rgb");
    reset.compare(std::get<1>(replay),std::get<1>(first),"reset_state");
    double resident_first_rgb_ms=0;
    if (diagnostic_frame<0) {
        auto request=Clock::now();
        encode();
        auto resident=infer(0,zero);
        resident_first_rgb_ms=elapsed(request);
        reset.compare(std::get<0>(resident),std::get<0>(first),"resident_rgb");
        reset.compare(std::get<1>(resident),std::get<1>(first),"resident_state");
    }
    if (cache_mode=="save" && errors.max_scaled<=1 && reset.max_scaled<=1)
        check(cache->save_cache(cache_path.c_str()),"save ncnn pipeline cache");
    rusage usage{}; getrusage(RUSAGE_SELF,&usage);
    std::ostringstream out;
    out<<"{\"schema_version\":1,\"architecture\":\"mouth_refinement_v2\",\"runtime\":\"ncnn-20261020\","
       <<"\"vulkan_requested\":"<<(use_gpu ? "true" : "false")<<",\"device\":"<<quote(gpu ? gpu->info.device_name() : "CPU")
       <<",\"compute_layers_without_vulkan_support\":"<<no_gpu_compute<<",\"layers_without_vulkan_support\":["<<enc_unsupported;
    if (!enc_unsupported.empty() && !dec_unsupported.empty()) out<<',';
    out<<dec_unsupported<<"],\"status\":"<<quote(errors.max_scaled<=1 && reset.max_scaled<=1 ? "numerical_validation_passed" : "numerical_validation_failed")
       <<",\"frames\":"<<frames<<",\"warmup\":"<<warmup<<",\"precision\":\"fp32\",\"all_outputs_finite\":true"
       <<",\"checked_elements\":"<<errors.count<<",\"max_abs_error\":"<<errors.max_abs<<",\"max_scaled_error\":"<<errors.max_scaled
       <<",\"largest_error_stage\":"<<quote(errors.stage)<<",\"state_reset_max_scaled_error\":"<<reset.max_scaled
       <<",\"largest_error_actual\":"<<errors.largest_actual<<",\"largest_error_expected\":"<<errors.largest_expected
       <<",\"largest_error_location\":["<<errors.largest_c<<','<<errors.largest_y<<','<<errors.largest_x<<"]"
       <<",\"diagnostic_frame\":"<<diagnostic_frame
       <<",\"pipeline_cache_mode\":"<<quote(cache_mode.empty() ? "off" : cache_mode)
       <<",\"pipeline_cache_load_ms\":"<<cache_load_ms
       <<",\"resident_first_rgb_ms\":"<<resident_first_rgb_ms
       <<",\"fixture_read_ms\":"<<fixture_read_ms
       <<",\"setup_and_weight_upload_ms\":"<<setup_ms<<",\"identity_encode_ms\":"<<encode_ms<<",\"native_first_rgb_ms\":"<<cold_ms
       <<",\"p50_ms\":"<<percentile(durations,.5)<<",\"p95_ms\":"<<percentile(durations,.95)
       <<",\"max_ms\":"<<*std::max_element(durations.begin(),durations.end())<<",\"stream_wall_including_validation_ms\":"<<stream_ms
       <<",\"process_peak_rss_kib\":"<<usage.ru_maxrss<<",\"phone_audio_frontend\":false,\"has_player\":false,\"video_quality_validated\":false}";
    return out.str();
}
int main(int argc,char** argv) {
    try {
        std::string directory,fixture,diagnostic_dir,cache_mode; int frames=0,warmup=3,diagnostic_frame=-1; bool gpu=true;
        for (int i=1;i<argc;++i) {
            std::string arg=argv[i]; if (i+1>=argc) throw std::runtime_error("missing argument value");
            if (arg=="--directory") directory=argv[++i]; else if (arg=="--fixture") fixture=argv[++i];
            else if (arg=="--frames") frames=std::stoi(argv[++i]); else if (arg=="--warmup") warmup=std::stoi(argv[++i]);
            else if (arg=="--gpu") gpu=std::stoi(argv[++i])!=0;
            else if (arg=="--diagnostic-frame") diagnostic_frame=std::stoi(argv[++i]);
            else if (arg=="--diagnostic-dir") diagnostic_dir=argv[++i];
            else if (arg=="--pipeline-cache") cache_mode=argv[++i];
            else throw std::runtime_error("unknown argument");
        }
        if (directory.empty() || fixture.empty() || frames<0 || frames>160 || warmup<0 || warmup>20) throw std::runtime_error("invalid arguments");
        if (diagnostic_frame < -1 || (diagnostic_frame>=0 && diagnostic_dir.empty())) throw std::runtime_error("invalid diagnostic arguments");
        if (!cache_mode.empty() && (!gpu || diagnostic_frame>=0 || (cache_mode!="save" && cache_mode!="restore")))
            throw std::runtime_error("pipeline cache requires Vulkan without diagnostics");
        std::cout<<run(directory,fixture,frames,warmup,gpu,diagnostic_frame,diagnostic_dir,cache_mode)<<'\n';
        if (gpu) ncnn::destroy_gpu_instance(); return 0;
    } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}

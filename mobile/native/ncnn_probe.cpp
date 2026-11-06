#include "ncnn_probe_common.hpp"
std::string benchmark(const ncnn::VulkanDevice* gpu,const Case& definition,int mode,int warmup,int iterations) {
    bool half=mode!=0;
    GpuOptions options(gpu,mode);
    if ((half && !options.option.use_fp16_storage) || (mode==1 && !options.option.use_fp16_arithmetic))
        return "{\"workload\":"+quote(definition.name)+",\"precision\":"+quote(precision_name(mode))+",\"status\":\"unsupported\"}";
    ncnn::Mat reference=cpu_reference(definition);
    auto start=Clock::now();
    std::vector<std::unique_ptr<Layer>> layers;
    for (const auto& item:definition.layers) layers.push_back(std::make_unique<Layer>(item,options.option,gpu));
    double setup_ms=elapsed(start);
    std::vector<double> durations;
    ncnn::Mat result;
    int observed_bits=0;
    for (int i=0;i<warmup+iterations;++i) {
        auto begin=Clock::now();
        ncnn::VkCompute command(gpu);
        ncnn::VkMat input;
        command.record_upload(definition.input,input,options.option);
        if (layers[0]->layer->support_vulkan_packing) {
            int count=definition.input.dims==1 ? definition.input.w : definition.input.dims==2 ? definition.input.h : definition.input.c;
            int pack=count%4==0 ? 4 : 1;
            if (input.elempack!=pack) { ncnn::VkMat packed; gpu->convert_packing(input,packed,pack,command,options.option); input=packed; }
        }
        observed_bits=int(input.elemsize*8/input.elempack);
        for (const auto& operation:layers) {
            if (operation->layer->one_blob_only) {
                if (operation->layer->support_inplace) check(operation->layer->forward_inplace(input,command,options.option),"GPU inplace");
                else { ncnn::VkMat output; check(operation->layer->forward(input,output,command,options.option),"GPU forward"); input=output; }
            } else {
                std::vector<ncnn::VkMat> output(1);
                check(operation->layer->forward(std::vector<ncnn::VkMat>{input},output,command,options.option),"GPU vector forward"); input=output[0];
            }
        }
        command.record_download(input,result,options.option);
        check(command.submit_and_wait(),"inference submit");
        if (i>=warmup) durations.push_back(elapsed(begin));
    }
    result=vanilla(result); reference=vanilla(reference);
    if (result.w!=reference.w || result.h!=reference.h || result.d!=reference.d || result.c!=reference.c)
        throw std::runtime_error("output shape differs from FP32 CPU reference");
    double maximum=0,scaled=0,squared=0; size_t count=0;
    for (int c=0;c<result.c;++c) for (int z=0;z<result.d;++z) for (int y=0;y<result.h;++y) {
        const float* actual=result.channel(c).depth(z).row(y);
        const float* expected=reference.channel(c).depth(z).row(y);
        for (int x=0;x<result.w;++x) {
            if (!std::isfinite(expected[x])) throw std::runtime_error("non-finite FP32 CPU reference");
            if (!std::isfinite(actual[x])) throw std::runtime_error("non-finite GPU result at channel="+std::to_string(c)+
                ", depth="+std::to_string(z)+", row="+std::to_string(y)+", column="+std::to_string(x));
            double difference=std::abs(double(actual[x])-expected[x]);
            double tolerance=(half ? 1e-4 : 1e-6)+(half ? .02 : .001)*std::abs(expected[x]);
            maximum=std::max(maximum,difference); scaled=std::max(scaled,difference/tolerance);
            squared+=difference*difference; ++count;
        }
    }
    bool precision_ok=observed_bits==(half ? 16 : 32);
    std::sort(durations.begin(),durations.end());
    std::ostringstream out;
    out << "{\"workload\":" << quote(definition.name) << ",\"precision\":" << quote(precision_name(mode))
        << ",\"observed_gpu_input_bits\":" << observed_bits << ",\"fp16_arithmetic_option\":" << (options.option.use_fp16_arithmetic ? "true" : "false")
        << ",\"status\":" << quote(scaled<=1 && precision_ok ? "success" : "numerical_or_precision_check_failed")
        << ",\"setup_and_weight_upload_ms\":" << setup_ms
        << ",\"p50_ms\":" << (durations[(durations.size()-1)/2]+durations[durations.size()/2])/2
        << ",\"p95_ms\":" << durations[size_t(std::ceil(.95*durations.size()))-1]
        << ",\"iterations\":" << iterations << ",\"warmup\":" << warmup
        << ",\"max_abs_error_all_elements\":" << maximum << ",\"max_scaled_error_all_elements\":" << scaled
        << ",\"rmse\":" << std::sqrt(squared/count) << ",\"checked_elements\":" << count << "}";
    return out.str();
}

int main(int argc,char** argv) {
    try {
        int iterations=20,warmup=3;
        for (int i=1;i<argc;++i) {
            std::string arg=argv[i];
            if (i+1>=argc) throw std::runtime_error("missing argument value");
            if (arg=="--iterations") iterations=std::stoi(argv[++i]);
            else if (arg=="--warmup") warmup=std::stoi(argv[++i]);
            else throw std::runtime_error("unknown argument");
        }
        if (iterations<1 || iterations>100 || warmup<0 || warmup>20) throw std::runtime_error("invalid iterations");
        check(ncnn::create_gpu_instance(),"create_gpu_instance");
        if (ncnn::get_gpu_count()<1) throw std::runtime_error("no Vulkan GPU");
        const ncnn::VulkanDevice* gpu=ncnn::get_gpu_device(ncnn::get_default_gpu_index());
        if (gpu->info.type()!=0 && gpu->info.type()!=1) throw std::runtime_error("CPU/virtual Vulkan devices rejected");
        std::cout << "{\"schema_version\":1,\"api\":\"ncnn_Vulkan\",\"runtime_version\":\"20261020\","
                  << "\"cpu_included\":false,\"scope\":\"random-weight FFN, Attention and 2D decoder proxy; no trained Wan model\",\"devices\":[{"
                  << "\"name\":" << quote(gpu->info.device_name()) << ",\"hardware_driver\":true,\"type\":" << gpu->info.type()
                  << ",\"fp16_storage_supported\":" << (gpu->info.support_fp16_storage() ? "true" : "false")
                  << ",\"fp16_arithmetic_supported\":" << (gpu->info.support_fp16_arithmetic() ? "true" : "false") << ",\"results\":[";
        bool first=true;
        for (int id=0;id<3;++id) for (int mode:{0,1,2}) {
            if (!first) std::cout << ','; first=false;
            try { std::cout << benchmark(gpu,make_case(id),mode,warmup,iterations); }
            catch (const std::exception& error) {
                std::cout << "{\"status\":\"failed\",\"case_index\":" << id << ",\"precision\":" << quote(precision_name(mode))
                          << ",\"workload\":" << quote(make_case(id).name)
                          << ",\"error\":" << quote(error.what()) << '}';
            }
        }
        std::cout << "]}],\"operator_support\":[";
        ncnn::Layer* conv3d=ncnn::create_layer_vulkan("Convolution3D");
        std::cout << "{\"operator\":\"Convolution3D\",\"vulkan_factory_available\":" << (conv3d ? "true" : "false")
                  << ",\"note\":\"factory probe only; actual Wan VAE not converted or tested\"}]";
        delete conv3d;
        rusage usage{}; getrusage(RUSAGE_SELF,&usage);
        std::cout << ",\"process_peak_rss_kib\":" << usage.ru_maxrss << "}\n";
        ncnn::destroy_gpu_instance(); return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

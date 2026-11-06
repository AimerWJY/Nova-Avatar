// GPU compute transport and numerical probe. No model weights or video generation.
#include <vulkan/vulkan.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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
void check(VkResult result, const char* action) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(action)+": Vulkan error "+std::to_string(result));
}
std::string quote(const std::string& text) {
    std::ostringstream out; out << '"';
    for (unsigned char c : text) {
        if (c=='"' || c=='\\') out << '\\' << c;
        else if (c<32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << c;
    }
    out << '"'; return out.str();
}
double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}

struct Context {
    VkInstance instance=VK_NULL_HANDLE;
    VkPhysicalDevice physical=VK_NULL_HANDLE;
    VkDevice device=VK_NULL_HANDLE;
    VkQueue queue=VK_NULL_HANDLE;
    uint32_t family=0, timestamp_bits=0;
    VkPhysicalDeviceProperties properties{};
    VkDescriptorSetLayout set_layout=VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout=VK_NULL_HANDLE;
    VkShaderModule shader=VK_NULL_HANDLE;
    VkPipeline pipeline=VK_NULL_HANDLE;
    VkCommandPool command_pool=VK_NULL_HANDLE;
    double pipeline_ms=0;
    ~Context() {
        if (device) {
            vkDeviceWaitIdle(device);
            if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
            if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
            if (shader) vkDestroyShaderModule(device, shader, nullptr);
            if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            if (set_layout) vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }
    void initialize(const std::string& shader_path) {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName="Nova GPU probe"; app.apiVersion=VK_API_VERSION_1_1;
        VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; create.pApplicationInfo=&app;
        check(vkCreateInstance(&create, nullptr, &instance), "createInstance");
        uint32_t count=0;
        check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "physicalDeviceCount");
        std::vector<VkPhysicalDevice> candidates(count);
        check(vkEnumeratePhysicalDevices(instance, &count, candidates.data()), "physicalDevices");
        for (auto candidate : candidates) {
            VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(candidate, &props);
            if (props.deviceType!=VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && props.deviceType!=VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
            uint32_t queues=0; vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queues, nullptr);
            std::vector<VkQueueFamilyProperties> families(queues);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queues, families.data());
            for (uint32_t i=0; i<queues; ++i) if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                physical=candidate; family=i; timestamp_bits=families[i].timestampValidBits; properties=props; break;
            }
            if (physical) break;
        }
        if (!physical) throw std::runtime_error("no hardware GPU compute device; CPU Vulkan implementations rejected");
        if (properties.limits.maxComputeWorkGroupInvocations<256 || properties.limits.maxComputeWorkGroupSize[0]<16 || properties.limits.maxComputeWorkGroupSize[1]<16)
            throw std::runtime_error("GPU does not support the 16x16 workgroup");
        float priority=1;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex=family; qi.queueCount=1; qi.pQueuePriorities=&priority;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; di.queueCreateInfoCount=1; di.pQueueCreateInfos=&qi;
        check(vkCreateDevice(physical, &di, nullptr, &device), "createDevice");
        vkGetDeviceQueue(device, family, 0, &queue);
        VkDescriptorSetLayoutBinding bindings[3]{};
        for (int i=0; i<3; ++i) {
            bindings[i].binding=i; bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount=1; bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        sli.bindingCount=3; sli.pBindings=bindings;
        check(vkCreateDescriptorSetLayout(device, &sli, nullptr, &set_layout), "descriptorLayout");
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount=1; pli.pSetLayouts=&set_layout; pli.pushConstantRangeCount=1; pli.pPushConstantRanges=&range;
        check(vkCreatePipelineLayout(device, &pli, nullptr, &pipeline_layout), "pipelineLayout");
        std::ifstream file(shader_path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("cannot read SPIR-V shader");
        auto length=file.tellg();
        if (length<=0 || size_t(length)%4) throw std::runtime_error("invalid shader length");
        std::vector<uint32_t> code(size_t(length)/4); file.seekg(0); file.read(reinterpret_cast<char*>(code.data()), length);
        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; sm.codeSize=size_t(length); sm.pCode=code.data();
        check(vkCreateShaderModule(device, &sm, nullptr, &shader), "shaderModule");
        VkComputePipelineCreateInfo pi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pi.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pi.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; pi.stage.module=shader; pi.stage.pName="main"; pi.layout=pipeline_layout;
        auto start=Clock::now();
        check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline), "computePipeline");
        pipeline_ms=ms(start);
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pool.queueFamilyIndex=family;
        check(vkCreateCommandPool(device, &pool, nullptr, &command_pool), "commandPool");
    }
};

struct Buffer {
    VkDevice device;
    VkBuffer buffer=VK_NULL_HANDLE;
    VkDeviceMemory memory=VK_NULL_HANDLE;
    void* mapped=nullptr;
    VkDeviceSize bytes;
    bool coherent=false;
    Buffer(Context& ctx, size_t count): device(ctx.device), bytes(count*sizeof(float)) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size=bytes; bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        check(vkCreateBuffer(device, &bi, nullptr, &buffer), "createBuffer");
        VkMemoryRequirements requirements; vkGetBufferMemoryRequirements(device, buffer, &requirements);
        VkPhysicalDeviceMemoryProperties memories; vkGetPhysicalDeviceMemoryProperties(ctx.physical, &memories);
        int selected=-1;
        for (uint32_t i=0; i<memories.memoryTypeCount; ++i) {
            auto flags=memories.memoryTypes[i].propertyFlags;
            if ((requirements.memoryTypeBits & (1u<<i)) && (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                selected=int(i); if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) break;
            }
        }
        if (selected<0) throw std::runtime_error("no host-visible memory for probe");
        coherent=memories.memoryTypes[selected].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=requirements.size; ai.memoryTypeIndex=selected;
        check(vkAllocateMemory(device, &ai, nullptr, &memory), "allocateMemory");
        check(vkBindBufferMemory(device, buffer, memory, 0), "bindMemory");
        check(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped), "mapMemory");
    }
    ~Buffer() {
        if (mapped) vkUnmapMemory(device, memory);
        if (buffer) vkDestroyBuffer(device, buffer, nullptr);
        if (memory) vkFreeMemory(device, memory, nullptr);
    }
    void upload(const std::vector<float>& values) {
        std::memcpy(mapped, values.data(), bytes);
        if (!coherent) {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE}; range.memory=memory; range.size=VK_WHOLE_SIZE;
            check(vkFlushMappedMemoryRanges(device, 1, &range), "flushMemory");
        }
    }
    void invalidate() {
        if (!coherent) {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE}; range.memory=memory; range.size=VK_WHOLE_SIZE;
            check(vkInvalidateMappedMemoryRanges(device, 1, &range), "invalidateMemory");
        }
    }
};

struct RunResources {
    Context& ctx;
    VkDescriptorPool pool=VK_NULL_HANDLE;
    VkQueryPool queries=VK_NULL_HANDLE;
    VkFence fence=VK_NULL_HANDLE;
    VkCommandBuffer command=VK_NULL_HANDLE;
    ~RunResources() {
        if (command) vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &command);
        if (fence) vkDestroyFence(ctx.device, fence, nullptr);
        if (queries) vkDestroyQueryPool(ctx.device, queries, nullptr);
        if (pool) vkDestroyDescriptorPool(ctx.device, pool, nullptr);
    }
};

double percentile(std::vector<double> values, double quantile) {
    std::sort(values.begin(), values.end());
    if (quantile==0.5) return (values[(values.size()-1)/2]+values[values.size()/2])/2;
    return values[size_t(std::ceil(quantile*values.size()))-1];
}
float activate(float value, uint32_t activation) {
    return activation==1 ? std::max(0.f, value) : 0.5f*value*(1+std::tanh(0.7978845608f*(value+0.044715f*value*value*value)));
}

std::string benchmark(Context& ctx, uint32_t m, uint32_t d, uint32_t h,
                      uint32_t activation, int warmup, int iterations) {
    std::vector<float> input(size_t(m)*d), w1(size_t(h)*d), w2(size_t(d)*h);
    std::mt19937 rng(42); std::uniform_real_distribution<float> random(-0.03f, 0.03f);
    for (auto* values : {&input, &w1, &w2}) for (auto& value : *values) value=random(rng);
    Buffer a(ctx, input.size()), b(ctx, w1.size()), hidden(ctx, size_t(m)*h), b2(ctx, w2.size()), output(ctx, size_t(m)*d);
    a.upload(input); b.upload(w1); b2.upload(w2);
    RunResources resources{ctx};
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=2; pool.poolSizeCount=1; pool.pPoolSizes=&size;
    check(vkCreateDescriptorPool(ctx.device, &pool, nullptr, &resources.pool), "descriptorPool");
    VkDescriptorSetLayout layouts[2]={ctx.set_layout, ctx.set_layout};
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool=resources.pool; allocate.descriptorSetCount=2; allocate.pSetLayouts=layouts;
    VkDescriptorSet sets[2]; check(vkAllocateDescriptorSets(ctx.device, &allocate, sets), "descriptorSets");
    Buffer* buffers[6]={&a, &b, &hidden, &hidden, &b2, &output};
    VkDescriptorBufferInfo buffer_info[6]{}; VkWriteDescriptorSet writes[6]{};
    for (int i=0; i<6; ++i) {
        buffer_info[i]={buffers[i]->buffer, 0, buffers[i]->bytes};
        writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet=sets[i/3]; writes[i].dstBinding=i%3;
        writes[i].descriptorCount=1; writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo=&buffer_info[i];
    }
    vkUpdateDescriptorSets(ctx.device, 6, writes, 0, nullptr);
    if (ctx.timestamp_bits) {
        VkQueryPoolCreateInfo query{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; query.queryType=VK_QUERY_TYPE_TIMESTAMP; query.queryCount=2;
        check(vkCreateQueryPool(ctx.device, &query, nullptr, &resources.queries), "queryPool");
    }
    VkCommandBufferAllocateInfo cmd{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmd.commandPool=ctx.command_pool; cmd.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cmd.commandBufferCount=1;
    check(vkAllocateCommandBuffers(ctx.device, &cmd, &resources.command), "commandBuffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    check(vkBeginCommandBuffer(resources.command, &begin), "beginCommand");
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; barrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(resources.command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    if (resources.queries) {
        vkCmdResetQueryPool(resources.command, resources.queries, 0, 2);
        vkCmdWriteTimestamp(resources.command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, resources.queries, 0);
    }
    vkCmdBindPipeline(resources.command, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipeline);
    uint32_t shape[4]={m, h, d, activation};
    vkCmdBindDescriptorSets(resources.command, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipeline_layout, 0, 1, &sets[0], 0, nullptr);
    vkCmdPushConstants(resources.command, ctx.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, shape);
    vkCmdDispatch(resources.command, (h+15)/16, (m+15)/16, 1);
    barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(resources.command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    shape[1]=d; shape[2]=h; shape[3]=0;
    vkCmdBindDescriptorSets(resources.command, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipeline_layout, 0, 1, &sets[1], 0, nullptr);
    vkCmdPushConstants(resources.command, ctx.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, shape);
    vkCmdDispatch(resources.command, (d+15)/16, (m+15)/16, 1);
    barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(resources.command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    if (resources.queries) vkCmdWriteTimestamp(resources.command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, resources.queries, 1);
    check(vkEndCommandBuffer(resources.command), "endCommand");
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; check(vkCreateFence(ctx.device, &fi, nullptr, &resources.fence), "fence");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&resources.command;
    std::vector<double> wall, gpu;
    for (int i=0; i<warmup+iterations; ++i) {
        check(vkResetFences(ctx.device, 1, &resources.fence), "resetFence");
        auto start=Clock::now();
        check(vkQueueSubmit(ctx.queue, 1, &submit, resources.fence), "queueSubmit");
        check(vkWaitForFences(ctx.device, 1, &resources.fence, VK_TRUE, 30'000'000'000ULL), "waitFence");
        double elapsed=ms(start);
        if (i>=warmup) wall.push_back(elapsed);
        if (resources.queries) {
            uint64_t stamps[2];
            check(vkGetQueryPoolResults(ctx.device, resources.queries, 0, 2, sizeof(stamps), stamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "timestamps");
            uint64_t mask=ctx.timestamp_bits>=64 ? ~uint64_t(0) : (uint64_t(1)<<ctx.timestamp_bits)-1;
            double elapsed_gpu=((stamps[1]-stamps[0]) & mask)*ctx.properties.limits.timestampPeriod/1e6;
            if (i>=warmup) gpu.push_back(elapsed_gpu);
        }
    }
    output.invalidate();
    auto* actual=static_cast<float*>(output.mapped);
    for (size_t i=0; i<output.bytes/sizeof(float); ++i)
        if (!std::isfinite(actual[i])) throw std::runtime_error("non-finite GPU output");
    double max_error=0, scaled_error=0;
    for (uint32_t token : {0u, m-1}) {
        std::vector<float> expected_hidden(h);
        for (uint32_t row=0; row<h; ++row) {
            float value=0;
            for (uint32_t k=0; k<d; ++k) value+=input[token*d+k]*w1[row*d+k];
            expected_hidden[row]=activate(value, activation);
        }
        for (uint32_t col : {0u, d/2, d-1}) {
            float expected=0;
            for (uint32_t k=0; k<h; ++k) expected+=expected_hidden[k]*w2[col*h+k];
            double error=std::abs(expected-actual[token*d+col]);
            max_error=std::max(max_error, error);
            scaled_error=std::max(scaled_error, error/(1e-5+0.01*std::abs(expected)));
        }
    }
    std::ostringstream out;
    out << "{\"workload\":" << quote(activation==1 ? "fc_relu_fc" : "fc_gelu_approx_fc")
        << ",\"tokens\":" << m << ",\"width\":" << d << ",\"hidden\":" << h
        << ",\"status\":" << quote(scaled_error<=1 ? "success" : "numerical_check_failed")
        << ",\"precision\":\"fp32\",\"p50_ms\":" << percentile(wall, .5) << ",\"p95_ms\":" << percentile(wall, .95)
        << ",\"mean_ms\":" << std::accumulate(wall.begin(), wall.end(), 0.0)/wall.size()
        << ",\"gpu_p50_ms\":" << (gpu.empty() ? "null" : std::to_string(percentile(gpu, .5)))
        << ",\"warmup\":" << warmup << ",\"iterations\":" << iterations
        << ",\"constant_weight_bytes\":" << (w1.size()+w2.size())*sizeof(float)
        << ",\"buffer_bytes\":" << a.bytes+b.bytes+hidden.bytes+b2.bytes+output.bytes
        << ",\"max_abs_error_sampled\":" << max_error << ",\"max_scaled_error_sampled\":" << scaled_error << "}";
    return out.str();
}

int main(int argc, char** argv) {
    try {
        int iterations=20, warmup=3; std::string shader;
        for (int i=1; i<argc; ++i) {
            std::string arg=argv[i];
            if (i+1>=argc) throw std::runtime_error("incomplete argument");
            if (arg=="--shader") shader=argv[++i];
            else if (arg=="--iterations") iterations=std::stoi(argv[++i]);
            else if (arg=="--warmup") warmup=std::stoi(argv[++i]);
            else throw std::runtime_error("unknown argument: "+arg);
        }
        if (shader.empty() || iterations<1 || iterations>100 || warmup<0 || warmup>20) throw std::runtime_error("invalid arguments");
        Context ctx; ctx.initialize(shader);
        std::cout << "{\"schema_version\":1,\"api\":\"Vulkan_compute\",\"cpu_included\":false,"
                  << "\"scope\":\"custom tiled FP32 two-layer FFN probes; no attention or full video model\",\"devices\":[{"
                  << "\"name\":" << quote(ctx.properties.deviceName) << ",\"hardware_driver\":true,\"type\":" << ctx.properties.deviceType
                  << ",\"vendor_id\":" << ctx.properties.vendorID << ",\"driver_version\":" << ctx.properties.driverVersion
                  << ",\"vulkan_api_version\":" << ctx.properties.apiVersion
                  << ",\"pipeline_compile_ms\":" << ctx.pipeline_ms << ",\"results\":[";
        for (int i=0; i<3; ++i) {
            if (i) std::cout << ',';
            try { std::cout << benchmark(ctx, i==2 ? 17 : 128, i==2 ? 33 : 512, i==2 ? 65 : 2048, i==1 ? 2 : 1, warmup, iterations); }
            catch (const std::exception& error) {
                std::cout << "{\"status\":\"failed\",\"case_index\":" << i << ",\"error\":" << quote(error.what()) << '}';
            }
        }
        rusage usage{}; getrusage(RUSAGE_SELF, &usage);
        std::cout << "]}],\"process_peak_rss_kib\":" << usage.ru_maxrss << "}\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

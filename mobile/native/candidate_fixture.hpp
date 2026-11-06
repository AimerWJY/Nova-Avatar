#pragma once
// Bounded little-endian FP32 weight/reference transport, not a generic ncnn model format.
struct Fixture {
    int frames=0,reference_frames=0;
    ncnn::Mat image,history,spatial,context;
    std::vector<ncnn::Mat> rgb,state;
};
uint32_t read_u32(std::ifstream& stream) {
    unsigned char b[4];
    if (!stream.read(reinterpret_cast<char*>(b),4)) throw std::runtime_error("truncated bundle integer");
    return uint32_t(b[0]) | uint32_t(b[1])<<8 | uint32_t(b[2])<<16 | uint32_t(b[3])<<24;
}
ncnn::Mat read_tensor(std::ifstream& stream,int count) {
    ncnn::Mat value(count);
    if (!stream.read(reinterpret_cast<char*>((float*)value),size_t(count)*sizeof(float)))
        throw std::runtime_error("truncated bundle tensor");
    for (int i=0;i<count;++i) if (!std::isfinite(value[i])) throw std::runtime_error("non-finite bundle tensor");
    return value;
}
Fixture load_fixture(const std::string& filename,std::vector<Definition>& definitions) {
    static_assert(__BYTE_ORDER__==__ORDER_LITTLE_ENDIAN__,"fixture needs a little-endian target");
    std::ifstream stream(filename,std::ios::binary|std::ios::ate);
    if (!stream || stream.tellg()>32*1024*1024) throw std::runtime_error("missing/oversized candidate bundle");
    stream.seekg(0); char magic[8];
    if (!stream.read(magic,8) || std::string(magic,8)!="NOVAAV01" || read_u32(stream)!=1)
        throw std::runtime_error("candidate bundle schema mismatch");
    if (read_u32(stream)!=definitions.size()) throw std::runtime_error("candidate bundle node count mismatch");
    for (auto& node:definitions) {
        if (read_u32(stream)!=node.weights.size()) throw std::runtime_error("candidate bundle weight slot mismatch");
        for (auto& value:node.weights) {
            uint32_t count=read_u32(stream);
            if (count!=value.total()) throw std::runtime_error("candidate bundle weight shape mismatch");
            value=read_tensor(stream,int(count));
        }
    }
    Fixture result;
    result.frames=int(read_u32(stream)); result.reference_frames=int(read_u32(stream));
    if (result.frames<8 || result.frames>160 || result.reference_frames!=8)
        throw std::runtime_error("candidate bundle frame count mismatch");
    result.image=read_tensor(stream,SIDE*SIDE*3).reshape(SIDE,SIDE,3);
    result.history=read_tensor(stream,(result.frames+HISTORY-1)*MEL);
    result.spatial=read_tensor(stream,(SIDE/4)*(SIDE/4)*32).reshape(SIDE/4,SIDE/4,32);
    result.context=read_tensor(stream,WIDTH);
    for (int i=0;i<result.reference_frames;++i) {
        result.rgb.push_back(read_tensor(stream,SIDE*SIDE*3).reshape(SIDE,SIDE,3));
        result.state.push_back(read_tensor(stream,WIDTH));
    }
    if (stream.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("unexpected trailing candidate bundle data");
    return result;
}

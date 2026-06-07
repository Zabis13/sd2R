// Standalone C++ test for the meta backend split logic (sd2r_meta_backend.hpp).
// Checks compute_split_ne, the decide_split policy, and the C callback — no GPU.
// Mirrors the R PoC inst/examples/meta_backend_poc_encoder.R at the C++ level.
//
// Compile (GR = path to ggmlR):
//   g++ -std=c++17 -I<sd2R>/src/sd -I<GR>/src \
//       inst/examples/test_meta_backend.cpp -o /tmp/test_meta && /tmp/test_meta
#include <cstdio>
#include <cmath>
#include <vector>
#include "sd2r_meta_backend.hpp"

static ggml_tensor mk(int64_t ne0,int64_t ne1,int64_t ne2,int64_t ne3){
    ggml_tensor t; std::memset(&t,0,sizeof(t));
    t.ne[0]=ne0;t.ne[1]=ne1;t.ne[2]=ne2;t.ne[3]=ne3; return t;
}

int main(){
    using namespace sd2r::meta;
    int fails=0;

    // 1. compute_split_ne — same cases as the R PoC
    struct C{int64_t d; size_t n; }; 
    C cases[]={{64,2},{64,8},{100,8},{7,3},{16,100}};
    for(auto c: cases){
        auto per=compute_split_ne(c.d,c.n);
        int64_t s=0; for(auto v:per) s+=v;
        bool ok = (s==c.d) && (per.size()==c.n);
        printf("compute_split_ne d=%lld n=%zu sum=%lld %s\n",
               (long long)c.d,c.n,(long long)s, ok?"OK":"FAIL");
        if(!ok) fails++;
    }

    // 2. decide_split policy.
    // Default policy: split disabled -> everything MIRRORED (safe default).
    Policy def;
    ggml_tensor big0=mk(512,1024,1,1);
    auto sdef=decide_split(&big0,4,def);
    printf("default policy big axis=%d (want %d MIRRORED)\n",
           (int)sdef.axis, GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    if(sdef.axis!=GGML_BACKEND_SPLIT_AXIS_MIRRORED) fails++;

    // Split policy enabled: validate the axis-split math.
    Policy pol; pol.tensor_split_enabled=true;
    ggml_tensor big=mk(512,1024,1,1);  // large Linear -> split axis 1
    ggml_tensor small=mk(8,16,1,1);    // small -> MIRRORED
    ggml_tensor bias=mk(1024,1,1,1);   // 1D -> MIRRORED (ne1==1 < split_min)
    auto sb=decide_split(&big,4,pol);
    auto ss=decide_split(&small,4,pol);
    auto sbias=decide_split(&bias,4,pol);
    printf("big axis=%d (want %d)  small axis=%d  bias axis=%d (want %d)\n",
           (int)sb.axis, GGML_BACKEND_SPLIT_AXIS_1, (int)ss.axis,
           (int)sbias.axis, GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    if(sb.axis!=GGML_BACKEND_SPLIT_AXIS_1) fails++;
    if(ss.axis!=GGML_BACKEND_SPLIT_AXIS_MIRRORED) fails++;
    if(sbias.axis!=GGML_BACKEND_SPLIT_AXIS_MIRRORED) fails++;
    // segment ne for big must sum to out_dim=1024
    { int64_t s=0; for(size_t d=0;d<4;d++) s+=sb.ne[d];
      printf("big split ne sum=%lld (want 1024) %s\n",(long long)s, s==1024?"OK":"FAIL");
      if(s!=1024) fails++; }

    // 3. callback round-trip
    SplitContext sc; sc.policy=pol; sc.n_devs=4;
    auto cb=get_split_state_cb(&big,&sc);
    if(cb.axis!=GGML_BACKEND_SPLIT_AXIS_1) fails++;
    auto cbnull=get_split_state_cb(nullptr,nullptr);
    if(cbnull.axis!=GGML_BACKEND_SPLIT_AXIS_MIRRORED) fails++;

    printf("%s\n", fails==0?"RESULT: PASS":"RESULT: FAIL");
    return fails==0?0:1;
}

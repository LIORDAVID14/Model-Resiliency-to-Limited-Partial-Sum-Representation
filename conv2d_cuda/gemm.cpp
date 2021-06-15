#include <torch/extension.h>
#include <vector>

// CUDA forward declerations
std::vector<torch::Tensor> cu_gemm(
    torch::Tensor a,
    torch::Tensor b,
    int bits_allowed,
    int bit_to_all);

// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> cu_gemm_c(
    torch::Tensor a,
    torch::Tensor b,
    int bits_allowed,
    int bits_to_all){

    CHECK_INPUT(a);
    CHECK_INPUT(b);

    return cu_gemm(a, b, bits_allowed, bits_to_all);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cu_gemm_c, "Run!");
}
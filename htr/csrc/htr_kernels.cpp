#include <torch/extension.h>
torch::Tensor gemv_partials(torch::Tensor qw, torch::Tensor xt, int64_t nsplit);
torch::Tensor dequant_f16(torch::Tensor qw, torch::Tensor scales, double cb);
torch::Tensor decode_raw(torch::Tensor qw);
std::vector<torch::Tensor> prologue(torch::Tensor x, torch::Tensor su);
torch::Tensor epilogue(torch::Tensor part, torch::Tensor xs, torch::Tensor tok, torch::Tensor scales, double cb, torch::Tensor sv,
                       c10::optional<torch::Tensor> bias, int64_t mode, int64_t out_dtype);
torch::Tensor gemv_fused(torch::Tensor qw, torch::Tensor x, torch::Tensor su, torch::Tensor sv, torch::Tensor scales, double cb,
                         c10::optional<torch::Tensor> bias, torch::Tensor counters, int64_t nsplit);
torch::Tensor gemv_epi(torch::Tensor qw, torch::Tensor xt, torch::Tensor xs, torch::Tensor tok, torch::Tensor scales, double cb,
                       torch::Tensor sv, c10::optional<torch::Tensor> bias, torch::Tensor counters, int64_t nsplit,
                       int64_t out_dtype, int64_t wg_reduce);
torch::Tensor dequant_w(torch::Tensor qw, torch::Tensor scales, double cb, torch::Tensor su, torch::Tensor sv);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv_partials", &gemv_partials); m.def("dequant_f16", &dequant_f16); m.def("decode_raw", &decode_raw);
  m.def("prologue", &prologue); m.def("gemv_fused", &gemv_fused); m.def("epilogue", &epilogue);
  m.def("dequant_w", &dequant_w); m.def("gemv_epi", &gemv_epi);
}

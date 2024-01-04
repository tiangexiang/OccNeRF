#include <stdint.h>
#include <torch/torch.h>
#include <torch/script.h>

torch::Tensor msknn(torch::Tensor x, torch::Tensor y,
                  torch::Tensor index,
                  torch::optional<torch::Tensor> x_index,
                  torch::optional<torch::Tensor> ptr_x,
                  torch::optional<torch::Tensor> ptr_y, int64_t k, bool cosine,
                  int64_t num_workers);
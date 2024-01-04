#include <torch/extension.h>

#include "msknn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("msknn", &msknn, "msknn (CUDA)");
}
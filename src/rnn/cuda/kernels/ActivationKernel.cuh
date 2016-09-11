#pragma once

#include "../Types.hpp"
#include "../../LayerDef.hpp"
#include <cuda_runtime.h>

namespace rnn {
namespace cuda {
namespace ActivationKernel {

void Apply(const ConnectionActivation &layer, const LayerActivation &activation,
           cudaStream_t stream);

void Apply(const ConnectionActivation &layer, const LayerActivation &activation,
           const ConnectionActivation &copyLayer, cudaStream_t stream);
}
}
}

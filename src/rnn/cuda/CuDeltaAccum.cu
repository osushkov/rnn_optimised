
#include "CuDeltaAccum.hpp"
#include "kernels/MatrixFillKernel.cuh"
#include <cassert>

using namespace rnn;
using namespace rnn::cuda;

CuDeltaAccum::CuDeltaAccum(const RNNSpec &spec, unsigned maxTraceLength) {
  assert(maxTraceLength > 0);

  allDeltaAccum.reserve(maxTraceLength * spec.layers.size());
  accumBuffers.reserve(spec.layers.size());

  for (const auto &layer : spec.layers) {
    CuMatrix buffer = util::AllocMatrix(spec.maxBatchSize * maxTraceLength, layer.numNodes);
    accumBuffers.push_back(buffer);

    for (int timestamp = 0; timestamp < maxTraceLength; timestamp++) {
      allDeltaAccum.emplace_back(layer.uid, timestamp,
                                 CuMatrix::FromBuffer(buffer, spec.maxBatchSize, timestamp));
    }
  }
}

void CuDeltaAccum::Cleanup(void) {
  for (auto &ab : accumBuffers) {
    util::FreeMatrix(ab);
  }
}

CuLayerAccum *CuDeltaAccum::GetDelta(unsigned layerId, int timestamp) {
  for (auto &da : allDeltaAccum) {
    if (da.layerId == layerId && da.timestamp == timestamp) {
      return &da;
    }
  }

  assert(false);
  return nullptr;
}

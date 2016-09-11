
#include "CuLayerMemory.hpp"
#include <cassert>

using namespace rnn;
using namespace rnn::cuda;

CuLayerMemory::CuLayerMemory(const RNNSpec &spec, unsigned maxTraceLength) {
  assert(maxTraceLength > 0);

  outputBuffer = util::AllocMatrix(spec.maxBatchSize * maxTraceLength * 2, spec.numOutputs + 1);

  connectionBuffers.reserve(spec.connections.size());
  for (const auto &conn : spec.connections) {
    unsigned connectionCols = spec.LayerSize(conn.srcLayerId) + 1;

    CuMatrix m = util::AllocMatrix(spec.maxBatchSize * maxTraceLength * 2, connectionCols);
    connectionBuffers.emplace_back(conn, m);
  }

  memory.reserve(maxTraceLength);
  for (int timestamp = 0; timestamp < maxTraceLength; timestamp++) {
    memory.emplace_back(spec, timestamp, connectionBuffers, outputBuffer);
  }
}

void CuLayerMemory::Cleanup(void) {
  util::FreeMatrix(outputBuffer);
  for (auto &cb : connectionBuffers) {
    util::FreeMatrix(cb.second);
  }
}

CuTimeSlice *CuLayerMemory::GetTimeSlice(int timestamp) {
  for (auto &ts : memory) {
    if (ts.timestamp == timestamp) {
      return &ts;
    }
  }

  return nullptr;
}

void CuLayerMemory::Clear(void) {
  for (auto &ts : memory) {
    ts.Clear();
  }
}

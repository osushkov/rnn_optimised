
#include "CuTimeSlice.hpp"
#include "kernels/MatrixFillKernel.cuh"
#include "kernels/InitialiseOutputKernel.cuh"
#include <cassert>

using namespace rnn;
using namespace rnn::cuda;

CuTimeSlice::CuTimeSlice(const RNNSpec &spec, int timestamp,
                         const vector<pair<LayerConnection, CuMatrix>> &matrixBuffers,
                         CuMatrix outputBuffer)
    : timestamp(timestamp),
      networkOutput(LayerConnection(0, 0, 0),
                    CuMatrix::FromBuffer(outputBuffer, spec.maxBatchSize, timestamp),
                    CuMatrix::FromBuffer(outputBuffer, spec.maxBatchSize, timestamp + spec.maxTraceLength)) {

  assert(timestamp >= 0);
  for (const auto &connection : spec.connections) {

    CuMatrix connectionBuf;
    for (const auto& mb : matrixBuffers) {
      if (mb.first == connection) {
        connectionBuf = mb.second;
        break;
      }
    }
    assert(connectionBuf.cols == spec.LayerSize(connection.srcLayerId) + 1);

    connectionData.emplace_back(connection,
        CuMatrix::FromBuffer(connectionBuf, spec.maxBatchSize, timestamp),
        CuMatrix::FromBuffer(connectionBuf, spec.maxBatchSize, timestamp + spec.maxTraceLength));
  }
}

CuConnectionMemoryData *CuTimeSlice::GetConnectionData(const LayerConnection &connection) {
  for (auto &cmd : connectionData) {
    if (cmd.connection == connection) {
      return &cmd;
    }
  }

  assert(false);
  return nullptr;
}

void CuTimeSlice::Clear(void) {
  networkOutput.haveActivation = false;
  MatrixFillKernel::Apply(networkOutput.activation, 0.0f, 0);
  MatrixFillKernel::Apply(networkOutput.derivative, 0.0f, 0);

  for (auto &cd : connectionData) {
    cd.haveActivation = false;
    MatrixFillKernel::Apply(cd.activation, 0.0f, 0);
    MatrixFillKernel::Apply(cd.derivative, 0.0f, 0);
    InitialiseOutputKernel::Apply(cd.activation, 0);
  }
}

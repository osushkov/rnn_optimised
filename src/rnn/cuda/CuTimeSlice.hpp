#pragma once

#include "../LayerDef.hpp"
#include "../RNNSpec.hpp"
#include "Types.hpp"
#include "Util.hpp"
#include <cassert>
#include <utility>
#include <vector>

using namespace std;

namespace rnn {
namespace cuda {

struct CuConnectionMemoryData {
  LayerConnection connection;
  bool haveActivation;

  CuMatrix activation; // batch output, row per batch element.
  CuMatrix derivative;

  CuConnectionMemoryData(const LayerConnection &connection, CuMatrix activation,
                         CuMatrix derivative)
      : connection(connection), haveActivation(false), activation(activation),
        derivative(derivative) {}
};

struct CuTimeSlice {
  int timestamp;
  CuConnectionMemoryData networkOutput;
  vector<CuConnectionMemoryData> connectionData;

  CuTimeSlice(const RNNSpec &spec, int timestamp,
              const vector<pair<LayerConnection, CuMatrix>> &matrixBuffers, CuMatrix outputBuffer);

  CuConnectionMemoryData *GetConnectionData(const LayerConnection &connection);
  void Clear(void);
};
}
}

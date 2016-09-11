#pragma once

#include "../RNNSpec.hpp"
#include "CuTimeSlice.hpp"
#include <vector>

using namespace std;

namespace rnn {
namespace cuda {

class CuLayerMemory {
public:
  CuLayerMemory(const RNNSpec &spec, unsigned maxTraceLength);
  void Cleanup(void);

  CuTimeSlice *GetTimeSlice(int timestamp);
  void Clear(void);

  CuMatrix outputBuffer;
  vector<pair<LayerConnection, CuMatrix>> connectionBuffers;

private:
  vector<CuTimeSlice> memory;
};
}
}

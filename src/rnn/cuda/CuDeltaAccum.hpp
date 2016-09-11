#pragma once

#include "../LayerDef.hpp"
#include "../RNNSpec.hpp"
#include "Types.hpp"
#include "Util.hpp"
#include <cassert>
#include <vector>

using namespace std;

namespace rnn {
namespace cuda {

struct CuLayerAccum {
  unsigned layerId;
  int timestamp;

  unsigned samples;
  CuMatrix accumDelta;

  CuLayerAccum(unsigned layerId, int timestamp, CuMatrix accumDelta)
      : layerId(layerId), timestamp(timestamp), samples(0), accumDelta(accumDelta) {}
};

struct CuDeltaAccum {
  vector<CuLayerAccum> allDeltaAccum;
  vector<CuMatrix> accumBuffers;

  CuDeltaAccum(const RNNSpec &spec, unsigned maxTraceLength);
  void Cleanup(void);

  CuLayerAccum *GetDelta(unsigned layerId, int timestamp);
};
}
}

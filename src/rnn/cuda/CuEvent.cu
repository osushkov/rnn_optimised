
#include "CuEvent.hpp"
#include <cuda_runtime.h>

using namespace rnn;
using namespace rnn::cuda;

struct CuEvent::CuEventImpl {
  cudaEvent_t event;

  CuEventImpl() {
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  }

  void *GetCudaEvent(void) {
    return &event;
  }
};

CuEvent::CuEvent() : impl(new CuEventImpl()) {}
CuEvent::~CuEvent() = default;

void *CuEvent::GetCudaEvent(void) {
  return impl->GetCudaEvent();
}

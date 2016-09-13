#pragma once

#include <memory>

namespace rnn {
namespace cuda {

class CuEvent {
public:
  CuEvent();
  ~CuEvent();

  void *GetCudaEvent(void);

private:
  struct CuEventImpl;
  std::unique_ptr<CuEventImpl> impl;
};
}
}

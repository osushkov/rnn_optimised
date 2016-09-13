#pragma once

#include "CuEvent.hpp"
#include "Task.hpp"
#include <memory>

namespace rnn {
namespace cuda {

class TaskExecutor {
public:
  TaskExecutor();
  ~TaskExecutor();

  void Execute(const Task &task);
  void EventSync(CuEvent *event);
  void EventRecord(CuEvent *event);

private:
  struct TaskExecutorImpl;
  std::unique_ptr<TaskExecutorImpl> impl;
};
}
}

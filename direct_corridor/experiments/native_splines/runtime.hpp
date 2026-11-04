// SPDX-License-Identifier: MIT
#pragma once
#include "../../src/generation_pipeline.hpp"
#include <chrono>
#include <stdexcept>

namespace gcopter::direct::native {
using Clock=std::chrono::steady_clock;
struct DeadlineExceeded:std::runtime_error {
  DeadlineExceeded():std::runtime_error("cubic generation deadline"){}
};
inline double elapsedMs(Clock::time_point start) {
  return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
// Stage profiling is opt-in; normal generation only measures its total time.
class StageTimer {
  bool enabled;
  Clock::time_point start;
public:
  explicit StageTimer(bool profile):enabled(profile) {if(enabled)start=Clock::now();}
  double elapsed()const{return enabled?elapsedMs(start):0;}
  void restart(){if(enabled)start=Clock::now();}
};
// Poll at stage/patch/Newton boundaries. Inner obstacle loops only increment
// a counter, checking the deadline once per 1024 selected obstacles.
class Deadline {
  Clock::time_point end;
  unsigned work=0;
public:
  explicit Deadline(double milliseconds,Clock::time_point start=Clock::now()):end(start+
      std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double,std::milli>(milliseconds))){}
  void check()const {
    if(Clock::now()>end)throw DeadlineExceeded{};
  }
  void obstacle(){if((++work&1023u)==0)check();}
};
}

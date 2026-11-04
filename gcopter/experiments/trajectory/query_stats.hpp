// SPDX-License-Identifier: MIT
#pragma once
#include <chrono>
namespace tube_query {
using Clock=std::chrono::steady_clock;
struct Stats {
  bool enabled=false;
  double objective_ms=0,integral_ms=0,dynamics_ms=0,coefficient_ms=0,tube_ms=0;
};
struct Timer {
  double* destination=nullptr;Clock::time_point start;
  Timer(bool enabled,double& value){if(enabled){destination=&value;start=Clock::now();}}
  ~Timer(){if(destination)*destination+=std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
};
}

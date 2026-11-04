// SPDX-License-Identifier: MIT
#pragma once
#include "query_stats.hpp"
#include "tube_reference.hpp"
#include <SweepCorridor/InverseQuery.hpp>
namespace sweep_benchmark {
class Query {
  SweepCorridor::InverseQuery query_;
public:
  tube_query::Stats stats;
  Query(const SweepCorridor::Corridor& c,bool profile):query_(c){stats.enabled=profile;}
  void reset(){bool enabled=stats.enabled;stats={};stats.enabled=enabled;query_.reset();}
  SweepCorridor::InverseResult operator()(const Vec3& p,double hint){
    tube_query::Timer timer(stats.enabled,stats.tube_ms);return query_(p,hint);
  }
};
}

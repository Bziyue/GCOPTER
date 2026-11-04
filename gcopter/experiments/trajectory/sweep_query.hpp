// SPDX-License-Identifier: MIT
#pragma once
#include "tube_query.hpp"
#include <SweepCorridor/InverseQuery.hpp>
#include <variant>
namespace sweep_benchmark {
namespace sc=SweepCorridor;
class Query {
  std::variant<sc::InverseQuery<3>,sc::InverseQuery<5>> query_;
public:
  tube_query::Stats stats;
  template<int D> Query(const sc::Corridor<D>& c,bool profile):query_(std::in_place_type<sc::InverseQuery<D>>,c){stats.enabled=profile;}
  void reset(){bool enabled=stats.enabled;stats={};stats.enabled=enabled;std::visit([](auto& q){q.reset();},query_);}
  sc::InverseResult operator()(const Vec3& p,double hint){
    tube_query::Timer timer(stats.enabled,stats.tube_ms);
    return std::visit([&](auto& q){return q(p,hint);},query_);
  }
};
}

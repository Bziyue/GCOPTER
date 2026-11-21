// SPDX-License-Identifier: MIT
#pragma once
#include "model.hpp"
#include <map>
#include <string>

namespace SweepCorridor::experiments::h3 {
enum class MapInput { Points, Voxels, Spheres };
struct BenchmarkPreset { OptimizerSettings solver; MapInput input; };

inline BenchmarkPreset benchmarkPreset(const std::string& name,const c3::OptimizerSettings& baseline) {
  OptimizerSettings current(baseline);
  if(name=="h3_points")return {current,MapInput::Points};
  if(name=="h3_new")return {current,MapInput::Voxels};
  if(name=="h3_spheres")return {current,MapInput::Spheres};
  if(name=="h3_core_full"){current.cap_volume_weight=1;return {current,MapInput::Voxels};}
  if(name=="h3_core_pressure"){current.use_section_gain=false;return {current,MapInput::Voxels};}
  if(name=="h3_core_two_rounds"){current.refresh_rounds=2;return {current,MapInput::Voxels};}
  if(name=="h3_core_span_scale"){current.separator_scale=SeparatorScale::Span;return {current,MapInput::Voxels};}
  if(name=="h3_core_no_share"){current.share_cover_planes=false;return {current,MapInput::Voxels};}

  // Frozen presets inherit only from an explicitly named historical experiment.
  // Keep their parameters independent of today's demo settings.
  OptimizerSettings old(baseline);
  old.iterations=8;old.corrector_iterations=3;old.bending_weight=1;old.initial_radius_fraction=.65;
  old.fine_iterations=4;old.refresh_rounds=2;old.refresh_fraction=.05;old.joint_budget_share=.99;old.group_axis_limit=INFINITY;
  old.cap_volume_weight=1;old.relative_jacobian_margin=.075;old.pointwise_jacobian=false;
  old.min_center_progress=0;old.intermediate_radius=false;old.smoothing_volume_tolerance=0;
  old.use_section_gain=false;old.require_refresh_for_ends=true;
  old.separator_scale=SeparatorScale::GroupOnlyDuringInflation;
  if(name=="h3_pre_core" || name=="h3_pre_core_spheres")return {old,name=="h3_pre_core"?MapInput::Voxels:MapInput::Spheres};
  if(name=="h3_voxel_single"){old.cover_group_size=1;return {old,MapInput::Voxels};}
  if(name=="h3_voxel_fine"){old.coarse_start=false;return {old,MapInput::Voxels};}
  if(name=="h3_voxel_positive"){old.relative_jacobian_margin=1e-9;return {old,MapInput::Voxels};}
  auto jets=old;
  jets.cover_group_size=1;jets.coarse_start=false;jets.relative_jacobian_margin=1e-9;
  jets.refresh_rounds=0;jets.refresh_fraction=0;jets.end_steps=0;jets.joint_budget_share=1;
  jets.volume_jet_energy=1;jets.joint_jet_steps=0;jets.joint_sweeps=1;jets.joint_final_only=false;
  std::map<std::string,OptimizerSettings> presets{{"h3_jets",jets}};
  const auto add=[&](const char* key,const char* parent,auto edit) {
    auto settings=presets.at(parent);
    edit(settings);
    presets.emplace(key,settings);
  };
  add("h3_joint","h3_jets",[](auto& s){s.joint_jet_steps=2;});
  add("h3_final","h3_joint",[](auto& s){s.joint_final_only=true;});
  add("h3_two","h3_final",[](auto& s){s.joint_sweeps=2;});
  add("h3_two_reg3","h3_two",[](auto& s){s.volume_jet_energy=3;});
  add("h3_new_end4","h3_two_reg3",[](auto& s){s.end_steps=4;});
  add("h3_new_local5","h3_new_end4",[](auto& s){s.refresh_rounds=1;s.refresh_fraction=0.05;});
  add("h3_new_share99","h3_new_local5",[](auto& s){s.refresh_rounds=2;s.joint_budget_share=0.99;});
  add("h3_balanced","h3_final",[](auto& s){s.volume_jet_energy=3;s.bending_weight=0.5;});
  add("h3","h3_jets",[](auto& s){s.jet_schedule=JetSchedule::Frozen;});
  add("h3_adaptive_no_budget","h3_new_local5",[](auto& s){s.refresh_rounds=2;});
  add("h3_adaptive_no_end","h3_new_share99",[](auto& s){s.end_steps=0;});
  add("h3_adaptive_no_interior","h3_new_share99",[](auto& s){s.refresh_fraction=0;});
  add("h3_adaptive_no_refresh","h3_new_share99",[](auto& s){s.refresh_rounds=0;});
  add("h3_adaptive_one_round","h3_new_local5",[](auto& s){s.joint_budget_share=0.99;});
  add("h3_balanced2","h3_balanced",[](auto& s){s.corrector_iterations=2;});
  add("h3_balanced6","h3_balanced",[](auto& s){s.iterations=6;});
  add("h3_final4","h3_final",[](auto& s){s.joint_jet_steps=4;});
  add("h3_final_reg3","h3_final",[](auto& s){s.volume_jet_energy=3;});
  add("h3_four","h3_final",[](auto& s){s.joint_jet_steps=1;s.joint_sweeps=4;});
  add("h3_half","h3_jets",[](auto& s){s.bending_weight=0.5;});
  add("h3_halving","h3_final",[](auto& s){s.joint_fast_line=false;});
  add("h3_joint4","h3_jets",[](auto& s){s.joint_jet_steps=4;});
  add("h3_jointreg","h3_joint",[](auto& s){s.volume_jet_energy=10;});
  add("h3_mid","h3_jets",[](auto& s){s.jet_schedule=JetSchedule::BeforeTightening;});
  add("h3_more","h3_jets",[](auto& s){s.iterations=12;});
  add("h3_new_adaptive10","h3_new_local5",[](auto& s){s.joint_budget_fraction=0.5;s.refresh_fraction=0.1;});
  add("h3_new_adaptive2","h3_adaptive_no_budget",[](auto& s){s.joint_budget_fraction=0.5;});
  add("h3_new_adaptive5","h3_new_local5",[](auto& s){s.joint_budget_fraction=0.5;});
  add("h3_new_budget50","h3_two_reg3",[](auto& s){s.joint_budget_fraction=0.5;});
  add("h3_new_budget75","h3_two_reg3",[](auto& s){s.joint_budget_fraction=0.75;});
  add("h3_new_end6","h3_new_end4",[](auto& s){s.end_nodes=6;});
  add("h3_new_end8","h3_new_end4",[](auto& s){s.end_nodes=8;});
  add("h3_new_end_soft","h3_new_end4",[](auto& s){s.end_energy=0.1;});
  add("h3_new_faces","h3_new_end4",[](auto& s){s.refresh_rounds=1;});
  add("h3_new_faces2","h3_new_end4",[](auto& s){s.refresh_rounds=2;});
  add("h3_new_faces_center","h3_new_faces",[](auto& s){s.refresh_shape=false;});
  add("h3_new_faces_soft","h3_new_end_soft",[](auto& s){s.refresh_rounds=1;});
  add("h3_new_halving","h3_two_reg3",[](auto& s){s.joint_fast_line=false;});
  add("h3_new_local10","h3_new_local5",[](auto& s){s.refresh_fraction=0.1;});
  add("h3_new_rebuild","h3_two_reg3",[](auto& s){s.reuse_joint_supports=false;});
  add("h3_new_share995","h3_new_share99",[](auto& s){s.joint_budget_share=0.995;});
  add("h3_new_share_only","h3_two_reg3",[](auto& s){s.joint_budget_share=0.99;});
  add("h3_polish","h3_jets",[](auto& s){s.polish=true;});
  presets.emplace("h3_previous",presets.at("h3_jets"));
  add("h3_rebuild","h3_balanced2",[](auto& s){s.reuse_joint_supports=false;});
  add("h3_relaxed","h3_jets",[](auto& s){s.bending_weight=0.25;});
  add("h3_relaxed1","h3_jets",[](auto& s){s.bending_weight=0.1;});
  add("h3_relaxed_joint","h3_final",[](auto& s){s.bending_weight=0.25;});
  add("h3_three","h3_jets",[](auto& s){s.iterations=4;s.corrector_iterations=2;s.inflation_iterations=3;});
  add("h3_two_half","h3_two",[](auto& s){s.bending_weight=0.5;});
  add("h3_two_soft","h3_two_reg3",[](auto& s){s.bending_weight=0.5;});
  add("h3_vmid","h3_mid",[](auto& s){s.volume_jet_steps=2;});
  add("h3_volume","h3_jets",[](auto& s){s.volume_jet_steps=2;});
  add("h3_vreg","h3_volume",[](auto& s){s.volume_jet_energy=10;});
  presets.emplace("h3_world",presets.at("h3_jets"));
  const auto it=presets.find(name);
  if(it==presets.end())throw std::invalid_argument("unknown H3 benchmark mode: "+name);
  return {it->second,MapInput::Spheres};
}
}

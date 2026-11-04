// SPDX-License-Identifier: MIT
#pragma once
#include <SweepCorridor.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
namespace sc=SweepCorridor;
namespace c3=SweepCorridor::internal::cubic;
namespace fs=std::filesystem;
inline std::vector<sc::Vec3> readPoints(const fs::path& path) {
  std::ifstream file(path);if(!file)throw std::runtime_error("missing input: "+path.string());
  std::vector<sc::Vec3> out;std::string line;
  while(std::getline(file,line)){if(line.empty()||line[0]=='#')continue;std::istringstream row(line);sc::Vec3 p;
    if(!(row>>p[0]>>p[1]>>p[2])||!p.allFinite())throw std::runtime_error("invalid point");out.push_back(p);}return out;
}
inline double seedLength(const std::vector<sc::Vec3>& seed) {double l=0;for(size_t i=1;i<seed.size();++i)l+=(seed[i]-seed[i-1]).norm();return l;}
inline void writeGeometry(const c3::Geometry& g,const fs::path& path) {
  std::ofstream out(path);out<<std::setprecision(17)<<"degree,u0,u1,cx,cy,cz,m00,m10,m20,m01,m11,m21\n";
  for(size_t i=0;i<g.controls.size();++i){out<<3<<','<<g.breaks[i/4]<<','<<g.breaks[i/4+1];for(double x:g.controls[i])out<<','<<x;out<<'\n';}
}
inline c3::Geometry readGeometry(const fs::path& path) {
  std::ifstream in(path);if(!in)throw std::runtime_error("missing geometry");std::string line;std::getline(in,line);c3::Geometry g;
  while(std::getline(in,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);int degree;double a,b;c3::V9 p;
    row>>degree>>a>>b;for(double& x:p)row>>x;if(!row||degree!=3||!(b>a)||!p.allFinite())throw std::runtime_error("invalid geometry");
    if(g.controls.size()%4==0){if(g.controls.empty())g.breaks.push_back(a);g.breaks.push_back(b);}g.controls.push_back(p);}
  if(g.controls.empty()||g.controls.size()%4)throw std::runtime_error("incomplete geometry");return g;
}
inline c3::Geometry restrictGeometry(const c3::Geometry& g,double a,double b) {
  c3::Geometry out;out.breaks.push_back(0);
  for(int i=0;i<int(g.pieces());++i){const double begin=std::max(a,g.breaks[i]),end=std::min(b,g.breaks[i+1]);if(!(end>begin))continue;
    const double h=g.breaks[i+1]-g.breaks[i];const auto p=c3::restrictCurve<3>(g.curve(i),(begin-g.breaks[i])/h,(end-g.breaks[i])/h);
    out.controls.insert(out.controls.end(),p.begin(),p.end());out.breaks.push_back((end-a)/(b-a));}return out;
}
inline sc::Options optionsFor(const std::vector<sc::Vec3>& seed,const sc::Bounds& bounds) {
  sc::Options o;o.budget_ms=10000;double diagonal=0;sc::Vec3 a,b=seed.front();
  for(size_t i=1;i<seed.size();){a=b;if((a-seed[i]).norm()>7)b=a+(seed[i]-a).normalized()*7;else b=seed[i++];
    const sc::Vec3 lo=(a.cwiseMin(b)-sc::Vec3::Constant(3)).cwiseMax(bounds.lower+sc::Vec3::Constant(.252));
    const sc::Vec3 hi=(a.cwiseMax(b)+sc::Vec3::Constant(3)).cwiseMin(bounds.upper-sc::Vec3::Constant(.252));diagonal=std::max(diagonal,(hi-lo).norm());}
  o.max_radius=.5*diagonal+.001;o.max_offset=diagonal+.001;return o;
}
struct Quality {double swept_volume=0,boundary_bending=0,max_curvature=0,c2_error=0,min_axis=INFINITY;};
inline Quality quality(const c3::Geometry& g,double length) {
  Quality q;constexpr double points[]={.046910077030668,.230765344947158,.5,.769234655052842,.953089922969332};
  constexpr double weights[]={.118463442528095,.239314335249683,.284444444444444,.239314335249683,.118463442528095};
  for(int i=0;i<int(g.pieces());++i){const auto p=g.curve(i);const double h=g.breaks[i+1]-g.breaks[i];
    for(int k=0;k<5;++k){const auto s=c3::evaluate<3>(p,points[k],h);
      q.swept_volume+=std::acos(-1.)*h*weights[k]*s.shape.col(0).cross(s.shape.col(1)).dot(s.first);
      q.boundary_bending+=h*weights[k]*(s.second.squaredNorm()+.5*s.shape_second.squaredNorm())/(length*length*length);
      q.max_curvature=std::max(q.max_curvature,s.first.cross(s.second).norm()/std::pow(s.first.norm(),3));
      q.min_axis=std::min(q.min_axis,std::sqrt((s.shape.transpose()*s.shape).eigenvalues().real().minCoeff()));}
    if(i){const auto left=c3::evaluate<3>(g.curve(i-1),1,g.breaks[i]-g.breaks[i-1]),right=c3::evaluate<3>(p,0,h);
      q.c2_error=std::max({q.c2_error,(left.center-right.center).norm(),(left.shape-right.shape).norm(),
        (left.first-right.first).norm()/length,(left.shape_first-right.shape_first).norm()/length,
        (left.second-right.second).norm()/(length*length),(left.shape_second-right.shape_second).norm()/(length*length)});}}
  return q;
}

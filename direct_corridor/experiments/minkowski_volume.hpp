// SPDX-License-Identifier: MIT
#pragma once
#include "minkowski.hpp"
#include "../../gcopter/tools/corridor_volume.hpp"
namespace minkowski_volume {
using namespace gcopter::direct;
namespace mk=gcopter::direct::minkowski;
namespace cv=corridor_volume;
struct ShapePatch {
  cv::Patch patch;
  cv::Box box;
  bool valid=true;
  std::size_t* unresolved;
  bool contains(const Vec3& p) const {return patch.contains(p,*unresolved);}
};
inline std::vector<ShapePatch> patches(const mk::Corridor& c,std::size_t& unresolved) {
  std::vector<ShapePatch> out(c.pieces());
  for(std::size_t i=0;i<out.size();++i) {
    auto& p=out[i].patch;
    std::array<mk::Weights,6> weights;
    for(int j=0;j<6;++j) {
      p.c[j]=c.centerControls()[6*i+j]; p.m[j]=c.frameControls()[6*i+j];
      weights[j]=c.weightControls()[6*i+j];
    }
    p.prepare(); p.box=cv::Box();
    Vec3 extent=Vec3::Zero();
    for(int axis=0;axis<3;++axis) {
      mk::Weights h=mk::Weights::Zero();
      for(int k=0;k<c.templates();++k) {
        const Eigen::Vector2d a=p.m.front().row(axis),b=p.m.back().row(axis);
        h[k]=std::sqrt(std::max(a.dot(c.templateMatrices()[k]*a),b.dot(c.templateMatrices()[k]*b)));
      }
      for(const auto& r:weights) extent[axis]=std::max(extent[axis],r.dot(h));
    }
    for(const auto& center:p.c) {
      p.box.lo=p.box.lo.cwiseMin(center-extent); p.box.hi=p.box.hi.cwiseMax(center+extent);
    }
    p.box.lo.array()-=1e-9; p.box.hi.array()+=1e-9;
    p.section_contains=[centers=p.c,frames=p.m,weights,&c](double t,const Vec3& point) {
      auto cp=centers; auto fp=frames; auto r=weights;
      for(int k=5;k>0;--k) for(int j=0;j<k;++j) {
        cp[j]=(1-t)*cp[j]+t*cp[j+1]; fp[j]=(1-t)*fp[j]+t*fp[j+1];
        r[j]=(1-t)*r[j]+t*r[j+1];
      }
      const Eigen::Matrix2d gram=fp[0].transpose()*fp[0];
      const Eigen::Vector2d z=gram.inverse()*fp[0].transpose()*(point-cp[0]);
      return (point-cp[0]-fp[0]*z).norm()<1e-6 && c.gauge(z,r[0])<=1+1e-9;
    };
    out[i].box=p.box; out[i].unresolved=&unresolved;
  }
  return out;
}
}

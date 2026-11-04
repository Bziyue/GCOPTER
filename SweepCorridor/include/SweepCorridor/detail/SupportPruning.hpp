#pragma once
// SPDX-License-Identifier: MIT
#include "CubicOptimizer.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>

namespace SweepCorridor::internal::cubic {
inline void pruneProjectedSupports(CubicProblem::Group& group,const Eigen::VectorXd& x,
                            double max_offset,double max_radius,std::vector<CubicProblem::Group::Proof>* proofs) {
  struct Point {Eigen::Vector2d projected;Vec3 world;int id;};
  if(group.supports.size()<4)return;
  Eigen::Matrix3d gram=Eigen::Matrix3d::Zero();
  for(int k=0;k<group.count;++k)gram.noalias()+=group.maps[k]*group.maps[k].transpose();
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(gram);
  const Eigen::Matrix<double,3,2> basis=eigen.eigenvectors().rightCols<2>();
  const Vec3 proof_center=group.geometry(x).head<3>();
  const Vec3 shift=group.reference.head<3>()-proof_center;
  std::vector<Point> points;points.reserve(group.supports.size()+1);
  for(const auto& p:group.supports) {
    const double gap=p.bound-p.normal.dot(proof_center);
    // A separating origin is required for the polar representation. This is
    // established once per group, never inside a Newton iteration.
    if(gap<=0)return;
    const Vec3 q=p.normal/gap;
    points.push_back({basis.transpose()*q,q,p.original_id});
  }
  const auto original=points;
  points.push_back({Eigen::Vector2d::Zero(),Vec3::Zero(),-1});
  std::sort(points.begin(),points.end(),[](const Point& a,const Point& b) {
    return std::make_tuple(a.projected[0],a.projected[1],a.id)<
           std::make_tuple(b.projected[0],b.projected[1],b.id);
  });
  auto cross=[](const Eigen::Vector2d& a,const Eigen::Vector2d& b){return a[0]*b[1]-a[1]*b[0];};
  std::vector<Point> hull;hull.reserve(points.size()*2);
  for(const auto& p:points) {
    while(hull.size()>1&&cross(hull.back().projected-hull[hull.size()-2].projected,
                              p.projected-hull.back().projected)<=0)hull.pop_back();
    hull.push_back(p);
  }
  const auto lower=hull.size();
  for(int j=int(points.size())-2;j>=0;--j) {
    const auto& p=points[j];
    while(hull.size()>lower&&cross(hull.back().projected-hull[hull.size()-2].projected,
                                  p.projected-hull.back().projected)<=0)hull.pop_back();
    hull.push_back(p);
  }
  if(hull.size()<4)return;
  hull.pop_back();
  std::vector<bool> vertex(group.supports.size(),false);
  for(const auto& p:hull)if(p.id>=0)vertex[p.id]=true;
  for(const auto& p:original) {
    if(vertex[p.id])continue;
    for(size_t edge=0;edge<hull.size();++edge) {
      const auto& a=hull[edge];const auto& b=hull[(edge+1)%hull.size()];
      const double determinant=cross(a.projected,b.projected);
      if(determinant==0)continue;
      const double left=cross(p.projected,b.projected)/determinant;
      const double right=cross(a.projected,p.projected)/determinant;
      if(!(left>=0&&right>=0&&left+right<=1))continue;
      const Vec3 residual=p.world-left*a.world-right*b.world;
      double reach=0;
      for(int k=0;k<group.count;++k) {
        const Eigen::Vector2d v=group.maps[k].transpose()*residual;
        const double norm=v.norm();
        reach+=(max_offset+max_radius)*norm;
      }
      double upper=left+right+residual.dot(shift)+reach;
      double guard=1e-10*(1+left+right+residual.cwiseAbs().dot(shift.cwiseAbs())+reach);
      // This is a sufficient implication in the original 3-D variable domain.
      // Only hull vertices are witnesses, so simultaneous deletions cannot
      // depend on one another. A poor projection simply keeps more rows.
      if(upper<1-guard) {
        auto& support=group.supports[p.id];support.redundant=true;
        if(proofs)proofs->push_back({p.id,a.id,b.id,left,right,upper});
        break;
      }
    }
  }
}
}

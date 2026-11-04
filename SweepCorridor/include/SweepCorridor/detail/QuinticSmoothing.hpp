// SPDX-License-Identifier: MIT
#pragma once
#include "QuinticOptimizer.hpp"

namespace SweepCorridor::internal::detail {
using V9 = Eigen::Matrix<double,9,1>;
using Curve9 = std::array<V9,6>;
inline V9 pack(const Vec3& c, const Mat32& m) {
  V9 v; v.head<3>()=c; v.tail<6>()=Eigen::Map<const Eigen::Matrix<double,6,1>>(m.data()); return v;
}
inline Mat32 unpackShape(const V9& v) { return Eigen::Map<const Mat32>(v.data()+3); }
inline double maxAxisSquared(const Mat32& m) {
  const M2 g=m.transpose()*m;
  return .5*(g.trace()+std::hypot(g(0,0)-g(1,1),2*g(0,1)));
}
inline Curve9 restricted(Curve9 p, double begin, double end) {
  auto splitAt=[](Curve9 p,double t,bool left) {
    Curve9 out; out[left?0:5]=p[left?0:5];
    for(int k=1;k<6;++k) {
      for(int j=0;j<6-k;++j) p[j]=(1-t)*p[j]+t*p[j+1];
      out[left?k:5-k]=p[left?0:5-k];
    }
    return out;
  };
  if(end<1) p=splitAt(p,end,true);
  if(begin>0) p=splitAt(p,begin/end,false);
  return p;
}
struct JetCut {
  int span;
  double bound, offset[2];
  V2 z, direction[2];
};

// Natural cubic interpolation minimizes the integrated squared second
// derivative for fixed knot values. One tridiagonal solve supplies all nine
// world channels. Shared node jet fractions retain C2 after local clipping.
inline std::vector<Curve9> smoothSweep(const Corridor& base,
    const std::vector<SupportPlane>& cover, double reserve, double max_radius, double& mean_fraction,
    const Stop& stop) {
  const int count=base.hermiteNodes().size();
  std::vector<V9> values(count), second(count,V9::Zero()), first(count);
  std::vector<double> diag(count,1.), upper(count,0.);
  const auto& knots=base.knots();
  for(int i=0;i<count;++i) {
    const auto& s=base.hermiteNodes()[i].state;
    values[i]=pack(s.center,(1-reserve)*s.shape);
  }
  for(int i=1;i<count-1;++i) {
    const double a=knots[i]-knots[i-1], b=knots[i+1]-knots[i];
    diag[i]=2*(a+b)-a*upper[i-1]/diag[i-1]; upper[i]=b;
    second[i]=6*((values[i+1]-values[i])/b-(values[i]-values[i-1])/a)-a*second[i-1]/diag[i-1];
  }
  for(int i=count-2;i>=1;--i) second[i]=(second[i]-upper[i]*second[i+1])/diag[i];
  for(int i=0;i<count-1;++i) {
    const double h=knots[i+1]-knots[i];
    first[i]=(values[i+1]-values[i])/h-h*(2*second[i]+second[i+1])/6;
  }
  const double last_h=knots.back()-knots[count-2];
  first.back()=(values.back()-values[count-2])/last_h+last_h*(second[count-2]+2*second.back())/6;
  std::vector<Curve9> start(count-1), delta(count-1);
  for(int i=0;i<count-1;++i) {
    const double h=knots[i+1]-knots[i];
    Curve9 target{values[i],values[i]+h*first[i]/5,
      values[i]+2*h*first[i]/5+h*h*second[i]/20,
      values[i+1]-2*h*first[i+1]/5+h*h*second[i+1]/20,
      values[i+1]-h*first[i+1]/5,values[i+1]};
    for(int j=0;j<6;++j) {
      start[i][j]=pack(base.centerControls()[6*i+j],(1-reserve)*base.shapeControls()[6*i+j]);
      delta[i][j]=target[j]-start[i][j];
    }
  }
  std::vector<JetCut> cuts;
  cuts.reserve(6*cover.size());
  std::vector<std::vector<std::pair<int,int>>> incident(count);
  for(const auto& plane:cover) {
    const auto s=restricted(start[plane.span],plane.begin,plane.end);
    Curve9 left=delta[plane.span], right=left;
    for(int j=0;j<3;++j) right[j].setZero();
    for(int j=3;j<6;++j) left[j].setZero();
    left=restricted(left,plane.begin,plane.end); right=restricted(right,plane.begin,plane.end);
    for(int j=0;j<6;++j) {
      JetCut cut;cut.span=plane.span;
      cut.bound=plane.bound-plane.normal.dot(s[j].head<3>());
      cut.z=unpackShape(s[j]).transpose()*plane.normal;
      cut.offset[0]=plane.normal.dot(left[j].head<3>());
      cut.offset[1]=plane.normal.dot(right[j].head<3>());
      cut.direction[0]=unpackShape(left[j]).transpose()*plane.normal;
      cut.direction[1]=unpackShape(right[j]).transpose()*plane.normal;
      for(int k=0;k<2;++k)
        if(std::abs(cut.offset[k])+cut.direction[k].norm()>1e-20)
          incident[plane.span+k].push_back({int(cuts.size()),k});
      cuts.push_back(cut);
    }
  }
  std::vector<double> fractions(count,0.);
  // Each unreduced Hermite control depends on one node's jet. Enforce the
  // spectral radius cap locally; only a boundary crossing needs bisection.
  auto radiusLimit=[&](int node,double current,double step,
                       const std::vector<Curve9>& origin,const std::vector<Curve9>& direction) {
    auto feasible=[&](double value) {
      for(int span=std::max(0,node-1);span<=std::min(count-2,node);++span)
        for(int j=(span==node?0:3);j<(span==node?3:6);++j)
          if(maxAxisSquared(unpackShape(origin[span][j]+value*direction[span][j]))>max_radius*max_radius)
            return false;
      return true;
    };
    if(feasible(current+step)) return step;
    double lo=0,hi=step;
    for(int k=0;k<24;++k) {
      const double mid=.5*(lo+hi);
      if(feasible(current+mid)) lo=mid; else hi=mid;
    }
    return lo;
  };
  auto limitStep=[](const JetCut& c,int k,double limit) {
    const double t=c.bound, dt=-c.offset[k];
    const V2& dz=c.direction[k];
    if(t+limit*dt>(c.z+limit*dz).norm()) return limit;
    const double slack=(t-c.z.norm())*(t+c.z.norm());
    if(slack<=0) return 0.;
    if(dt<0) limit=std::min(limit,-t/dt);
    return std::min(limit,Optimizer::positiveRoot(dt*dt-dz.squaredNorm(),
      2*(t*dt-c.z.dot(dz)),slack));
  };
  for(int pass=0;pass<4;++pass) {
    stop();
    for(int at=0;at<count;++at) {
      const int i=pass%2?count-1-at:at;
      double step=1-fractions[i];
      for(const auto& item:incident[i]) step=limitStep(cuts[item.first],item.second,step);
      step=radiusLimit(i,fractions[i],step,start,delta);
      step=std::max(0.,.995*step); fractions[i]+=step;
      for(const auto& item:incident[i]) {
        auto& cut=cuts[item.first]; const int k=item.second;
        cut.bound-=step*cut.offset[k]; cut.z+=step*cut.direction[k];
      }
    }
  }
  std::vector<Curve9> out=start;
  auto construct=[&] {
    for(int i=0;i<count-1;++i) for(int j=0;j<6;++j)
      out[i][j]=start[i][j]+fractions[i+(j>=3)]*delta[i][j];
  };
  construct();
  // Changing jets can alter local orientation. The native tube was already
  // certified; interpolate shared jets back only when the full-disk test needs it.
  for(int attempt=0;attempt<24;++attempt) {
    bool regular=true;
    for(int i=0;i<count-1;++i) {
      std::array<Vec3,6> c;std::array<Mat32,6> m;
      for(int j=0;j<6;++j) {c[j]=out[i][j].head<3>();m[j]=unpackShape(out[i][j]);}
      if(!regularTube(c.data(),m.data())) {regular=false;break;}
    }
    if(regular) break;
    if(attempt==23) throw Failure{Status::DegeneratePath,"smoothed full-disk certificate failed"};
    for(auto& f:fractions) f*=.5;
    construct();
  }
  // Refill the radius reserve using the same cover. Add a C2 value basis to
  // the already smooth shape; center jets are unaffected. Each coordinate
  // reaches the exact boundary of its incident SOCs without a new solver.
  const auto smooth=out;
  std::vector<Curve9> refill(count-1);
  for(int i=0;i<count-1;++i) for(int j=0;j<6;++j)
    refill[i][j]=pack(Vec3::Zero(),unpackShape(values[i+(j>=3)]));
  cuts.clear(); for(auto& list:incident) list.clear();
  for(const auto& plane:cover) {
    auto p=restricted(out[plane.span],plane.begin,plane.end);
    Curve9 left=refill[plane.span],right=left;
    for(int j=0;j<3;++j) right[j].setZero();
    for(int j=3;j<6;++j) left[j].setZero();
    left=restricted(left,plane.begin,plane.end);right=restricted(right,plane.begin,plane.end);
    for(int j=0;j<6;++j) {
      JetCut cut;cut.span=plane.span;
      cut.bound=plane.bound-plane.normal.dot(p[j].head<3>());
      cut.z=unpackShape(p[j]).transpose()*plane.normal;
      cut.offset[0]=cut.offset[1]=0;
      cut.direction[0]=unpackShape(left[j]).transpose()*plane.normal;
      cut.direction[1]=unpackShape(right[j]).transpose()*plane.normal;
      for(int k=0;k<2;++k) if(cut.direction[k].squaredNorm()>1e-28)
        incident[plane.span+k].push_back({int(cuts.size()),k});
      cuts.push_back(cut);
    }
  }
  std::vector<double> increase(count,0.);
  for(int pass=0;pass<2;++pass) {
    stop();
    for(int at=0;at<count;++at) {
      const int i=pass?count-1-at:at;
      double step=.05-increase[i];
      for(const auto& item:incident[i]) step=limitStep(cuts[item.first],item.second,step);
      step=radiusLimit(i,increase[i],step,smooth,refill);
      step=std::max(0.,.995*step);increase[i]+=step;
      for(const auto& item:incident[i]) cuts[item.first].z+=step*cuts[item.first].direction[item.second];
    }
  }
  for(int attempt=0;attempt<24;++attempt) {
    bool regular=true;
    for(int i=0;i<count-1;++i) {
      std::array<Vec3,6> c;std::array<Mat32,6> m;
      for(int j=0;j<6;++j) {
        out[i][j]=smooth[i][j]+increase[i+(j>=3)]*refill[i][j];
        c[j]=out[i][j].head<3>();m[j]=unpackShape(out[i][j]);
      }
      regular &= regularTube(c.data(),m.data());
    }
    if(regular) break;
    if(attempt==23) throw Failure{Status::DegeneratePath,"refilled full-disk certificate failed"};
    for(auto& f:increase) f*=.5;
  }
  // One audit at the stage boundary, reusing the same continuous cover.
  for(const auto& plane:cover) {
    const auto p=restricted(out[plane.span],plane.begin,plane.end);
    for(const auto& v:p)
      if(plane.normal.dot(v.head<3>())+(unpackShape(v).transpose()*plane.normal).norm()>plane.bound+1e-9)
        throw Failure{Status::VerificationFailed,"smoothed support certificate failed"};
  }
  mean_fraction=std::accumulate(fractions.begin(),fractions.end(),0.)/count;
  return out;
}
}

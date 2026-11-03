// SPDX-License-Identifier: MIT
#pragma once
#include "minkowski_geometry.hpp"

namespace gcopter::direct::minkowski {
struct Optimizer {
  const Options& settings;
  const direct::Options& o;
  const Templates& bank;
  const Stop& stop;
  const std::vector<Node>& nodes;
  std::vector<H6> diagonal, upper, transfer;
  std::vector<Eigen::LLT<H6>> factors;
  std::vector<V6> gradient, rhs, step, scaling;
  Optimizer(const Options& s, const Templates& t, const Stop& check,
            const std::vector<Node>& n)
      : settings(s), o(s.geometry), bank(t), stop(check), nodes(n) {}
  double objective(const std::vector<V6>& x, const std::vector<LinearCut>& cuts,
                   double mu, bool derivatives) {
    const int count=x.size();
    if(derivatives) {
      diagonal.assign(count,H6::Zero());
      upper.assign(count-1,H6::Zero());
      gradient.assign(count,V6::Zero());
    }
    double cost=0;
    for(int i=0;i<count;++i) {
      const Weights r=x[i].tail<4>(), v=bank.mixed*r;
      const double area=r.dot(v);
      if(area<=0) return INFINITY;
      cost-=std::log(area);
      if(derivatives) {
        gradient[i].tail<4>()-=2*v/area;
        diagonal[i].bottomRightCorner<4,4>()+=4*(v*v.transpose())/(area*area)-2*bank.mixed/area;
      }
      for(int k=0;k<bank.count;++k) {
        const double lo=r[k]-1e-6*o.min_radius, hi=o.max_radius-r[k];
        if(lo<=0 || hi<=0) return INFINITY;
        cost-=mu*(std::log(lo)+std::log(hi));
        if(derivatives) {
          gradient[i][2+k]+=mu*(1/hi-1/lo);
          diagonal[i](2+k,2+k)+=mu*(1/(hi*hi)+1/(lo*lo));
        }
      }
      const V2 offset=x[i].head<2>();
      const double gap=o.max_offset*o.max_offset-offset.squaredNorm();
      if(gap<=0) return INFINITY;
      const double anchor=.02/(o.max_radius*o.max_radius);
      cost+=anchor*offset.squaredNorm()-mu*std::log(gap);
      if(derivatives) {
        gradient[i].head<2>()+=(2*mu/gap+2*anchor)*offset;
        diagonal[i].topLeftCorner<2,2>()+=(2*mu/gap+2*anchor)*M2::Identity()+
            4*mu/(gap*gap)*(offset*offset.transpose());
      }
    }
    for(int i=0;i+1<count;++i) {
      const double smoothing=.01*std::min(4.,o.control_spacing/(nodes[i+1].s-nodes[i].s));
      const V6 delta=x[i+1]-x[i];
      cost+=smoothing*delta.squaredNorm();
      if(derivatives) {
        gradient[i]-=2*smoothing*delta;
        gradient[i+1]+=2*smoothing*delta;
        diagonal[i].diagonal().array()+=2*smoothing;
        diagonal[i+1].diagonal().array()+=2*smoothing;
        upper[i].diagonal().array()-=2*smoothing;
      }
    }
    for(const auto& c:cuts) {
      const double gap=c.h-value(c,x);
      if(gap<=0) return INFINITY;
      cost-=mu*std::log(gap);
      if(derivatives) {
        const double first=mu/gap, second=first/gap;
        const int i=c.span;
        if(c.a.squaredNorm()>0) {
          gradient[i]+=first*c.a;
          diagonal[i].noalias()+=second*(c.a*c.a.transpose());
        }
        if(c.b.squaredNorm()>0) {
          gradient[i+1]+=first*c.b;
          diagonal[i+1].noalias()+=second*(c.b*c.b.transpose());
        }
        if(c.a.squaredNorm()>0 && c.b.squaredNorm()>0)
          upper[i].noalias()+=second*(c.a*c.b.transpose());
      }
    }
    return cost;
  }
  void solve(std::vector<V6>& x, const std::vector<LinearCut>& cuts, Metrics& metrics) {
    const int count=x.size();
    factors.resize(count); transfer.resize(count-1); rhs.resize(count);
    step.resize(count); scaling.resize(count);
    for(double mu:{.003,.0001,.00001}) for(int iteration=0;iteration<settings.newton_iterations;++iteration) {
      stop(); ++metrics.newton_steps;
      const double before=objective(x,cuts,mu,true);
      if(!std::isfinite(before)) throw Failure{Status::VerificationFailed,"noninterior Minkowski iterate"};
      for(int i=0;i<count;++i) for(int k=0;k<6;++k) {
        if(!((k<2 && (i==0 || i==count-1)) || k>=2+bank.count)) continue;
        gradient[i][k]=0;
        diagonal[i].row(k).setZero(); diagonal[i].col(k).setZero(); diagonal[i](k,k)=1;
        if(i+1<count) upper[i].row(k).setZero();
        if(i) upper[i-1].col(k).setZero();
      }
      for(int i=0;i<count;++i) scaling[i]=diagonal[i].diagonal().cwiseMax(1e-30).cwiseSqrt();
      for(int i=0;i<count;++i) {
        diagonal[i]=scaling[i].cwiseInverse().asDiagonal()*diagonal[i]*scaling[i].cwiseInverse().asDiagonal();
        if(i+1<count) upper[i]=scaling[i].cwiseInverse().asDiagonal()*upper[i]*scaling[i+1].cwiseInverse().asDiagonal();
      }
      bool factored=false;
      for(double damping:{1e-10,1e-8,1e-6,1e-4}) {
        factored=true;
        for(int i=0;i<count;++i) {
          H6 h=diagonal[i]; rhs[i]=-gradient[i].cwiseQuotient(scaling[i]);
          if(i) {
            h-=upper[i-1].transpose()*transfer[i-1];
            rhs[i]-=upper[i-1].transpose()*rhs[i-1];
          }
          h.diagonal().array()+=damping;
          factors[i].compute(h);
          if(factors[i].info()!=Eigen::Success) { factored=false; break; }
          rhs[i]=factors[i].solve(rhs[i]).eval();
          if(i+1<count) transfer[i]=factors[i].solve(upper[i]);
        }
        if(factored) break;
      }
      if(!factored) throw Failure{Status::VerificationFailed,"Minkowski Newton factorization failed"};
      for(int i=count-1;i>=0;--i)
        step[i]=rhs[i]-(i+1<count ? V6(transfer[i]*step[i+1]):V6::Zero());
      double slope=0;
      for(int i=0;i<count;++i) {
        step[i]=step[i].cwiseQuotient(scaling[i]);
        slope+=gradient[i].dot(step[i]);
      }
      if(-slope<1e-7*count) break;
      // All obstacle rows are affine: their first boundary along the step is
      // exact. Weight bounds are affine as well. The offset ball is left to
      // the usual objective/Armijo check.
      double alpha=1;
      for(const auto& c:cuts) {
        const double rate=c.a.dot(step[c.span])+c.b.dot(step[c.span+1]);
        if(rate>0) alpha=std::min(alpha,.99*(c.h-value(c,x))/rate);
      }
      for(int i=0;i<count;++i) for(int k=2;k<2+bank.count;++k) {
        if(step[i][k]>0) alpha=std::min(alpha,.99*(o.max_radius-x[i][k])/step[i][k]);
        if(step[i][k]<0) alpha=std::min(alpha,.99*(1e-6*o.min_radius-x[i][k])/step[i][k]);
      }
      bool accepted=false;
      auto trial=x;
      for(int ls=0;ls<28;++ls) {
        for(int i=0;i<count;++i) trial[i]=x[i]+alpha*step[i];
        const double after=objective(trial,cuts,mu,false);
        if(std::isfinite(after) && after<=before+1e-3*alpha*slope) {
          x.swap(trial); accepted=true; break;
        }
        alpha*=.5;
      }
      if(!accepted) break;
    }
  }
  void optimize(std::vector<V6>& x, const std::vector<LinearCut>& cuts, Metrics& metrics) {
    bool centered=true;
    for(const auto& v:x) centered &= v.head<2>().squaredNorm()<1e-24;
    if(centered) {
      std::vector<double> radius(x.size(),.9*o.max_radius);
      for(const auto& c:cuts) {
        const double a=c.a.tail<4>().sum(), b=c.b.tail<4>().sum();
        if(a+b<=0) continue;
        const double r=.85*c.h/(a+b);
        if(a>0) radius[c.span]=std::min(radius[c.span],r);
        if(b>0) radius[c.span+1]=std::min(radius[c.span+1],r);
      }
      auto trial=x;
      for(int i=0;i<int(x.size());++i) for(int k=0;k<bank.count;++k)
        trial[i][2+k]=std::max(x[i][2+k],radius[i]);
      bool feasible=true;
      for(const auto& c:cuts) if(value(c,trial)>=c.h-1e-10) { feasible=false; break; }
      if(feasible) x.swap(trial);
    }
    ++metrics.optimization_passes;
    solve(x,cuts,metrics);
    metrics.active_constraints=cuts.size();
  }
};
}  // namespace gcopter::direct::minkowski

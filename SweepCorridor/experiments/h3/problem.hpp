// SPDX-License-Identifier: MIT
#pragma once
#include "model.hpp"
#include <map>

namespace SweepCorridor::experiments::h3 {
struct Group : c3::CubicProblem::Group {
  struct SingleSupport {
    Vec3 normal,constant;
    Eigen::Vector2d q;
    Eigen::Matrix3d shape_gram;
  };
  std::vector<SingleSupport> single_supports;
  void prepareSingle(){
    if(count!=1)return;
    single_supports.reserve(retained);
    for(size_t i=0;i<retained;++i){const auto& p=supports[i];SingleSupport s;
      s.normal=center_weights[0]*p.normal;s.q=maps[0].transpose()*p.normal;s.constant=p.project(reference);
      const double a=s.q[0],b=s.q[1];s.shape_gram<<a*a,a*b,0,a*b,a*a+b*b,a*b,0,a*b,b*b;
      single_supports.push_back(std::move(s));}
  }
  // H3 B0/B1 share one nodal value, as do B2/B3. Accumulate their
  // Lorentz jets directly in six variables: no 9x9 world Hessian or pullback.
  template<bool WithValue>
  bool singleJets(const Eigen::VectorXd& x,double mu,Eigen::VectorXd& gradient,BlockTridiagonal& hessian,double& cost)const{
    State g=State::Zero();Block h=Block::Zero();const auto state=x.segment<6>(first);
    for(const auto& p:single_supports){
      const Vec3 v(p.constant[0]-p.normal.dot(state.head<3>()),p.constant[1]+p.q.dot(state.segment<2>(3)),p.constant[2]+p.q.dot(state.tail<2>()));
      const double slack=v[0]*v[0]-v.tail<2>().squaredNorm();if(v[0]<=0||slack<=0)return false;
      if constexpr(WithValue)cost-=mu*std::log(slack);
      const double f=2*mu/slack;State a;a.head<3>()=v[0]*p.normal;
      a.tail<3>()<<v[1]*p.q[0],v[1]*p.q[1]+v[2]*p.q[0],v[2]*p.q[1];
      g+=f*a;h.noalias()+=(2*f/slack)*a*a.transpose();
      h.topLeftCorner<3,3>().noalias()-=f*p.normal*p.normal.transpose();h.bottomRightCorner<3,3>()+=f*p.shape_gram;
    }
    gradient.segment<6>(first)+=g;hessian.diagonal[first/6]+=h;return true;
  }
  template<int Count> void accumulateFixed(const V9& g,const Eigen::Matrix<double,9,9>& h,Eigen::VectorXd& gradient,BlockTridiagonal& hessian)const{
    accumulateGradient(g,gradient);
    std::array<Map,Count> a;
    for(int k=0;k<Count;++k){a[k].setZero();a[k].template topLeftCorner<3,3>().diagonal().array()=center_weights[k];
      a[k].template block<3,1>(3,3)=maps[k].col(0);a[k].template block<3,1>(3,4)=maps[k].col(1);
      a[k].template block<3,1>(6,4)=maps[k].col(0);a[k].template block<3,1>(6,5)=maps[k].col(1);}
    for(int b=0;b<Count;++b){const Map product=h*a[b];
      for(int a_id=b;a_id<Count;++a_id)hessian.addBlock(first/6+a_id,first/6+b,a[a_id].transpose()*product);}
  }
  void accumulate(const V9& g,const Eigen::Matrix<double,9,9>& h,Eigen::VectorXd& gradient,BlockTridiagonal& hessian)const{
    if(count==1)accumulateFixed<1>(g,h,gradient,hessian);else accumulateFixed<2>(g,h,gradient,hessian);
  }
};
struct Cone {
  int node;
  Vec3 constant;
  Eigen::Matrix<double,3,12> map;
  Eigen::Matrix<double,12,12> curvature;
  Vec3 value(const Eigen::VectorXd& x)const{return constant+map*x.segment<12>(6*node);}
  void accumulate(const Vec3& v,double mu,double slack,Eigen::VectorXd& g,BlockTridiagonal& h)const{
    const Vec3 signed_v(-v[0],v[1],v[2]);
    const Eigen::Matrix<double,12,1> a=map.transpose()*signed_v;
    const double f=2*mu/slack;
    g.segment<12>(6*node)+=f*a;
    const Eigen::Matrix<double,12,12> block=(2*f/slack)*a*a.transpose()+f*curvature;
    h.diagonal[node]+=block.topLeftCorner<6,6>();h.diagonal[node+1]+=block.bottomRightCorner<6,6>();h.lower[node]+=block.bottomLeftCorner<6,6>();
  }
};
struct Problem {
  const Model& model;const GeometryOptions& options;OptimizerSettings settings;
  struct Constraints {std::vector<Group> groups;int count=0,retained=0,merged=0,axes=0;};
  std::shared_ptr<const Constraints> geometry;
  std::shared_ptr<const Energy> energy;
  const BlockTridiagonal& quadratic;const Eigen::VectorXd& linear;
  std::vector<Cone> cuts;
  bool reused_construction=false;
  double pruning_ms=0;
  int basis_cache_hits=0,basis_cache_misses=0,group_lookup_reuses=0;
  using LineModel=c3::CubicProblem::LineModel;
  Problem(const Model& m,const std::vector<internal::detail::SupportPlane>& cover,double length,
      const GeometryOptions& o,const OptimizerSettings& s,std::shared_ptr<const Energy> e={},const Problem* previous=nullptr,bool progress=true);
  double objective(const Eigen::VectorXd& x,double mu,Eigen::VectorXd* g=nullptr,BlockTridiagonal* h=nullptr,bool evaluate_cost=true)const;
  LineModel prepareLine(const Eigen::VectorXd& x,const Eigen::VectorXd& direction,double mu)const;
  void optimize(Eigen::VectorXd& x,OptimizerMetrics& metrics,const Deadline& deadline,int first_stage=0,int last_stage=3)const;
};

inline Problem::Problem(const Model& m,const std::vector<internal::detail::SupportPlane>& cover,double length,
    const GeometryOptions& o,const OptimizerSettings& s,std::shared_ptr<const Energy> e,const Problem* previous,bool progress)
    :model(m),options(o),settings(s),energy(e?std::move(e):std::make_shared<Energy>(m,length,s)),quadratic(energy->quadratic),linear(energy->linear){
  if(previous){geometry=previous->geometry;reused_construction=true;}
  else {
    auto data=std::make_shared<Constraints>();
    struct Rows {c3::Curve<3,Eigen::Vector2d> weights;c3::Curve<3,V9> offsets;std::array<int,4> ids{{-1,-1,-1,-1}};};
    std::map<std::tuple<int,double,double>,Rows> rows;
    std::map<std::tuple<int,double,double,int>,int> ids;
    for(const auto& p:cover){auto inserted=rows.try_emplace(std::make_tuple(p.span,p.begin,p.end));auto& row=inserted.first->second;
      if(inserted.second){
        const c3::Curve<3,Eigen::Vector2d> weights{{{1,0},{1,0},{0,1},{0,1}}};
        row.weights=c3::restrictCurve<3>(weights,p.begin,p.end);row.offsets=c3::restrictCurve<3>(m.offsets(p.span),p.begin,p.end);++basis_cache_misses;
      }else ++basis_cache_hits;
      for(int j=0;j<(p.begin==p.end?1:4);++j){
        if(row.ids[j]<0){auto key=std::make_tuple(p.span,p.begin,p.end,j);
          if(j==0||j==3){const double t=j==0?p.begin:p.end;key=t==1?std::make_tuple(p.span+1,0.,0.,0):std::make_tuple(p.span,t,t,0);}
          const auto found=ids.emplace(key,data->groups.size());row.ids[j]=found.first->second;
          if(found.second){Group g;g.reference=row.offsets[j];g.first=6*p.span;g.count=2;
            for(int k=0;k<2;++k){const double w=row.weights[j][k];g.center_weights[k]=w;g.maps[k]=w*m.frames[p.span+k];g.reference.head<3>()+=w*m.reference[p.span+k];}
            // Unsubdivided controls are single-node rows, not 12-variable rows.
            if(row.weights[j][1]==0)g.count=1;
            else if(row.weights[j][0]==0){g.first+=6;g.count=1;g.center_weights[0]=g.center_weights[1];g.maps[0]=g.maps[1];}
            data->groups.push_back(std::move(g));}
        }else ++group_lookup_reuses;
        data->groups[row.ids[j]].supports.push_back({p.normal,p.bound});
      }
    }
    for(auto& g:data->groups){auto& supports=g.supports;
      auto key=[](const auto& p){return std::make_tuple(p.normal[0],p.normal[1],p.normal[2]);};
      std::sort(supports.begin(),supports.end(),[&](const auto& a,const auto& b){return key(a)<key(b);});
      size_t count=0;for(const auto& p:supports){if(count&&key(p)==key(supports[count-1]))supports[count-1].bound=std::min(supports[count-1].bound,p.bound);else supports[count++]=p;}
      data->merged+=supports.size()-count;supports.resize(count);
      for(size_t i=0;i<supports.size();++i){auto& p=supports[i];p.original_id=i;
        for(int a=0;a<3;++a)if(p.normal[a]!=0&&p.normal[(a+1)%3]==0&&p.normal[(a+2)%3]==0)p.axis=a;}
    }
    StageTimer timer(settings.profile);
    for(auto& g:data->groups){c3::pruneProjectedSupportsImpl<true>(g,m.x,o.max_offset,o.max_radius,nullptr);
      g.retained=std::stable_partition(g.supports.begin(),g.supports.end(),[](const auto& p){return !p.redundant;})-g.supports.begin();
      data->count+=g.supports.size();data->retained+=g.retained;for(size_t j=0;j<g.retained;++j)data->axes+=g.supports[j].axis>=0;}
#ifndef H3_WORLD_ASSEMBLY
    for(auto& g:data->groups)g.prepareSingle();
#endif
    pruning_ms=timer.elapsed();geometry=std::move(data);
  }
  if(!progress)return;
  const auto initial=m.geometry();
  for(int i=0;i<m.basis.count-1;++i){const double h=m.basis.breaks[i+1]-m.basis.breaks[i];
    // Q1-Q0 and Q3-Q2 contain only shared jets. Their barriers are constant
    // in this solve. Only 3(Q2-Q1) needs a two-node cone.
    const V9 offset=pack(3*(m.reference[i+1]-m.reference[i]),Mat32::Zero())-h*(m.jets[i]+m.jets[i+1]);
    for(double t:{0.,.5,1.}){const auto section=initial.section(m.basis.breaks[i]+t*h);
      const Vec3 n=section.shape.col(0).cross(section.shape.col(1)).normalized();
      Cone c;c.node=i;c.constant=Vec3(n.dot(offset.head<3>())-1e-5*length*h,n.dot(offset.segment<3>(3)),n.dot(offset.tail<3>()));
      for(int k=0;k<2;++k){const Map j=(k?3.:-3.)*m.jacobian(i+k);
        for(int channel=0;channel<3;++channel)c.map.block<1,6>(channel,6*k)=n.transpose()*j.block<3,6>(3*channel,0);}
      const Vec3 v=c.value(m.x);if(v[0]<=v.tail<2>().norm()+1e-9)continue;
      Eigen::Matrix<double,3,12> signed_map=c.map;signed_map.row(0)*=-1;
      c.curvature=c.map.transpose()*signed_map;cuts.push_back(std::move(c));
    }
  }
}
}

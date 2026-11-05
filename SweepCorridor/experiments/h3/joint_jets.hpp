// SPDX-License-Identifier: MIT
#pragma once
#include "volume_jets.hpp"
#include "problem.hpp"
namespace SweepCorridor::experiments::h3 {
using V15=Eigen::Matrix<double,15,1>;
using M15=Eigen::Matrix<double,15,15>;
using NodalGroup=Group;
struct JointPatch {
  HermitePatch patch;
  const Model& model;const GeometryOptions& options;
  State value;
  Mat32 frame;
  struct Sample {V9 value;Vec3 tangent;double weight,node_basis,jet_basis,node_first,jet_first;int span;Eigen::Vector4d basis,first;};
  struct Group {
    V9 reference;std::vector<c3::CubicProblem::Support> supports;int span;Eigen::Vector4d basis;double node_basis,jet_basis;
    const std::vector<c3::CubicProblem::Support>* reused=nullptr;
    const auto& rows()const{return reused?*reused:supports;}
  };
  struct EnergyRow {int span;double f0,f1,q0,q1,weight,inverse_h2;};
  V9 energy_weights;
  std::vector<EnergyRow> energy_rows;
  std::array<Sample,10> samples;
  int sample_count=0;
  std::vector<Group> groups;
  V15 energy_gradient=V15::Zero();M15 energy_hessian=M15::Zero();
  double old_volume=0;
  JointPatch(const Model& m,const Geometry& geometry,int node,double length,const GeometryOptions& o,double factor,
             const std::vector<internal::detail::SupportPlane>& cover,const std::vector<std::vector<int>>& ids,
             const std::vector<const NodalGroup*>* shared=nullptr)
      :patch(m,geometry,node),model(m),options(o),value(m.x.segment<6>(6*node)),frame(m.frames[node]){
    const Map j=m.jacobian(node);
    energy_weights.head<3>().setConstant(factor*OptimizerSettings::center_energy/(length*length*length));
    energy_weights.tail<6>().setConstant(factor*OptimizerSettings::shape_energy/(length*length*length));
    std::map<std::tuple<int,double,double,int>,int> shared_rows;
    constexpr double t[5]={.046910077030668,.230765344947158,.5,.769234655052842,.953089922969332};
    constexpr double w[5]={.118463442528095,.239314335249683,.284444444444444,.239314335249683,.118463442528095};
    for(int span=patch.begin;span<patch.end;++span){const int k=span-patch.begin;const double h=m.basis.breaks[span+1]-m.basis.breaks[span];
      const auto& p=patch.curves[k];
      const double f0=(span==node?-6.:6.)/(h*h),f1=-f0;
      const auto& b=patch.basis[k];const double q0=6*(b[2]-2*b[1]+b[0])/(h*h),q1=6*(b[3]-2*b[2]+b[1])/(h*h);
      const double nn=2*h/3*(f0*f0+f0*f1+f1*f1),jj=2*h/3*(q0*q0+q0*q1+q1*q1);
      const double nj=h/3*(2*f0*q0+f0*q1+f1*q0+2*f1*q1);
      const Eigen::Matrix2d gram=frame.transpose()*frame;Eigen::Matrix3d shape_gram;
      shape_gram<<gram(0,0),gram(0,1),0,gram(0,1),gram.trace(),gram(0,1),0,gram(0,1),gram(1,1);
      energy_hessian.topLeftCorner<3,3>().diagonal().array()+=nn*energy_weights[0];
      energy_hessian.block<3,3>(3,3)+=nn*energy_weights[3]*shape_gram;
      energy_hessian.bottomRightCorner<9,9>().diagonal()+=jj*energy_weights;
      const Map cross=nj*energy_weights.asDiagonal()*j;
      energy_hessian.topRightCorner<6,9>()+=cross.transpose();energy_hessian.bottomLeftCorner<9,6>()+=cross;
      energy_rows.push_back({span,f0,f1,q0,q1,h/3,1/(h*h)});
      for(int q=0;q<5;++q){const double u=t[q],r=1-u;auto& s=samples[sample_count++];
        s.value=r*r*r*p[0]+3*r*r*u*p[1]+3*r*u*u*p[2]+u*u*u*p[3];
        s.node_basis=span==node?r*r*(1+2*u):u*u*(3-2*u);
        s.node_first=(span==node?-1.:1.)*6*r*u/h;
        s.jet_basis=3*r*r*u*patch.basis[k][1]+3*r*u*u*patch.basis[k][2];
        s.jet_first=(3*r*(1-3*u)*patch.basis[k][1]+3*u*(2-3*u)*patch.basis[k][2])/h;
        s.tangent=(3*r*r*(p[1]-p[0])+6*r*u*(p[2]-p[1])+3*u*u*(p[3]-p[2])).head<3>()/h;

        s.weight=std::acos(-1.)*h*w[q];
        s.span=span;s.basis<<r*r*r,3*r*r*u,3*r*u*u,u*u*u;
        s.first<<-3*r*r,3*r*(1-3*u),3*u*(2-3*u),3*u*u;s.first/=h;
      }
      std::map<std::pair<double,double>,std::array<int,4>> intervals;
      if(!shared)for(int id:ids[span]){const auto& c=cover[id];const auto key=std::make_pair(c.begin,c.end);
        auto found=intervals.find(key);
        if(found==intervals.end()){
          const auto p0=c3::restrictCurve<3>(p,c.begin,c.end);
          const c3::Curve<3,double> node_basis=span==node?c3::Curve<3,double>{1,1,0,0}:c3::Curve<3,double>{0,0,1,1};
          const auto nw=c3::restrictCurve<3>(node_basis,c.begin,c.end),jw=c3::restrictCurve<3>(patch.basis[k],c.begin,c.end);
          c3::Curve<3,Eigen::Vector4d> eye;for(int j=0;j<4;++j)eye[j]=Eigen::Vector4d::Unit(j);
          const auto weights=c3::restrictCurve<3>(eye,c.begin,c.end);
          std::array<int,4> rows;rows.fill(-1);
          for(int b=0;b<(c.begin==c.end?1:4);++b){if(nw[b]==0&&jw[b]==0)continue;
            auto key=std::make_tuple(span,c.begin,c.end,b);
            if(b==0||b==3){const double t=b==0?c.begin:c.end;key=t==1?std::make_tuple(span+1,0.,0.,0):std::make_tuple(span,t,t,0);}
            const auto entry=shared_rows.emplace(key,groups.size());rows[b]=entry.first->second;
            if(entry.second)groups.push_back({p0[b],{},span,weights[b],nw[b],jw[b]});
          }
          found=intervals.emplace(key,rows).first;
        }
        for(int b=0;b<(c.begin==c.end?1:4);++b)if(found->second[b]>=0)
          groups[found->second[b]].supports.push_back({c.normal,c.bound});
      }
    }
    if(shared)for(const auto* source:*shared){Group g;g.span=source->span;g.basis=source->bernstein;g.reference.setZero();
      for(int j=0;j<4;++j)g.reference+=g.basis[j]*geometry.controls[4*g.span+j];
      const double h=m.basis.breaks[g.span+1]-m.basis.breaks[g.span];
      g.node_basis=node==g.span?g.basis[0]+g.basis[1]:g.basis[2]+g.basis[3];
      g.jet_basis=patch.scale*h/3*(node==g.span?g.basis[1]:-g.basis[2]);
      // Keep ALL rows: frozen-jet redundancy proofs cannot be applied after
      // releasing jets. Only construction and exact parallel-plane merging
      // are reused; the retained prefix is deliberately not used here.
      g.reused=&source->supports;groups.push_back(std::move(g));}
    for(auto& g:groups){auto key=[](const auto& p){return std::make_tuple(p.normal.x(),p.normal.y(),p.normal.z(),p.bound);};
      std::sort(g.supports.begin(),g.supports.end(),[&](const auto& a,const auto& b){return key(a)<key(b);});
      g.supports.erase(std::unique(g.supports.begin(),g.supports.end(),[](const auto& a,const auto& b){return a.normal==b.normal;}),g.supports.end());}
    const double offset=2*OptimizerSettings::offset_weight*m.basis.weight(node);
    energy_hessian.topLeftCorner<3,3>().diagonal().array()+=offset;
    refresh(geometry);
  }
  void refresh(const Geometry& geometry){
    value=model.x.segment<6>(6*patch.node);energy_gradient.setZero();
    auto evaluate=[&](int span,const Eigen::Vector4d& weights)->V9 {V9 result=V9::Zero();
      for(int j=0;j<4;++j)result+=weights[j]*geometry.controls[4*span+j];
      return result;};
    for(int span=patch.begin;span<patch.end;++span)patch.curves[span-patch.begin]=geometry.curve(span);
    V9 gn=V9::Zero(),gj=V9::Zero();
    for(const auto& row:energy_rows){const auto& p=patch.curves[row.span-patch.begin];
      const V9 a=6*row.inverse_h2*(p[2]-2*p[1]+p[0]),b=6*row.inverse_h2*(p[3]-2*p[2]+p[1]);
      gn+=row.weight*((2*row.f0+row.f1)*a+(row.f0+2*row.f1)*b);
      gj+=row.weight*((2*row.q0+row.q1)*a+(row.q0+2*row.q1)*b);
    }
    energy_gradient.head<6>()=pull(energy_weights.cwiseProduct(gn));energy_gradient.tail<9>()=energy_weights.cwiseProduct(gj);
    energy_gradient.head<3>()+=2*OptimizerSettings::offset_weight*model.basis.weight(patch.node)*value.head<3>();
    for(auto& g:groups)g.reference=evaluate(g.span,g.basis);
    for(int q=0;q<sample_count;++q){auto& s=samples[q];s.value=evaluate(s.span,s.basis);s.tangent=evaluate(s.span,s.first).head<3>();}
    old_volume=volume(V15::Zero());
  }
  V9 nodeDirection(const State& d)const{
    Eigen::Matrix2d shape;shape<<d[3],d[4],d[4],d[5];return pack(d.head<3>(),frame*shape);
  }
  State pull(const V9& g)const{State r;r.head<3>()=g.head<3>();
    r.tail<3>()<<frame.col(0).dot(g.segment<3>(3)),frame.col(1).dot(g.segment<3>(3))+frame.col(0).dot(g.tail<3>()),frame.col(1).dot(g.tail<3>());return r;}
  double volume(const V15& d,V15* gradient=nullptr)const{
    double result=0;const V9 node=nodeDirection(d.head<6>()),jet=d.tail<9>();V9 gn=V9::Zero(),gj=V9::Zero();
    for(int q=0;q<sample_count;++q){const auto& s=samples[q];const V9 y=s.value+s.node_basis*node+s.jet_basis*jet;
      const auto m=shape(y);const Vec3 v=s.tangent+s.node_first*node.head<3>()+s.jet_first*jet.head<3>(),n=m.col(0).cross(m.col(1));
      result+=s.weight*n.dot(v);
      if(gradient){Mat32 mg;mg.col(0)=m.col(1).cross(v);mg.col(1)=v.cross(m.col(0));const V9 g=pack(Vec3::Zero(),mg);
        gn+=s.weight*s.node_basis*g;gn.head<3>()+=s.weight*s.node_first*n;
        gj+=s.weight*s.jet_basis*g;gj.head<3>()+=s.weight*s.jet_first*n;}}
    if(gradient){gradient->head<6>()=pull(gn);gradient->tail<9>()=gj;}
    return result;
  }
  double objective(const V15& d,double global_volume,double mu,V15* gradient=nullptr,M15* hessian=nullptr)const{
    V15 vg;const double total=global_volume+volume(d,gradient?&vg:nullptr)-old_volume;if(total<=0)return INFINITY;
    double cost=energy_gradient.dot(d)+.5*d.dot(energy_hessian*d)-std::log(total/global_volume);
    if(gradient)*gradient=energy_gradient+energy_hessian*d-vg/total;
    if(hessian){*hessian=energy_hessian;hessian->noalias()+=vg*vg.transpose()/(total*total);}
    Eigen::Matrix2d shape;const State x=value+d.head<6>();shape<<x[3],x[4],x[4],x[5];
    auto logdet=[&](const Eigen::Matrix2d& s,double weight,double sign){const double det=s.determinant();if(s(0,0)<=0||det<=0)return false;
      cost-=weight*std::log(det);if(gradient){const double a=s(1,1)/det,b=-s(0,1)/det,c=s(0,0)/det;
        gradient->segment<3>(3)-=weight*sign*Vec3(a,2*b,c);
        if(hessian){Eigen::Matrix3d h;h<<a*a,2*a*b,b*b,2*a*b,2*(a*c+b*b),2*b*c,b*b,2*b*c,c*c;hessian->block<3,3>(3,3)+=weight*h;}}
      return true;};
    if(!logdet(shape,OptimizerSettings::area_weight*model.basis.weight(patch.node),1)||
       !logdet(shape-options.min_radius*Eigen::Matrix2d::Identity(),mu,1)||
       !logdet(options.max_radius*Eigen::Matrix2d::Identity()-shape,mu,-1))return INFINITY;
    const double gap=options.max_offset*options.max_offset-x.head<3>().squaredNorm();if(gap<=0)return INFINITY;
    cost-=mu*std::log(gap);if(gradient){gradient->head<3>()+=2*mu/gap*x.head<3>();
      if(hessian){hessian->topLeftCorner<3,3>().diagonal().array()+=2*mu/gap;
        hessian->topLeftCorner<3,3>().noalias()+=4*mu/(gap*gap)*x.head<3>()*x.head<3>().transpose();}}
    const V9 node_delta=nodeDirection(d.head<6>());
    for(const auto& group:groups){const V9 y=group.reference+group.node_basis*node_delta+group.jet_basis*d.tail<9>();V9 g=V9::Zero();M9 h=M9::Zero();
      for(const auto& p:group.rows()){const Vec3 v=p.project(y);const double slack=v[0]*v[0]-v.tail<2>().squaredNorm();
        if(v[0]<=0||slack<=0)return INFINITY;
        cost-=mu*std::log(slack);
        if(gradient){const double f=2*mu/slack;
          for(int channel=0;channel<3;++channel)g.segment<3>(3*channel)+=f*v[channel]*p.normal;
          if(hessian){const Eigen::Matrix3d nn=f*p.normal*p.normal.transpose();
            for(int row=0;row<3;++row)for(int col=row;col<3;++col)
              h.block<3,3>(3*row,3*col)+=(2/slack*v[row]*v[col]+(row==col?(row?1.:-1.):0))*nn;}}
      }
      if(gradient){gradient->head<6>()+=group.node_basis*pull(g);gradient->tail<9>()+=group.jet_basis*g;}
      if(hessian){for(int row=1;row<3;++row)for(int col=0;col<row;++col)h.block<3,3>(3*row,3*col)=h.block<3,3>(3*col,3*row).transpose();
        Map hj;hj.leftCols<3>()=h.leftCols<3>();hj.col(3)=h.block<9,3>(0,3)*frame.col(0);
        hj.col(4)=h.block<9,3>(0,3)*frame.col(1)+h.rightCols<3>()*frame.col(0);hj.col(5)=h.rightCols<3>()*frame.col(1);
        Block nn;for(int col=0;col<6;++col)nn.col(col)=pull(hj.col(col));
        hessian->topLeftCorner<6,6>()+=group.node_basis*group.node_basis*nn;
        hessian->topRightCorner<6,9>()+=group.node_basis*group.jet_basis*hj.transpose();
        hessian->bottomLeftCorner<9,6>()+=group.node_basis*group.jet_basis*hj;
        hessian->bottomRightCorner<9,9>()+=group.jet_basis*group.jet_basis*h;}
    }
    return cost;
  }
  c3::Curve<3,V9> curve(int span,const V15& d)const{auto p=patch.curves[span-patch.begin];
    const V9 node=nodeDirection(d.head<6>());
    for(int j=0;j<4;++j){if((span==patch.node&&j<2)||(span!=patch.node&&j>=2))p[j]+=node;
      p[j]+=patch.basis[span-patch.begin][j]*d.tail<9>();}return p;}
  struct Line {
    c3::CubicProblem::LineModel barrier;
    Vec3 volume=Vec3::Zero();
    double total=0;
    double delta(double alpha)const{
      const double dv=alpha*(volume[0]+alpha*(volume[1]+alpha*volume[2]));
      return dv>-total?barrier.delta(alpha)-std::log1p(dv/total):INFINITY;
    }
  };
  Line prepareLine(const V15& d,const V15& step,double global_volume,double mu)const{
    Line line;line.volume.setZero();line.total=global_volume+volume(d)-old_volume;
    line.barrier.linear=step.dot(energy_gradient+energy_hessian*d);line.barrier.quadratic=.5*step.dot(energy_hessian*step);
    double limit=INFINITY;
    auto append=[&](double a,double b,double c,double weight,double value,double derivative,bool constrain=true){
      line.barrier.terms.push_back({a/c,b/c,weight});if(!constrain)return;
      const double alpha=std::min(1.,limit);
      if((a*alpha+b)*alpha+c>0&&value+alpha*derivative>0)return;
      if(derivative<0)limit=std::min(limit,-value/derivative);
      limit=std::min(limit,c3::positiveRoot(a,b,c));};
    auto spd=[&](const Eigen::Matrix2d& s,const Eigen::Matrix2d& v,double weight,bool constrain){
      append(v.determinant(),s(0,0)*v(1,1)+s(1,1)*v(0,0)-2*s(0,1)*v(0,1),s.determinant(),weight,s(0,0),v(0,0),constrain);};
    const State x=value+d.head<6>();Eigen::Matrix2d shape,ds;shape<<x[3],x[4],x[4],x[5];ds<<step[3],step[4],step[4],step[5];
    spd(shape,ds,OptimizerSettings::area_weight*model.basis.weight(patch.node),false);
    spd(shape-options.min_radius*Eigen::Matrix2d::Identity(),ds,mu,true);
    spd(options.max_radius*Eigen::Matrix2d::Identity()-shape,-ds,mu,true);
    append(-step.head<3>().squaredNorm(),-2*x.head<3>().dot(step.head<3>()),options.max_offset*options.max_offset-x.head<3>().squaredNorm(),mu,1,0);
    const V9 node=nodeDirection(d.head<6>()),dn=nodeDirection(step.head<6>());
    for(const auto& g:groups){const V9 y=g.reference+g.node_basis*node+g.jet_basis*d.tail<9>();const V9 dy=g.node_basis*dn+g.jet_basis*step.tail<9>();
      for(const auto& p:g.rows()){const Vec3 v=p.project(y),delta=p.projectDirection(dy);
        append(delta[0]*delta[0]-delta.tail<2>().squaredNorm(),2*(v[0]*delta[0]-v.tail<2>().dot(delta.tail<2>())),v[0]*v[0]-v.tail<2>().squaredNorm(),mu,v[0],delta[0]);}}
    for(int q=0;q<sample_count;++q){const auto& s=samples[q];const Mat32 m=shapeMatrix(s.value+s.node_basis*node+s.jet_basis*d.tail<9>()),dm=shapeMatrix(s.node_basis*dn+s.jet_basis*step.tail<9>());
      const Vec3 tangent=s.tangent+s.node_first*node.head<3>()+s.jet_first*d.segment<3>(6),dt=s.node_first*dn.head<3>()+s.jet_first*step.segment<3>(6);
      const Vec3 normal=m.col(0).cross(m.col(1)),first=dm.col(0).cross(m.col(1))+m.col(0).cross(dm.col(1)),second=dm.col(0).cross(dm.col(1));
      line.volume+=s.weight*Vec3(normal.dot(dt)+first.dot(tangent),first.dot(dt)+second.dot(tangent),second.dot(dt));}
    line.barrier.step=std::min(1.,.99*limit);return line;
  }
  static Mat32 shapeMatrix(const V9& value){return shape(value);}
};
inline void optimizeJointJets(Model& m,const std::vector<internal::detail::SupportPlane>& cover,double length,
    const GeometryOptions& options,const OptimizerSettings& settings,JetMetrics& metrics,const Deadline& deadline,
    const Problem::Constraints* shared=nullptr){
  StageTimer timer(settings.profile);auto geometry=m.geometry();double volume=sweptVolume(m,m.x);
  std::vector<std::vector<int>> planes(geometry.pieces());for(int k=0;k<int(cover.size());++k)planes[cover[k].span].push_back(k);
  std::array<Anchor,2> anchors;for(int end=0;end<2;++end)anchors[end].u=m.anchor_parameters[end];anchorObjective(m,m.x,anchors,1.);
  const double mu=1e-5/m.basis.count;
  std::vector<JointPatch> patches;patches.reserve(m.basis.count);
  StageTimer phase(settings.profile);
  std::vector<std::vector<const NodalGroup*>> node_groups(m.basis.count);
  if(shared)for(const auto& g:shared->groups)for(int k=0;k<g.count;++k)node_groups[g.first/6+k].push_back(&g);
  for(int node=0;node<m.basis.count;++node)patches.emplace_back(m,geometry,node,length,options,settings.volume_jet_energy,cover,planes,shared?&node_groups[node]:nullptr);
  metrics.joint_setup_ms+=phase.elapsed();
  for(const auto& p:patches){metrics.joint_groups+=p.groups.size();for(const auto& g:p.groups)metrics.joint_supports+=g.rows().size();}
  for(int sweep=0;sweep<settings.joint_sweeps;++sweep)for(int index=0;index<m.basis.count;++index){deadline.check();
    const int node=sweep%2?m.basis.count-1-index:index;
    auto& patch=patches[node];phase.restart();patch.refresh(geometry);metrics.joint_refresh_ms+=phase.elapsed();
    const V9 old=m.jets[node];V15 d=V15::Zero();bool changed=false;
    const bool touches_anchor=(anchors[0].span>=patch.patch.begin&&anchors[0].span<patch.patch.end)||
                             (anchors[1].span>=patch.patch.begin&&anchors[1].span<patch.patch.end);
    for(int iteration=0;iteration<settings.joint_jet_steps;++iteration){V15 gradient;M15 hessian;
      phase.restart();
      const double cost=patch.objective(d,volume,mu,&gradient,&hessian);if(!std::isfinite(cost))break;
      metrics.joint_assembly_ms+=phase.elapsed();phase.restart();
      const V15 scale=hessian.diagonal().cwiseMax(1e-20).cwiseSqrt().cwiseInverse();
      M15 balanced=scale.asDiagonal()*hessian*scale.asDiagonal();balanced.diagonal().array()+=1e-8;
      Eigen::LLT<M15> factor(balanced);if(factor.info()!=Eigen::Success)break;
      auto solve=[&](const V15& rhs)->V15{return scale.cwiseProduct(factor.solve(scale.cwiseProduct(rhs)));};
      V15 step=solve(-gradient);
      if(node==0||node+1==m.basis.count){const Vec3 n=inwardCapNormal(m,node!=0);
        const double rhs=n.dot(patch.value.head<3>()+d.head<3>()+step.head<3>());
        if(rhs<0){V15 normal=V15::Zero();normal.head<3>()=n;const V15 inverse=solve(normal);step-=rhs/(normal.dot(inverse))*inverse;}}
      const double decrease=gradient.dot(step);if(decrease>=-1e-10)break;bool accepted=false;
      metrics.joint_solve_ms+=phase.elapsed();phase.restart();
      JointPatch::Line line;if(settings.joint_fast_line)line=patch.prepareLine(d,step,volume,mu);
      const double initial_step=settings.joint_fast_line?line.barrier.step:1.;
      for(double alpha=initial_step;alpha>=1e-10;alpha*=.5){const V15 next=d+alpha*step;
        metrics.joint_backtracks+=alpha<initial_step;
        const double delta=settings.joint_fast_line?line.delta(alpha):patch.objective(next,volume,mu)-cost;
        if(delta>1e-4*alpha*decrease)continue;
        bool regular=true;for(int span=patch.patch.begin;span<patch.patch.end;++span){c3::Certificate certificate;
          if(!c3::jacobianCertificate<3>(patch.curve(span,next),certificate)){regular=false;break;}}
        if(!regular)continue;
        m.jets[node]=old+patch.patch.scale*next.tail<9>();m.x.segment<6>(6*node)=patch.value+next.head<6>();auto trial=anchors;
        if(touches_anchor&&!std::isfinite(anchorObjective(m,m.x,trial,1.))){m.jets[node]=old+patch.patch.scale*d.tail<9>();m.x.segment<6>(6*node)=patch.value+d.head<6>();continue;}
        anchors=trial;d=next;accepted=changed=true;break;
      }
      metrics.joint_line_ms+=phase.elapsed();metrics.joint_steps+=accepted;
      if(!accepted)break;
    }
    if(changed){volume+=patch.volume(d)-patch.old_volume;for(int span=patch.patch.begin;span<patch.patch.end;++span){const auto p=patch.curve(span,d);
        std::copy(p.begin(),p.end(),geometry.controls.begin()+4*span);}}
  }
  for(int end=0;end<2;++end)m.anchor_parameters[end]=anchors[end].u;
  m.cacheQuadrature();metrics.milliseconds+=timer.elapsed();
}
}

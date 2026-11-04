// SPDX-License-Identifier: MIT
#include "cubic_optimizer.hpp"
#include "cubic_certificates.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <Eigen/QR>
#include <chrono>
#include <map>
#include <gcopter/quickhull.hpp>
namespace gcopter::direct::native {
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static Eigen::Matrix2d matrix(const Eigen::VectorXd& x,int i) {
  Eigen::Matrix2d s;s<<x[5*i+2],x[5*i+3],x[5*i+3],x[5*i+4];return s;
}
Eigen::Matrix<double,9,5> CubicModel::jacobian(int i)const {
  Eigen::Matrix<double,9,5> j=Eigen::Matrix<double,9,5>::Zero();
  j.topLeftCorner<3,2>()=frames[i];
  j.block<3,1>(3,2)=frames[i].col(0);j.block<3,1>(3,3)=frames[i].col(1);
  j.block<3,1>(6,3)=frames[i].col(0);j.block<3,1>(6,4)=frames[i].col(1);return j;
}
Geometry CubicModel::geometry(const Eigen::VectorXd& values)const {
  std::vector<V9> controls(basis.count);
  for(int i=0;i<basis.count;++i)controls[i]=pack(reference[i]+frames[i]*values.segment<2>(5*i),frames[i]*matrix(values,i));
  return Geometry::fromCubic(basis,controls);
}
CubicModel CubicModel::project(const Geometry& natural) {
  CubicModel out(natural.breaks);const int count=out.basis.count;
  std::vector<Eigen::Triplet<double>> entries;
  Eigen::Matrix<double,Eigen::Dynamic,9> rhs=Eigen::Matrix<double,Eigen::Dynamic,9>::Zero(count,9);
  for(int i=0;i<int(natural.pieces());++i) {
    const auto& e=out.basis.spans[i].extraction;
    for(int j=0;j<4;++j)for(int k=0;k<4;++k) {
      rhs.row(i+k)+=e(j,k)*natural.controls[4*i+j].transpose();
      for(int l=0;l<4;++l)entries.emplace_back(i+k,i+l,e(j,k)*e(j,l));
    }
  }
  Eigen::SparseMatrix<double> a(count,count);a.setFromTriplets(entries.begin(),entries.end());
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(a);
  const Eigen::Matrix<double,Eigen::Dynamic,9> values=solver.solve(rhs);
  out.x.setZero();
  for(int i=0;i<count;++i) {
    const V9 y=values.row(i);const Mat32 m=shape(y);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> e(m.transpose()*m);
    if(e.eigenvalues()[0]<1e-14)throw std::runtime_error("rank-deficient projected cubic control");
    const Eigen::Matrix2d s=e.eigenvectors()*e.eigenvalues().cwiseSqrt().asDiagonal()*e.eigenvectors().transpose();
    out.reference[i]=y.head<3>();out.frames[i]=m*s.inverse();
    out.x.segment<3>(5*i+2)<<s(0,0),s(0,1),s(1,1);
  }
  return out;
}
CubicModel CubicModel::fromSeed(const std::vector<Vec3>& seed,double spacing,double radius,double smoothing) {
  std::vector<double> arc(seed.size(),0);std::vector<Vec3> moment(seed.size(),Vec3::Zero());
  for(size_t i=1;i<seed.size();++i){arc[i]=arc[i-1]+(seed[i]-seed[i-1]).norm();moment[i]=moment[i-1]+.5*(arc[i]+arc[i-1])*(seed[i]-seed[i-1]);}
  const double length=arc.back();const int segments=std::max(1,int(std::ceil(length/spacing)));
  std::vector<double> breaks;for(int i=0;i<=segments;++i)breaks.push_back(double(i)/segments);
  CubicModel out(breaks);out.x.setZero();
  auto segment=[&](double s){return std::clamp(int(std::upper_bound(arc.begin(),arc.end(),s)-arc.begin())-1,0,int(seed.size())-2);};
  auto integral=[&](double s) {
    s=std::clamp(s,0.,length);const int k=segment(s);
    const Vec3 displacement=(s-arc[k])/(arc[k+1]-arc[k])*(seed[k+1]-seed[k]);
    return std::make_pair(Vec3(seed[k]-seed[0]+displacement),Vec3(moment[k]+.5*(s+arc[k])*displacement));
  };
  Vec3 previous=Vec3::Zero(),axis=Vec3::Zero();
  for(int i=0;i<out.basis.count;++i) {
    const double s=length*out.basis.greville(i);const int k=segment(s);
    out.reference[i]=seed[k]+(s-arc[k])/(arc[k+1]-arc[k])*(seed[k+1]-seed[k]);
    Vec3 tangent=(seed[k+1]-seed[k]).normalized();
    if(smoothing>0) {
      const auto a=integral(s-smoothing),b=integral(s),c=integral(s+smoothing);
      const Vec3 average=(1-s/smoothing)*(b.first-a.first)+(b.second-a.second)/smoothing+
                         (1+s/smoothing)*(c.first-b.first)-(c.second-b.second)/smoothing;
      if(average.dot(tangent)>.15*average.norm())tangent=average.normalized();
    }
    if(!i)axis=tangent.unitOrthogonal();else {axis=Eigen::Quaterniond::FromTwoVectors(previous,tangent)*axis;axis-=tangent*tangent.dot(axis);axis.normalize();}
    out.frames[i].col(0)=axis;out.frames[i].col(1)=tangent.cross(axis);previous=tangent;
    out.x[5*i+2]=out.x[5*i+4]=radius;
  }
  return out;
}
Eigen::Matrix<double,3,20> CubicProblem::Cut::dense()const {
  Eigen::Matrix<double,3,20> a=Eigen::Matrix<double,3,20>::Zero();
  for(int k=0;k<count;++k) {
    const auto& n=normals[k];
    a.block<1,2>(0,5*k)=center_sign*n.transpose();
    a(1,5*k+2)=n[0];a(1,5*k+3)=n[1];a(2,5*k+3)=n[0];a(2,5*k+4)=n[1];
  }
  return a;
}
Vec3 CubicProblem::Cut::apply(const Eigen::VectorXd& x)const {
  Vec3 v=Vec3::Zero();
  for(int k=0;k<count;++k) {
    const double* p=x.data()+first+5*k;const auto& n=normals[k];
    v[0]+=center_sign*(n[0]*p[0]+n[1]*p[1]);
    v[1]+=n[0]*p[2]+n[1]*p[3];v[2]+=n[0]*p[3]+n[1]*p[4];
  }
  return v;
}
bool CubicProblem::Cut::compact() {
  double norm=0;for(int k=0;k<count;++k)norm+=3*normals[k].squaredNorm();
  if(norm<1e-25)return false;
  while(count && normals[count-1].isZero(0))--count;
  int start=0;while(start<count && normals[start].isZero(0))++start;
  for(int k=start;k<count;++k)normals[k-start]=normals[k];
  first+=5*start;count-=start;return count>0;
}
void CubicProblem::Cut::accumulate(const Vec3& v,double mu,double slack,
    Eigen::VectorXd& gradient,BandedSPD& hessian)const {
  // J has only two nonzeros per row and control. Assemble the rank-one
  // barrier curvature directly into contiguous lower-band rows, then add
  // three sparse Gram updates. No 20x20 temporary and no zero outer products.
  Eigen::Matrix<double,20,1> reverse;
  const int size=5*count;const double first_factor=mu/slack;
  for(int k=0;k<count;++k) {
    const auto& n=normals[k];const int j=5*k;
    reverse[size-1-j]=-2*center_sign*v[0]*n[0];
    reverse[size-2-j]=-2*center_sign*v[0]*n[1];
    reverse[size-3-j]=-2*v[1]*n[0];
    reverse[size-4-j]=-2*(v[1]*n[1]+v[2]*n[0]);
    reverse[size-5-j]=-2*v[2]*n[1];
  }
  for(int j=0;j<size;++j) {
    const double d=reverse[size-1-j];gradient[first+j]-=first_factor*d;
    hessian.data.row(first+j).head(j+1).noalias()+=
        (first_factor/slack*d)*reverse.segment(size-1-j,j+1).transpose();
  }
  // Each Gram row has the same n_x,n_y sequence, at these variable slots.
  for(int row=0;row<3;++row)for(int j=0;j<2*count;++j) {
    const int col=5*(j/2)+(row==0?j%2:row+1+j%2);
    const double nj=2*first_factor*(row==0?-1:1)*normals[j/2][j%2];
    for(int k=0;k<=j;++k) {
      const int other=5*(k/2)+(row==0?k%2:row+1+k%2);
      hessian.data(first+col,col-other)+=nj*normals[k/2][k%2];
    }
  }
}
CubicProblem::CubicProblem(const CubicModel& m,const std::vector<detail::SupportPlane>& cover,
                           double length,const Options& o,const OptimizerSettings& s)
    :model(m),options(o),settings(s),quadratic(5*m.basis.count,19),linear(Eigen::VectorXd::Zero(5*m.basis.count)) {
  // On the same parameter interval, equal normals only need the tightest
  // bound. World/local-box duplicates otherwise add both work and barrier bias.
  std::map<std::tuple<int,double,double,double,double,double>,detail::SupportPlane> unique;
  for(const auto& p:cover) {
    const auto key=std::make_tuple(p.span,p.begin,p.end,p.normal[0],p.normal[1],p.normal[2]);
    auto found=unique.find(key);
    if(found==unique.end()||p.bound<found->second.bound)unique[key]=p;
  }
  std::map<std::tuple<int,double,double,int>,size_t> group_ids;
  for(const auto& entry:unique) {
    const auto& plane=entry.second;
    const auto& span=m.basis.spans[plane.span];
    Curve<3,Eigen::Vector4d> basis;
    for(int j=0;j<4;++j)basis[j]=span.extraction.row(j);
    basis=restrictCurve<3>(basis,plane.begin,plane.end);
    for(int j=0;j<4;++j) {
      Cut c;c.first=5*span.ids[0];c.bound=plane.bound;
      for(int k=0;k<4;++k) {
        const int id=span.ids[k];const Eigen::Vector2d n=basis[j][k]*m.frames[id].transpose()*plane.normal;
        c.bound-=basis[j][k]*m.reference[id].dot(plane.normal);
        c.normals[k]=n;
      }
      // Skip constraints which are identically constant (e.g. a straight end cap).
      if(!c.compact()){if(c.bound < -1e-9)throw std::runtime_error("inconsistent constant support plane");continue;}
      cuts.push_back(c);
      const auto group_key=std::make_tuple(plane.span,plane.begin,plane.end,j);
      auto inserted=group_ids.emplace(group_key,groups.size());
      if(inserted.second) {
        Group group;group.first=5*span.ids[0];group.count=4;
        group.reference.setZero();group.map.setZero();
        for(int k=0;k<4;++k) {
          group.reference.head<3>()+=basis[j][k]*m.reference[span.ids[k]];
          group.map.middleCols<5>(5*k)=basis[j][k]*m.jacobian(span.ids[k]);
        }
        while(group.count && basis[j][group.count-1]==0)--group.count;
        int start=0;while(start<group.count && basis[j][start]==0)++start;
        group.first+=5*start;group.count-=start;
        if(start)group.map.leftCols(5*group.count)=group.map.middleCols(5*start,5*group.count).eval();
        group.map.rightCols(20-5*group.count).setZero();
        groups.push_back(std::move(group));
      }
      groups[inserted.first->second].supports.push_back({plane.normal,plane.bound,!settings.working_set});
    }
  }
  // Polar hull pruning is independent of ellipse shape: a strictly interior
  // dual point is implied by the remaining spatial halfspaces. Keep boundary
  // and degenerate cases. All original cuts still bound the line search and
  // remain in the final continuous certificate.
  if(settings.polar_pruning)for(auto& group:groups) {
    const int count=group.supports.size();if(count<5)continue;
    const int n=5*group.count;
    const V9 y=group.reference+group.map.leftCols(n)*m.x.segment(group.first,n);
    Eigen::Matrix<double,3,Eigen::Dynamic> points(3,count+1);points.col(count).setZero();
    bool interior=true;
    for(int j=0;j<count;++j) {
      const auto& p=group.supports[j];const double gap=p.bound-p.normal.dot(y.head<3>());
      if(gap<=1e-12){interior=false;break;}
      points.col(j)=p.normal/gap;
    }
    if(!interior)continue;
    const Vec3 mean=points.rowwise().mean();
    const Eigen::Matrix3d covariance=points*points.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(covariance);
    if(eig.eigenvalues()[0]<1e-20*eig.eigenvalues()[2])continue;
    quickhull::QuickHull<double> hull;
    const auto result=hull.getConvexHull(points.data(),points.cols(),true,true,1e-12);
    const auto& indices=result.getIndexBuffer();
    std::vector<std::pair<Vec3,double>> faces;
    for(size_t f=0;f+2<indices.size();f+=3) {
      if(indices[f]>=size_t(count+1)||indices[f+1]>=size_t(count+1)||indices[f+2]>=size_t(count+1)){faces.clear();break;}
      const Vec3 a=points.col(indices[f]),b=points.col(indices[f+1]),c=points.col(indices[f+2]);
      Vec3 normal=(b-a).cross(c-a);const double norm=normal.norm();if(norm<1e-15)continue;
      normal/=norm;if(normal.dot(mean-a)>0)normal=-normal;
      faces.emplace_back(normal,normal.dot(a));
    }
    if(faces.size()<4)continue;
    for(int j=0;j<count;++j) {
      bool inside=true;const double guard=1e-10*(1+points.col(j).norm());
      for(const auto& face:faces)if(face.second-face.first.dot(points.col(j))<=guard){inside=false;break;}
      if(inside){group.supports[j].active=false;group.supports[j].redundant=true;}
    }
  }
  geometric_cuts=cuts.size();
  if(settings.working_set)activateSupports(m.x);
  // Local convex progress cones are added only after a full-disk failure.
  // They use fixed normals and a degree-two derivative hull. The nonlinear
  // full-disk certificate remains the acceptance test after every repair.
  const auto initial=m.geometry();
  for(int span:settings.progress_spans)for(double t:{0.,.5,1.}) {
    const double u=m.basis.breaks[span]+t*(m.basis.breaks[span+1]-m.basis.breaks[span]);
    const auto section=initial.section(u);
    const Vec3 normal=section.shape.col(0).cross(section.shape.col(1)).normalized();
    const auto& e=m.basis.spans[span].extraction;
    Curve<2,Eigen::Vector4d> derivative;
    for(int j=0;j<3;++j)derivative[j]=3*(e.row(j+1)-e.row(j));
    const auto halves=split<2>(derivative);
    for(const auto& part:{halves.first,halves.second})for(const auto& weight:part) {
      Cut c;c.first=5*span;c.center_sign=-1;c.bound=-1e-5*length*(m.basis.breaks[span+1]-m.basis.breaks[span]);
      for(int k=0;k<4;++k) {
        const Eigen::Vector2d v=weight[k]*m.frames[span+k].transpose()*normal;
        c.bound+=weight[k]*m.reference[span+k].dot(normal);
        c.normals[k]=v;
      }
      if(!c.compact())continue;
      const Vec3 value=c.apply(m.x);
      if(c.bound-value[0]>value.tail<2>().norm()+1e-9)cuts.push_back(c);
    }
  }
  // Exact two-point Gauss integration of squared cubic second derivatives.
  for(int i=0;i<int(m.basis.spans.size());++i) {
    const double h=m.basis.breaks[i+1]-m.basis.breaks[i];
    for(double t:{.5-.5/std::sqrt(3.),.5+.5/std::sqrt(3.)}) {
      const auto basis=m.basis.jet(i,m.basis.breaks[i]+h*t);
      Eigen::Matrix<double,9,20> map;V9 reference=V9::Zero();
      for(int k=0;k<4;++k) {
        const double d=basis(2,k)/(length*length);
        map.middleCols<5>(5*k)=d*m.jacobian(i+k);reference.head<3>()+=d*m.reference[i+k];
      }
      const double wc=std::sqrt(settings.center_energy*.5*h*length),wm=std::sqrt(settings.shape_energy*.5*h*length);
      map.topRows<3>()*=wc;map.bottomRows<6>()*=wm;reference.head<3>()*=wc;
      const Eigen::Matrix<double,20,20> gram=2*map.transpose()*map;
      linear.segment<20>(5*i)+=2*map.transpose()*reference;
      for(int a=0;a<20;++a)for(int b=0;b<=a;++b)quadratic.add(5*i+a,5*i+b,gram(a,b));
    }
  }
  for(int i=0;i<m.basis.count;++i)for(int k=0;k<2;++k)quadratic.add(5*i+k,5*i+k,2*settings.offset_weight*m.basis.weight(i));
  if(settings.natural_endpoints) {
    Eigen::MatrixXd a=Eigen::MatrixXd::Zero(18,5*m.basis.count);
    for(int end=0;end<2;++end) {
      const int span=end?int(m.basis.spans.size())-1:0;
      const auto b=m.basis.jet(span,end?1.:0.);
      for(int k=0;k<4;++k)a.block<9,5>(9*end,5*(span+k))=b(2,k)*m.jacobian(span+k)/(length*length);
    }
    for(int id:{0,1,5*(m.basis.count-1),5*(m.basis.count-1)+1})a.col(id).setZero();
    for(int i=0;i<a.rows();++i){const double n=a.row(i).norm();if(n>0)a.row(i)/=n;}
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(a.transpose());qr.setThreshold(1e-9);
    equalities.resize(qr.rank(),a.cols());
    for(int i=0;i<qr.rank();++i)equalities.row(i)=a.row(qr.colsPermutation().indices()[i]);
  }
}
double CubicProblem::objective(const Eigen::VectorXd& x,double mu,Eigen::VectorXd* g,BandedSPD* h)const {
  const bool derivatives=g;
  const auto qx=quadratic.multiply(x);
  double cost=.5*x.dot(qx)+linear.dot(x);
  if(derivatives){*g=qx+linear;*h=quadratic;}
  auto logdet=[&](int i,const Eigen::Matrix2d& p,double weight,double sign) {
    const double det=p.determinant();if(p(0,0)<=0 || det<=0)return false;
    cost-=weight*std::log(det);
    if(derivatives) {
      const double a=p(1,1)/det,b=-p(0,1)/det,c=p(0,0)/det;
      g->segment<3>(5*i+2)-=weight*sign*Vec3(a,2*b,c);
      Eigen::Matrix3d v;v<<a*a,2*a*b,b*b,2*a*b,2*(a*c+b*b),2*b*c,b*b,2*b*c,c*c;
      for(int j=0;j<3;++j)for(int k=0;k<=j;++k)h->add(5*i+2+j,5*i+2+k,weight*v(j,k));
    }
    return true;
  };
  for(int i=0;i<model.basis.count;++i) {
    const auto s=matrix(x,i);
    if(!logdet(i,s,model.basis.weight(i),1) ||
       !logdet(i,s-options.min_radius*Eigen::Matrix2d::Identity(),mu,1) ||
       !logdet(i,options.max_radius*Eigen::Matrix2d::Identity()-s,mu,-1))return INFINITY;
    const Eigen::Vector2d d=x.segment<2>(5*i);const double gap=options.max_offset*options.max_offset-d.squaredNorm();
    if(gap<=0)return INFINITY;
    cost-=mu*std::log(gap);
    if(derivatives) {
      g->segment<2>(5*i)+=2*mu/gap*d;
      const Eigen::Matrix2d v=2*mu/gap*Eigen::Matrix2d::Identity()+4*mu/(gap*gap)*d*d.transpose();
      for(int j=0;j<2;++j)for(int k=0;k<=j;++k)h->add(5*i+j,5*i+k,v(j,k));
    }
  }
  if(settings.grouped_assembly && !settings.reference_assembly)for(const auto& group:groups) {
    const int n=5*group.count;
    const V9 y=group.reference+group.map.leftCols(n)*x.segment(group.first,n);
    V9 gy=V9::Zero();Eigen::Matrix<double,9,9> hy=Eigen::Matrix<double,9,9>::Zero();
    for(const auto& support:group.supports) {
      if(!support.active)continue;
      const auto& normal=support.normal;
      const Vec3 v(support.bound-normal.dot(y.head<3>()),normal.dot(y.segment<3>(3)),normal.dot(y.tail<3>()));
      const double slack=v[0]*v[0]-v.tail<2>().squaredNorm();
      if(v[0]<=0 || slack<=0)return INFINITY;
      cost-=mu*std::log(slack);
      if(derivatives) {
        const double f=2*mu/slack;
        for(int a=0;a<3;++a)gy.segment<3>(3*a)+=(f*v[a])*normal;
        Eigen::Matrix3d hc=(2*f/slack)*v*v.transpose();
        hc(0,0)-=f;hc(1,1)+=f;hc(2,2)+=f;
        const Eigen::Matrix3d nn=normal*normal.transpose();
        for(int a=0;a<3;++a)for(int b=0;b<=a;++b)hy.block<3,3>(3*a,3*b).noalias()+=hc(a,b)*nn;
      }
    }
    if(derivatives) {
      for(int a=0;a<3;++a)for(int b=0;b<a;++b)hy.block<3,3>(3*b,3*a)=hy.block<3,3>(3*a,3*b).transpose();
      g->segment(group.first,n).noalias()+=group.map.leftCols(n).transpose()*gy;
      const Eigen::Matrix<double,9,20> product=hy*group.map;
      const Eigen::Matrix<double,20,20> local=group.map.transpose()*product;
      for(int j=0;j<n;++j)for(int k=0;k<=j;++k)h->data(group.first+j,j-k)+=local(j,k);
    }
  }
  const size_t begin=settings.grouped_assembly && !settings.reference_assembly?geometric_cuts:0;
  for(size_t id=begin;id<cuts.size();++id) {
    const auto& c=cuts[id];Eigen::Vector3d v=c.apply(x);v[0]=c.bound-v[0];
    const double slack=v[0]*v[0]-v.tail<2>().squaredNorm();
    if(v[0]<=0 || slack<=0)return INFINITY;
    cost-=mu*std::log(slack);
    if(derivatives && !settings.reference_assembly) c.accumulate(v,mu,slack,*g,*h);
    else if(derivatives) {
      const auto map=c.dense();
      const int n=5*c.count;
      const Eigen::Matrix<double,20,1> ds=-2*map.transpose()*v;
      g->segment(c.first,n)-=mu/slack*ds.head(n);
      const double first=mu/slack,second=first/slack;
      Eigen::Matrix<double,20,20> local=second*ds*ds.transpose();
      local.noalias()+=2*first*map.bottomRows<2>().transpose()*map.bottomRows<2>();
      local.noalias()-=2*first*map.row(0).transpose()*map.row(0);
      for(int j=0;j<n;++j)for(int k=0;k<=j;++k)h->add(c.first+j,c.first+k,local(j,k));
    }
  }
  return cost;
}
void CubicProblem::checkDerivatives(const Eigen::VectorXd& x,OptimizerMetrics& metrics)const {
  const double mu=.003/model.basis.count;
  Eigen::VectorXd g;BandedSPD h;const double f=objective(x,mu,&g,&h);(void)f;
  for(int trial=0;trial<8;++trial) {
    Eigen::VectorXd v(x.size());for(int j=0;j<v.size();++j)v[j]=std::sin((j+1)*(trial+1)*.381);
    v.normalize();double step=1e-6;
    while(step>1e-12 && (!std::isfinite(objective(x+step*v,mu))||!std::isfinite(objective(x-step*v,mu))))step*=.1;
    const double fd=(objective(x+step*v,mu)-objective(x-step*v,mu))/(2*step);
    metrics.gradient_error=std::max(metrics.gradient_error,std::abs(fd-g.dot(v))/std::max({1.,std::abs(fd),std::abs(g.dot(v))}));
    Eigen::VectorXd gp,gm;BandedSPD scratch;
    objective(x+step*v,mu,&gp,&scratch);objective(x-step*v,mu,&gm,&scratch);
    const Eigen::VectorXd hv=h.multiply(v),num=(gp-gm)/(2*step);
    metrics.hessian_error=std::max(metrics.hessian_error,(num-hv).norm()/std::max({1.,num.norm(),hv.norm()}));
  }
}
int CubicProblem::activateSupports(const Eigen::VectorXd& x)const {
  int active=0;
  for(const auto& group:groups) {
    const int n=5*group.count;
    const V9 y=group.reference+group.map.leftCols(n)*x.segment(group.first,n);
    for(const auto& support:group.supports) {
      if(!support.active && !support.redundant) {
        const double radius=std::hypot(support.normal.dot(y.segment<3>(3)),support.normal.dot(y.tail<3>()));
        support.active=support.bound-support.normal.dot(y.head<3>())<2*radius+1e-6;
      }
      active+=support.active;
    }
  }
  return active+cuts.size()-geometric_cuts;
}
double CubicProblem::feasibleStep(const Eigen::VectorXd& x,const Eigen::VectorXd& step)const {
  double limit=INFINITY;
  auto quadraticLimit=[&](double a,double b,double c) {
    if(a==0){if(b<0)limit=std::min(limit,-c/b);return;}
    const double discriminant=b*b-4*a*c;
    if(discriminant<=0)return;
    const double q=-.5*(b+std::copysign(std::sqrt(discriminant),b));
    const double r0=q/a,r1=c/q;
    if(r0>0)limit=std::min(limit,r0);
    if(r1>0)limit=std::min(limit,r1);
  };
  auto cone=[&](const Vec3& v,const Vec3& d) {
    const Vec3 trial=v+std::min(1.,limit)*d;
    if(trial[0]>0 && trial[0]*trial[0]>trial.tail<2>().squaredNorm())return;
    if(d[0]<0)limit=std::min(limit,-v[0]/d[0]);
    quadraticLimit(d[0]*d[0]-d.tail<2>().squaredNorm(),2*(v[0]*d[0]-v.tail<2>().dot(d.tail<2>())),v[0]*v[0]-v.tail<2>().squaredNorm());
  };
  auto spd=[&](const Eigen::Matrix2d& a,const Eigen::Matrix2d& d) {
    const Eigen::Matrix2d trial=a+std::min(1.,limit)*d;
    if(trial(0,0)>0 && trial.determinant()>0)return;
    if(d(0,0)<0)limit=std::min(limit,-a(0,0)/d(0,0));
    quadraticLimit(d.determinant(),a(0,0)*d(1,1)+a(1,1)*d(0,0)-2*a(0,1)*d(0,1),a.determinant());
  };
  for(int i=0;i<model.basis.count;++i) {
    const auto s=matrix(x,i),d=matrix(step,i);
    spd(s-options.min_radius*Eigen::Matrix2d::Identity(),d);
    spd(options.max_radius*Eigen::Matrix2d::Identity()-s,-d);
    const auto c=x.segment<2>(5*i),v=step.segment<2>(5*i);
    if((c+std::min(1.,limit)*v).squaredNorm()>=options.max_offset*options.max_offset)
      quadraticLimit(-v.squaredNorm(),-2*c.dot(v),options.max_offset*options.max_offset-c.squaredNorm());
  }
  for(const auto& group:groups) {
    const int n=5*group.count;
    const V9 y=group.reference+group.map.leftCols(n)*x.segment(group.first,n);
    const V9 d=group.map.leftCols(n)*step.segment(group.first,n);
    for(const auto& support:group.supports) {
      const auto& normal=support.normal;
      cone(Vec3(support.bound-normal.dot(y.head<3>()),normal.dot(y.segment<3>(3)),normal.dot(y.tail<3>())),
           Vec3(-normal.dot(d.head<3>()),normal.dot(d.segment<3>(3)),normal.dot(d.tail<3>())));
    }
  }
  for(size_t j=geometric_cuts;j<cuts.size();++j){const auto& c=cuts[j];auto v=c.apply(x),d=c.apply(step);v[0]=c.bound-v[0];d[0]=-d[0];cone(v,d);}
  return std::min(1.,(settings.fraction_boundary>0?settings.fraction_boundary:.99)*limit);
}
void CubicProblem::optimize(Eigen::VectorXd& x,OptimizerMetrics& metrics,const std::function<void()>& stop)const {
  metrics.controls=model.basis.count;metrics.variables=x.size()-4;metrics.constraints=cuts.size();metrics.equality_rank=equalities.rows();
  metrics.hessian_nonzero=x.size()*20-190;
  if(settings.check_gradients)checkDerivatives(x,metrics);
  const std::vector<double> barriers=settings.barrier_stages==1?std::vector<double>{.00001}:
      settings.barrier_stages==2?std::vector<double>{.003,.00001}:std::vector<double>{.003,.0001,.00001};
  int stage=0;
  for(double barrier:barriers) {
    const double mu=barrier/model.basis.count;
    bool converged=false;
    const int limit=stage++>0 && settings.corrector_iterations>0?settings.corrector_iterations:settings.iterations;
    for(int it=0;it<limit;++it) {
      stop();auto timer=Clock::now();
      metrics.active_constraints=settings.working_set?activateSupports(x):0;
      if(!settings.working_set){for(const auto& group:groups)for(const auto& p:group.supports)metrics.active_constraints+=p.active;metrics.active_constraints+=cuts.size()-geometric_cuts;}
      Eigen::VectorXd g;BandedSPD h;
      const double f=objective(x,mu,&g,&h);metrics.assembly_ms+=ms(timer);
      if(!std::isfinite(f))throw std::runtime_error("cubic optimizer initial point is infeasible");
      for(int id:{0,1,int(x.size())-5,int(x.size())-4}){g[id]=0;h.fix(id);}
      timer=Clock::now();double damping=0;
      while(!h.factor(damping)){damping=damping==0?1e-8:damping*10;++metrics.damping_retries;if(damping>1e6)throw std::runtime_error("cubic band factorization failed");}
      Eigen::VectorXd step=h.solve(-g);
      if(equalities.rows()) {
        Eigen::MatrixXd solve(x.size(),equalities.rows());
        for(int j=0;j<equalities.rows();++j)solve.col(j)=h.solve(equalities.row(j).transpose());
        const Eigen::MatrixXd schur=equalities*solve;
        step-=solve*schur.ldlt().solve(equalities*step);
      }
      metrics.solve_ms+=ms(timer);
      const double decrement=-g.dot(step);metrics.final_decrement=decrement;
      if(decrement<settings.gradient_tolerance){converged=true;break;}
      timer=Clock::now();double alpha=settings.fraction_boundary>0||settings.working_set?feasibleStep(x,step):1;Eigen::VectorXd next;
      for(;;) {
        next=x+alpha*step;
        if(objective(next,mu)<=f-1e-4*alpha*decrement)break;
        alpha*=.5;++metrics.backtracks;
        if(alpha<1e-12){metrics.termination="line_search_stalled";return;}
      }
      metrics.line_search_ms+=ms(timer);
      x=std::move(next);++metrics.steps;
    }
    metrics.termination=converged?"newton_decrement":"iteration_limit";
  }
}
}

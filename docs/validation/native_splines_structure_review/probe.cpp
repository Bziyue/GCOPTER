#include "cubic_generator.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
using namespace gcopter::direct;
namespace ns=gcopter::direct::native;
static std::vector<Vec3> read(const char* path) {
  std::ifstream in(path);std::string line;std::vector<Vec3> values;
  while(std::getline(in,line)){if(line.empty()||line[0]=='#')continue;std::istringstream row(line);Vec3 x;if(row>>x[0]>>x[1]>>x[2])values.push_back(x);}
  return values;
}
// A diagnostic sufficient redundancy proof, evaluated in the original 3-D
// model. The 2-D projection proposes witnesses; it is never the acceptance test.
static size_t projectedProofs(const ns::CubicProblem::Group& group,const Eigen::VectorXd& x,const Options& o) {
  struct P {Eigen::Vector2d q;Vec3 world;int id;};
  Eigen::Matrix3d gram=Eigen::Matrix3d::Zero();for(int k=0;k<group.count;++k)gram.noalias()+=group.maps[k]*group.maps[k].transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(gram);
  const Eigen::Matrix<double,3,2> u=eigen.eigenvectors().rightCols<2>();
  const Vec3 center=group.geometry(x).head<3>();std::vector<P> points;
  for(int j=0;j<int(group.supports.size());++j) {
    const auto& p=group.supports[j];const double gap=p.bound-p.normal.dot(center);
    if(gap<=0)return 0;const Vec3 q=p.normal/gap;points.push_back({u.transpose()*q,q,j});
  }
  const auto original=points;points.push_back({Eigen::Vector2d::Zero(),Vec3::Zero(),-1});
  std::sort(points.begin(),points.end(),[](const P& a,const P& b){return std::make_tuple(a.q[0],a.q[1],a.id)<std::make_tuple(b.q[0],b.q[1],b.id);});
  auto cross=[](const Eigen::Vector2d& a,const Eigen::Vector2d& b){return a[0]*b[1]-a[1]*b[0];};
  std::vector<P> hull;
  for(const auto& p:points){while(hull.size()>1&&cross(hull.back().q-hull[hull.size()-2].q,p.q-hull.back().q)<=0)hull.pop_back();hull.push_back(p);}
  const auto lower=hull.size();
  for(int j=int(points.size())-2;j>=0;--j){const auto& p=points[j];while(hull.size()>lower&&cross(hull.back().q-hull[hull.size()-2].q,p.q-hull.back().q)<=0)hull.pop_back();hull.push_back(p);}
  if(hull.size()<4)return 0;hull.pop_back();std::vector<bool> vertex(original.size(),false);for(const auto& p:hull)if(p.id>=0)vertex[p.id]=true;
  size_t count=0;
  for(const auto& p:original) {
    if(vertex[p.id])continue;
    for(size_t edge=0;edge<hull.size();++edge) {
      const auto& a=hull[edge];const auto& b=hull[(edge+1)%hull.size()];
      const double determinant=cross(a.q,b.q);if(std::abs(determinant)<1e-16)continue;
      const double l=cross(p.q,b.q)/determinant,r=cross(a.q,p.q)/determinant;
      if(l<0||r<0||l+r>1)continue;
      const Vec3 residual=p.world-l*a.world-r*b.world;
      double bound=residual.dot(group.reference.head<3>()-center);
      for(int k=0;k<group.count;++k)bound+=(o.max_offset+o.max_radius)*(group.maps[k].transpose()*residual).norm();
      if(l+r+bound<1-1e-10){++count;break;}
    }
  }
  return count;
}
int main(int argc,char**argv){try {
  if(argc!=11)throw std::runtime_error("seed map geometry planes xmin ymin zmin xmax ymax zmax");
  const auto seed=read(argv[1]),points=read(argv[2]);
  Bounds bounds{Vec3(std::stod(argv[5]),std::stod(argv[6]),std::stod(argv[7])),Vec3(std::stod(argv[8]),std::stod(argv[9]),std::stod(argv[10]))};
  double length=0;for(size_t i=1;i<seed.size();++i)length+=(seed[i]-seed[i-1]).norm();
  std::vector<Sphere> obstacles;for(const auto& x:points)obstacles.push_back({x,std::sqrt(3)*.25/2});
  ObstacleIndex index(std::move(obstacles));Options o;
  o.control_spacing=.4;o.min_radius=.03;o.budget_ms=10000;o.robot_radius=.2;o.safety_margin=.05;
  o.local_range=3;o.local_progress=7;o.frame_smoothing=1.5;o.section_smoothing=.2;o.jet_smoothing=.01;o.certify_jacobian=true;o.initial_iterations=8;
  double diagonal=0;Vec3 a,b=seed.front();
  for(size_t i=1;i<seed.size();) {
    a=b;if((a-seed[i]).norm()>7)b=a+(seed[i]-a).normalized()*7;else b=seed[i++];
    const Vec3 lo=(a.cwiseMin(b)-Vec3::Constant(3)).cwiseMax(bounds.lower+Vec3::Constant(.252));
    const Vec3 hi=(a.cwiseMax(b)+Vec3::Constant(3)).cwiseMin(bounds.upper-Vec3::Constant(.252));
    diagonal=std::max(diagonal,(hi-lo).norm());
  }
  o.max_radius=.5*diagonal+.001;o.max_offset=diagonal+.001;
  auto h5=generate(seed,index,bounds,o);if(!h5)throw std::runtime_error(h5.message);
  const auto geometry=ns::Geometry::read(argv[3]);const auto recovered=ns::CubicModel::project(geometry);
  auto model=ns::CubicModel::fromSeed(seed,.4,.033,1.5);
  for(int i=0;i<model.basis.count;++i) {
    model.x.segment<2>(5*i)=model.frames[i].transpose()*(recovered.reference[i]-model.reference[i]);
    Eigen::Matrix2d s;s<<recovered.x[5*i+2],recovered.x[5*i+3],recovered.x[5*i+3],recovered.x[5*i+4];
    s=(model.frames[i].transpose()*recovered.frames[i]*s).eval();
    model.x.segment<3>(5*i+2)<<s(0,0),.5*(s(0,1)+s(1,0)),s(1,1);
  }
  const auto reconstructed=model.geometry();double reconstruction_error=0;
  for(size_t i=0;i<geometry.controls.size();++i)reconstruction_error=std::max(reconstruction_error,(geometry.controls[i]-reconstructed.controls[i]).norm());
  std::ifstream input(argv[4]);std::string line;std::getline(input,line);std::vector<detail::SupportPlane> cover;
  while(std::getline(input,line)) {
    std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);detail::SupportPlane p;
    row>>p.span>>p.begin>>p.end>>p.normal[0]>>p.normal[1]>>p.normal[2]>>p.bound;cover.push_back(p);
  }
  ns::OptimizerSettings settings;ns::applyNativePreset(ns::NativePreset::Balanced,o,settings);
  ns::CubicProblem problem(model,cover,length,o,settings);
  size_t group_counts[5]={},cut_counts[5]={},rank2_groups=0,rank2_rows=0,rank2_survivors=0;
  double max_rank2_residual=0,min_radius=INFINITY;
  size_t proof_pruned=0;
  for(const auto& group:problem.groups) {
    proof_pruned+=projectedProofs(group,model.x,o);
    ++group_counts[group.count];cut_counts[group.count]+=group.supports.size();
    Eigen::Matrix3d gram=Eigen::Matrix3d::Zero();for(int k=0;k<group.count;++k)gram.noalias()+=group.maps[k]*group.maps[k].transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(gram);
    const auto y=group.geometry(model.x);
    for(const auto& p:group.supports)min_radius=std::min(min_radius,(ns::shape(y).transpose()*p.normal).norm());
    // Structural candidate count only: not a production redundancy proof.
    if(std::max(0.,eigen.eigenvalues()[0])>1e-20*eigen.eigenvalues()[2])continue;
    const Eigen::Matrix<double,3,2> u=eigen.eigenvectors().rightCols<2>();
    double residual=0;for(int k=0;k<group.count;++k)residual=std::max(residual,(group.maps[k]-u*u.transpose()*group.maps[k]).norm());
    if(residual>1e-10)continue;
    max_rank2_residual=std::max(max_rank2_residual,residual);++rank2_groups;rank2_rows+=group.supports.size();
    std::vector<detail::Node> nodes(2);nodes[0].x.setZero();nodes[1].x.setZero();
    nodes[0].x.head<2>()=u.transpose()*(y.head<3>()-group.reference.head<3>());
    std::vector<detail::Cut> cuts;
    for(const auto& p:group.supports) {
      const Eigen::Vector2d n=u.transpose()*p.normal;
      if(n.norm()<=1e-12)continue;
      cuts.push_back({0,n,Eigen::Vector2d::Zero(),p.bound-p.normal.dot(group.reference.head<3>()),false});
    }
    rank2_survivors+=detail::irredundant(nodes,cuts).size();
  }
  std::cout<<std::setprecision(17)<<"{\"h5_constraints\":"<<h5.metrics.constraints<<",\"h5_active\":"<<h5.metrics.active_constraints
           <<",\"h5_nodes\":"<<h5.metrics.nodes<<",\"h5_newton\":"<<h5.metrics.newton_steps
           <<",\"c3_geometric_cuts\":"<<problem.geometric_cuts<<",\"c3_groups\":"<<problem.groups.size()
           <<",\"c3_group_support_counts\":["<<cut_counts[1]<<','<<cut_counts[2]<<','<<cut_counts[3]<<','<<cut_counts[4]<<']'
           <<",\"c3_group_dependency_counts\":["<<group_counts[1]<<','<<group_counts[2]<<','<<group_counts[3]<<','<<group_counts[4]<<']'
           <<",\"rank2_candidate_groups\":"<<rank2_groups<<",\"rank2_candidate_rows\":"<<rank2_rows
           <<",\"rank2_candidate_survivors\":"<<rank2_survivors<<",\"max_rank2_residual\":"<<max_rank2_residual
           <<",\"projected_residual_proof_pruned\":"<<proof_pruned
           <<",\"min_support_radius\":"<<min_radius<<",\"reconstruction_error\":"<<reconstruction_error<<"}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

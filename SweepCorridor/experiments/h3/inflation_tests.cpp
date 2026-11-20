// SPDX-License-Identifier: MIT
#include "audit_cover.hpp"
#include "generator.hpp"
#include <cassert>
#include <iostream>
#include <random>

namespace sc=SweepCorridor;
namespace h3=sc::experiments::h3;
namespace c3=sc::internal::cubic;

void voxelDistances() {
  const sc::Vec3 center=sc::Vec3::Zero();
  auto distance=[&](const sc::Vec3& a,const sc::Vec3& b) {
    return sc::internal::segmentVoxelDistanceSquared(a,b,center,1.);
  };
  assert(std::abs(distance({-3,1.3,1.4},{3,1.3,1.4})-.25)<1e-14);
  assert(std::abs(distance({0,2.2,0},{2.2,0,0})-.02)<1e-14);
  assert(distance({2,0,0},{0,2,0})==0);
  assert(distance({2,2,2},{3,3,3})==3);
  assert(distance({2,2,2},{2,2,2})==3);
  // Independent one-dimensional minimization, including oblique corner
  // contacts and segments whose nearest point is an endpoint.
  std::mt19937_64 rng(37);
  std::uniform_real_distribution<double> draw(-4,4);
  for(int k=0;k<256;++k) {
    const sc::Vec3 a{draw(rng),draw(rng),draw(rng)}, b{draw(rng),draw(rng),draw(rng)};
    auto f=[&](double t) {
      const sc::Vec3 point=a+t*(b-a);
      return (point-point.cwiseMax(-1).cwiseMin(1)).squaredNorm();
    };
    double lo=0,hi=1;
    for(int j=0;j<100;++j) {
      const double l=(2*lo+hi)/3, r=(lo+2*hi)/3;
      if(f(l)<f(r)) hi=r; else lo=l;
    }
    assert(std::abs(distance(a,b)-f(.5*(lo+hi)))<1e-12);
  }
}

void subdivision() {
  const std::vector<sc::Vec3> seed{{0,0,0},{2,.5,0},{4,.6,.4}};
  const auto reference=c3::CubicModel::fromSeed(seed,.8,.25,1.5).geometry();
  h3::Model old(reference);
  old.x[0]=-.2;old.x[old.x.size()-6]=.3;
  const auto coarse=old.geometry();
  const auto model=h3::bisectModel(old,reference);
  const auto fine=model.geometry();
  assert(fine.pieces()==2*coarse.pieces());
  assert((model.reference.front()-seed.front()).norm()<1e-14);
  assert((model.reference.back()-seed.back()).norm()<1e-14);
  for(int i=0;i<=100;++i) {
    const auto a=coarse.section(i/100.), b=fine.section(i/100.);
    assert((a.center-b.center).norm()<1e-12 && (a.shape-b.shape).norm()<1e-12);
    assert((a.first-b.first).norm()<1e-11 && (a.shape_first-b.shape_first).norm()<1e-11);
  }
  const auto n=sc::Vec3::UnitY();
  const auto planes=h3::bisectCover({{0,.125,.75,n,2},{1,.74,.81,n,3}});
  assert(planes.size()==3 && planes[0].span==0 && planes[1].span==1 && planes[2].span==3);
  assert(planes[0].begin==.25 && planes[0].end==1 && planes[1].begin==0 && planes[1].end==.5);
  assert(std::abs(planes[2].begin-.48)<1e-14 && std::abs(planes[2].end-.62)<1e-14);
}

void voxelIndexQueries() {
  std::vector<sc::Vec3> centers;
  for(int i=0;i<257;++i)
    centers.emplace_back(.25*((17*i)%59-22),.25*((7*i)%23-11),.25*(i%5));
  centers.push_back(centers.front());
  centers.push_back({1e5,-2e5,3e5}); // Many distinct centers share a Morton key.
  const auto map=sc::ObstacleIndex::fromOccupiedVoxels(centers,.25);
  const double radius=std::sqrt(3.)*.125;
  for(size_t k=0;k<centers.size();++k) {
    const sc::Vec3 lo=centers[k]-sc::Vec3(.4,.1,.6), hi=centers[k]+sc::Vec3(.2,.7,.1);
    std::vector<int> expected, actual;
    for(size_t i=0;i<centers.size();++i) {
      if((centers[i].array()+radius>=lo.array()).all() && (centers[i].array()-radius<=hi.array()).all())
        expected.push_back(i);
    }
    map.query(lo,hi,actual);
    std::sort(actual.begin(),actual.end());
    assert(actual==expected);
  }
}

void movingCaps() {
  h3::Model model(c3::CubicModel::fromSeed({{-2,0,0},{2,0,0}},.45,.6,1.).geometry());
  for(int i=0;i<model.basis.count;++i) {
    model.x.segment<3>(6*i)<<1.5*(model.basis.breaks[i]-.5),.04,-.03;
    model.jets[i][0]+=1.5;
  }
  const auto anchors=[&]() {
    std::array<h3::Anchor,2> a;
    a[1].u=1;
    assert(std::isfinite(h3::anchorObjective(model,model.x,a,1.)));
    return a;
  };
  const auto a=anchors();
  assert(a[0].u>0 && a[1].u<1);
  const Eigen::VectorXd dx=Eigen::VectorXd::Random(model.x.size());
  std::vector<c3::V9> dj(model.basis.count);
  for(auto& j:dj)j.setRandom();
  double predicted=0;
  const double tail=h3::capVolume(model,model.x,a,[&](int span,const c3::Curve<3,c3::V9>& g) {
    const double h=model.basis.breaks[span+1]-model.basis.breaks[span];
    predicted+=(g[0]+g[1]).dot(model.nodeDirection(dx,span))+
               (g[2]+g[3]).dot(model.nodeDirection(dx,span+1))+
               (h/3)*(g[1].dot(dj[span])-g[2].dot(dj[span+1]));
  });
  const auto initial=model;
  auto evaluate=[&](double t) {
    model.x=initial.x+t*dx;
    for(int i=0;i<model.basis.count;++i)model.jets[i]=initial.jets[i]+t*dj[i];
    return h3::capVolume(model,model.x,anchors());
  };
  const double measured=(evaluate(1e-6)-evaluate(-1e-6))/2e-6;
  assert(std::abs(predicted-measured)<1e-6);
  model=initial;
  model.anchor_parameters={a[0].u,a[1].u};
  const auto deadline=c3::Deadline(10000,c3::Clock::now());
  assert(std::abs(h3::sweptVolume(model,model.x)-tail-h3::coreQuality(model,deadline).volume)<1e-10);
}

void sectionAreaAndWall() {
  c3::Geometry g;g.breaks={0,.5,1};
  sc::Mat32 shape;shape<<1,0,0,1,0,0;
  std::vector<sc::SupportPlane> cover;
  for(int i=0;i<2;++i) {
    for(int k=0;k<4;++k)g.controls.push_back(c3::pack(sc::Vec3(0,0,2*i+2.*k/3),shape));
    cover.push_back({i,0,1,sc::Vec3::UnitX(),2});
    cover.push_back({i,0,1,-sc::Vec3::UnitX(),1});
    cover.push_back({i,0,1,sc::Vec3::UnitY(),1});
    cover.push_back({i,0,1,-sc::Vec3::UnitY(),1});
  }
  h3::Model model(g);model.anchor_parameters={.25,.75};
  const auto gain=h3::sectionGains(g,cover,model.anchor_parameters);
  // Translate by .5 and stretch one axis by 1.5 in the only open direction.
  for(double v:gain)assert(std::abs(v-.5)<1e-12);
  const auto q=h3::coreQuality(model,c3::Deadline(1000,c3::Clock::now()));
  assert(std::abs(q.volume-2*std::acos(-1.))<1e-11);
  assert(std::abs(q.bending-4*std::acos(-1.))<1e-10);
}

int main() {
  sectionAreaAndWall();
  movingCaps();
  voxelDistances();
  voxelIndexQueries();
  subdivision();
  const auto voxels=sc::ObstacleIndex::fromOccupiedVoxels({{0,0,0}},.5);
  const sc::ObstacleIndex spheres(voxels.obstacles());
  assert(voxels.voxelHalfWidth()==.25 && spheres.voxelHalfWidth()==0);
  for(double resolution:{0.,-1.,std::numeric_limits<double>::infinity()}) {
    bool rejected=false;
    try { sc::ObstacleIndex::fromOccupiedVoxels({},resolution); }
    catch(const std::invalid_argument&) { rejected=true; }
    assert(rejected);
  }
  sc::Options options;
  const sc::Bounds bounds{{-5,-5,-5},{5,5,5}};
  const h3::OptimizerSettings settings;
  sc::Mat32 section;
  section<<1,0,0,1,0,0;
  c3::Curve<3,c3::V9> cylinder;
  // z(t)=.06*t+.5*t^2 has varying speed and constant radius.
  const std::array<double,4> heights{0.,.02,.56-1.06/3,.56};
  for(int j=0;j<4;++j)cylinder[j]=c3::pack(sc::Vec3(0,0,heights[j]),section);
  c3::Certificate certificate;
  assert(!c3::jacobianCertificate<3>(cylinder,certificate,.075));
  const c3::JacobianRequirement disk{.25,true,.02};
  assert(c3::jacobianCertificate<3>(cylinder,certificate,disk));
  const auto halves=c3::split<3>(cylinder);
  assert(c3::jacobianCertificate<3>(halves.first,certificate,disk.onSpan(.5)));
  assert(c3::jacobianCertificate<3>(halves.second,certificate,disk.onSpan(.5)));
  assert(!c3::jacobianCertificate<3>(cylinder,certificate,{.25,true,.1}));
  // An actually compressed inner edge has J_min/J_axis=.1.
  for(int j=0;j<4;++j) {
    section(2,0)=.9*j/3;
    cylinder[j]=c3::pack(sc::Vec3(0,0,j/3.),section);
  }
  assert(!c3::jacobianCertificate<3>(cylinder,certificate,disk));
  assert(c3::jacobianCertificate<3>(cylinder,certificate,{.05,true,.02}));
  const std::vector<sc::Vec3> seed{{-2,.58,0},{2,.58,0}};
  const auto cube=h3::generate(seed,voxels,bounds,options,settings);
  assert(cube.status==sc::Status::Success);
  const auto cert=c3::certify(cube.corridor,cube.cover,options.certificate_depth,settings.regularity(4));
  assert(cert.safe() && cert.cone_relative_margin>=settings.relative_jacobian_margin);
  const double pad=options.robot_radius+options.safety_margin+options.certificate_margin;
  assert(h3::auditObstacleCover(cube.corridor,cube.cover,voxels,pad));
  assert(h3::generate(seed,spheres,bounds,options,settings).status==sc::Status::SeedCollision);
  assert(h3::generate({{-2,.49,0},{2,.49,0}},voxels,bounds,options,settings).status==sc::Status::SeedCollision);
  const auto hidden=sc::ObstacleIndex::fromOccupiedVoxels({cube.corridor.section(.5).center},.5);
  assert(!h3::auditObstacleCover(cube.corridor,cube.cover,hidden,pad));
  std::cout<<"Voxel distances, subdivision, disk and speed certificates, moving caps, section gain, wall curvature and obstacle audit passed\n";
}

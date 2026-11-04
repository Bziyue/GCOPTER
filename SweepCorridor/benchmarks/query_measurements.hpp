// SPDX-License-Identifier: MIT
#pragma once
#include <SweepCorridor/InverseQuery.hpp>
// The same normalized query sequence is used for both degrees. Construction,
// point preparation and cache reset are outside the timed blocks.
void queryMeasurements(const sc::Corridor& c,std::ostream& log) {
  constexpr int D=3;sc::Query q(c);sc::InverseQuery inverse(c);
  constexpr int count=16384;
  std::vector<sc::Vec3> points;points.reserve(count);
  for(int i=0;i<count;++i){double u=(i+.5)/count;
    points.push_back(c.section(u).point(Eigen::Vector2d(1.1+.05*std::sin(i),.2)));}
  double error=0;
  for(int i=0;i<count;i+=43){double u=(i+.5)/count;auto a=q.evaluate(u,points[i]);auto b=q.evaluateFirst(u,points[i]);
    error=std::max(error,(a.gradient-b.gradient).norm()/(1+a.gradient.norm()));}
  for(const std::string mode:{"full_hessian","first_gradient","inverse_unique","inverse_repeated"}) {
    for(int rep=-1;rep<7;++rep){inverse.reset();double checksum=0;
      const auto start=std::chrono::steady_clock::now();
      for(int i=0;i<count;++i){int j=mode=="inverse_repeated"?i%64:i;double u=(j+.5)/count;
        if(mode=="full_hessian"){auto a=q.evaluate(u,points[j]);checksum+=a.value+a.gradient.sum()+a.hessian.sum();}
        else if(mode=="first_gradient"){auto a=q.evaluateFirst(u,points[j]);checksum+=a.value+a.gradient.sum();}
        else {auto a=inverse(points[j],std::clamp(u+.001,0.,1.));checksum+=a.radial+a.grad_radial.sum()+a.u;}
      }
      const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      log<<"{\"kind\":\"query\",\"degree\":"<<D<<",\"mode\":\""<<mode<<"\",\"repeat\":"<<rep
         <<",\"count\":"<<count<<",\"ms\":"<<elapsed<<",\"checksum\":"<<checksum<<",\"relative_gradient_error\":"<<error<<"}\n";
    }
  }
}

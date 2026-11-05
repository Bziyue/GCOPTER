// SPDX-License-Identifier: MIT
#pragma once
#include <Eigen/Cholesky>
#include <vector>

namespace SweepCorridor::experiments::h3 {
using Block = Eigen::Matrix<double,6,6>;
using State = Eigen::Matrix<double,6,1>;

// H3 couples only neighboring nodal VALUES when the shared jets are frozen.
// Store whole 6x6 blocks, equilibrate once, and reuse the Schur factors for
// the volume rank-one correction and the two free-end cap solves.
struct BlockTridiagonal {
  std::vector<Block> diagonal,lower;
  std::vector<Block> scaled_lower,transfer;
  std::vector<Eigen::LLT<Block>> factors;
  std::vector<State> scaling;
  int n=0;
  BlockTridiagonal()=default;
  explicit BlockTridiagonal(int size):diagonal(size/6,Block::Zero()),lower(size/6-1,Block::Zero()),n(size){}
  void assign(const BlockTridiagonal& q){n=q.n;diagonal=q.diagonal;lower=q.lower;}
  void add(int i,int j,double v){
    if(i<j)std::swap(i,j);
    if(i/6==j/6){auto& b=diagonal[i/6];b(i%6,j%6)+=v;if(i!=j)b(j%6,i%6)+=v;}
    else lower[j/6](i%6,j%6)+=v;
  }
  void addBlock(int row,int col,const Block& b){
    if(row==col)diagonal[row]+=b;
    else lower[col]+=b;
  }
  double get(int i,int j)const{
    if(i<j)std::swap(i,j);
    if(i/6==j/6)return diagonal[i/6](i%6,j%6);
    return i/6==j/6+1?lower[j/6](i%6,j%6):0.;
  }
  Eigen::VectorXd multiply(const Eigen::VectorXd& x)const{
    Eigen::VectorXd y(n);
    for(int i=0;i<n/6;++i)y.segment<6>(6*i).noalias()=diagonal[i]*x.segment<6>(6*i);
    for(int i=0;i<n/6-1;++i){y.segment<6>(6*i+6).noalias()+=lower[i]*x.segment<6>(6*i);
      y.segment<6>(6*i).noalias()+=lower[i].transpose()*x.segment<6>(6*i+6);}
    return y;
  }
  bool factor(double damping=0){
    const int count=n/6;scaling.resize(count);factors.resize(count);
    transfer.resize(count-1);scaled_lower.resize(count-1);
    for(int i=0;i<count;++i)scaling[i]=diagonal[i].diagonal().cwiseMax(1e-30).cwiseSqrt().cwiseInverse();
    for(int i=0;i<count-1;++i)scaled_lower[i]=scaling[i+1].asDiagonal()*lower[i]*scaling[i].asDiagonal();
    for(int i=0;i<count;++i){
      Block schur=scaling[i].asDiagonal()*diagonal[i]*scaling[i].asDiagonal();
      schur.diagonal().array()+=damping;
      if(i)schur.noalias()-=scaled_lower[i-1]*transfer[i-1];
      factors[i].compute(schur);
      if(factors[i].info()!=Eigen::Success)return false;
      if(i+1<count)transfer[i]=factors[i].solve(scaled_lower[i].transpose());
    }
    return true;
  }
  void solveInto(const Eigen::VectorXd& rhs,Eigen::VectorXd& answer)const{
    answer.resize(n);
    for(int i=0;i<n/6;++i){State b=rhs.segment<6>(6*i).cwiseProduct(scaling[i]);
      if(i)b.noalias()-=scaled_lower[i-1]*answer.segment<6>(6*i-6);
      answer.segment<6>(6*i)=factors[i].solve(b);}
    for(int i=n/6-2;i>=0;--i)answer.segment<6>(6*i).noalias()-=transfer[i]*answer.segment<6>(6*i+6);
    for(int i=0;i<n/6;++i)answer.segment<6>(6*i).array()*=scaling[i].array();
  }
};
}

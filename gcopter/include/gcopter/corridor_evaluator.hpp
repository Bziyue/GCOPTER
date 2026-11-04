// SPDX-License-Identifier: MIT
#pragma once
#include "gcopter/direct_corridor.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gcopter::direct {
struct FirstSection {
  Vec3 center, first;
  Mat32 shape, shape_first;
};

/** @brief Immutable prepared local-power coefficients; construction is the validation boundary.
 * The original Corridor and its Bernstein certificates are unchanged. Each instance owns its data.
 * Trusted callers guarantee u in [0,1]. Cursor is caller-owned, never shared mutable geometry.
 */
class PreparedCorridor {
 public:
  using Matrix = Eigen::Matrix3d; // columns: center, shape.col(0), shape.col(1)
  struct Piece { std::array<Matrix,6> power; double inverse_width; };
  explicit PreparedCorridor(const Corridor& corridor):knots_(corridor.knots()),centers_(corridor.centerControls()),shapes_(corridor.shapeControls()) {
    if(corridor.pieces()==0)throw std::invalid_argument("empty corridor evaluator");
    pieces_.resize(corridor.pieces());
    constexpr int binomial[6]={1,5,10,10,5,1};
    for(size_t i=0;i<pieces_.size();++i) {
      std::array<Matrix,6> differences;
      for(int j=0;j<6;++j) {
        differences[j].col(0)=corridor.centerControls()[6*i+j];
        differences[j].rightCols<2>()=corridor.shapeControls()[6*i+j];
      }
      for(int k=0;k<6;++k) {
        pieces_[i].power[k]=binomial[k]*differences[0];
        for(int j=0;j<5-k;++j)differences[j]=differences[j+1]-differences[j];
      }
      pieces_[i].inverse_width=1/(knots_[i+1]-knots_[i]);
    }
  }
  size_t locate(double u,size_t cursor) const {
    if(u>=knots_[cursor] && (u<knots_[cursor+1] || (cursor+1==pieces_.size()&&u==1)))return cursor;
    return std::min(pieces_.size()-1,size_t(std::upper_bound(knots_.begin(),knots_.end(),u)-knots_.begin()-1));
  }
  FirstSection firstUnchecked(double u,size_t& cursor) const {
    cursor=locate(u,cursor);
    const auto& piece=pieces_[cursor];
    const double t=(u-knots_[cursor])*piece.inverse_width;
    Matrix value=piece.power[5],derivative=Matrix::Zero();
    for(int k=4;k>=0;--k) {derivative=derivative*t+value;value=value*t+piece.power[k];}
    derivative*=piece.inverse_width;
    return {value.col(0),derivative.col(0),value.rightCols<2>(),derivative.rightCols<2>()};
  }
  /** @brief Preserve the reference Bernstein operation order for sensitive solvers.
   * Shared powers omit second derivatives without reassociating the value/first-derivative sums.
   */
  FirstSection firstCompatibleUnchecked(double u,size_t& cursor) const {
    cursor=locate(u,cursor);
    const double h=knots_[cursor+1]-knots_[cursor],t=(u-knots_[cursor])/h;
    std::array<double,6> left,right;
    for(int j=0;j<6;++j){left[j]=std::pow(t,j);right[j]=std::pow(1-t,j);}
    constexpr double choose5[6]={1,5,10,10,5,1},choose4[5]={1,4,6,4,1};
    FirstSection out{Vec3::Zero(),Vec3::Zero(),Mat32::Zero(),Mat32::Zero()};
    for(int j=0;j<6;++j){
      const size_t k=6*cursor+j;
      const double b=choose5[j]*left[j]*right[5-j];
      out.center+=b*centers_[k];out.shape+=b*shapes_[k];
      if(j<5){const double d=5/h*(choose4[j]*left[j]*right[4-j]);
        out.first+=d*(centers_[k+1]-centers_[k]);out.shape_first+=d*(shapes_[k+1]-shapes_[k]);}
    }
    return out;
  }
  const std::vector<double>& knots() const{return knots_;}
  const Piece& piece(size_t i) const{return pieces_[i];}
 private:
  std::vector<double> knots_;
  std::vector<Piece> pieces_;
  std::vector<Vec3> centers_;
  std::vector<Mat32> shapes_;
};
} // namespace gcopter::direct

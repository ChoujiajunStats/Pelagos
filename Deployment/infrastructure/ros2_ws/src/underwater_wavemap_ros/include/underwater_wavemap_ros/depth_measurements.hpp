#pragma once
#include <algorithm>
#include <cmath>
#include <utility>
#include <Eigen/Core>
#include <wavemap/core/integrator/measurement_model/continuous_ray.h>

namespace underwater_wavemap_ros {
// ProjectiveIntegratorConfig::max_range is an update range, not a measurement
// validity limit. Preserve actual finite far depths for native free-space beams;
// clipping a measurement to max_range would invent a surface at that distance.
inline float sanitizeDepthMeasurement(float value, double min_range) {
  return std::isfinite(value) && value >= min_range ? value : 0.f;
}

inline void filterDepthEdges(Eigen::MatrixXf* depth, double absolute, double relative) {
  if(absolute<=0 && relative<=0)return;
  const Eigen::MatrixXf original=*depth;
  const std::pair<int,int> offsets[]={{-1,0},{1,0},{0,-1},{0,1}};
  for(int row=0;row<depth->rows();++row)for(int col=0;col<depth->cols();++col) {
    const float z=original(row,col);if(z<=0)continue;
    for(const auto& offset:offsets) {
      const int r=row+offset.first,c=col+offset.second;
      if(r<0||r>=depth->rows()||c<0||c>=depth->cols())continue;
      const float other=original(r,c);
      if(other>0 && std::abs(z-other)>absolute+relative*std::min(z,other)) {
        (*depth)(row,col)=0.f;break;
      }
    }
  }
}

// Rejected stereo pixels remain unobserved, including near-camera cells.
class ValidDepthContinuousRay final : public wavemap::ContinuousRay {
 public:
  ValidDepthContinuousRay(const wavemap::ContinuousRayConfig& config,
      wavemap::ProjectorBase::ConstPtr projector,wavemap::Image<>::ConstPtr image)
      : wavemap::ContinuousRay(config,projector,image),projector_(projector),image_(image) {}
  wavemap::FloatingPoint computeUpdate(const wavemap::SensorCoordinates& coordinates) const override {
    const auto index=projector_->imageToNearestIndex(coordinates.image);
    if(!image_->isIndexWithinBounds(index))return 0.f;
    const auto depth=image_->at(index);
    if(!std::isfinite(depth)||depth<=0.f)return 0.f;
    return wavemap::ContinuousRay::computeUpdate(coordinates);
  }
 private:
  wavemap::ProjectorBase::ConstPtr projector_;
  wavemap::Image<>::ConstPtr image_;
};
}

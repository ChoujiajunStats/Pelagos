#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace dapeng {
using Point = std::array<double, 3>;
enum Status { Unknown = 0, Occupied = 1, Free = 2 };

class EsdfGrid {
 public:
  void assign(Point origin, double resolution, std::array<uint32_t, 3> shape,
              const std::vector<float>& distance) {
    uint64_t count = 1;
    for (auto dim : shape) {
      if (!dim || dim > 1000) throw std::invalid_argument("Invalid ESDF dimension");
      count *= dim;
    }
    if (!std::isfinite(resolution) || resolution < 0.01 || resolution > 1.0 ||
        count > 1000000 || count != distance.size())
      throw std::invalid_argument("Invalid ESDF grid size or resolution");
    for (auto value : origin)
      if (!std::isfinite(value)) throw std::invalid_argument("Invalid ESDF origin");
    origin_ = origin;
    resolution_ = resolution;
    shape_ = shape;
    data_ = distance;
    free_boxes_.clear();
  }

  double resolution() const { return resolution_; }
  bool empty() const { return data_.empty(); }
  void clear() { data_.clear(); free_boxes_.clear(); }

  float distance(const Point& point) const {
    std::array<int, 3> index;
    for (int i = 0; i < 3; ++i) {
      if (!std::isfinite(point[i])) return unknown();
      const double relative = (point[i] - origin_[i]) / resolution_;
      if (relative < 0 || relative >= shape_[i] || empty()) return unknown();
      index[i] = static_cast<int>(std::floor(relative));
    }
    const size_t offset = (size_t(index[0]) * shape_[1] + index[1]) * shape_[2] + index[2];
    float result = data_[offset];
    if (!std::isfinite(result) || result <= -999.0F) {
      for (const auto& box : free_boxes_) {
        bool inside = true;
        for (int i = 0; i < 3; ++i) inside &= std::abs(point[i] - box.first[i]) <= box.second[i] / 2;
        if (inside) return float(resolution_ * 2);
      }
      return unknown();
    }
    return result;
  }

  Status status(const Point& point) const {
    const float d = distance(point);
    return !std::isfinite(d) ? Unknown : (d <= resolution_ * 0.5 ? Occupied : Free);
  }

  Status box(const Point& center, const Point& size, bool stop_at_unknown) const {
    std::array<int, 3> low, high;
    uint64_t count = 1;
    for (int i = 0; i < 3; ++i) {
      if (!std::isfinite(center[i]) || !std::isfinite(size[i]) || size[i] < 0) return Unknown;
      const double a = (center[i] - size[i] / 2 - origin_[i]) / resolution_;
      const double b = (center[i] + size[i] / 2 - origin_[i]) / resolution_;
      if (!std::isfinite(a) || !std::isfinite(b) || a < -1000 || b > 2000) return Unknown;
      low[i] = int(std::floor(a));
      high[i] = int(std::floor(b));
      const int span = high[i] - low[i] + 1;
      if (span <= 0 || span > 1000) return Unknown;
      count *= span;
    }
    if (count > 1000000) return Unknown;
    bool seen_unknown = false;
    for (int x = low[0]; x <= high[0]; ++x)
      for (int y = low[1]; y <= high[1]; ++y)
        for (int z = low[2]; z <= high[2]; ++z) {
          const Status s = status({origin_[0] + (x + .5) * resolution_,
                                   origin_[1] + (y + .5) * resolution_,
                                   origin_[2] + (z + .5) * resolution_});
          if (s == Occupied) return Occupied;
          seen_unknown |= s == Unknown;
        }
    return seen_unknown && stop_at_unknown ? Unknown : Free;
  }

  Status path(const Point& start, const Point& end, const Point& size,
              bool stop_at_unknown) const {
    double length = 0;
    for (int i = 0; i < 3; ++i) length += (end[i] - start[i]) * (end[i] - start[i]);
    length = std::sqrt(length);
    if (!std::isfinite(length) || length > 100) return Unknown;
    const int steps = std::max(1, int(std::ceil(length / (resolution_ * 0.5))));
    for (int step = 0; step <= steps; ++step) {
      Point p;
      for (int i = 0; i < 3; ++i) p[i] = start[i] + (end[i] - start[i]) * step / steps;
      const Status s = box(p, size, stop_at_unknown);
      if (s != Free) return s;
    }
    return Free;
  }

  void augmentRobotBox(const Point& center, const Point& size) {
    for (int i = 0; i < 3; ++i)
      if (!std::isfinite(center[i]) || !std::isfinite(size[i]) || size[i] < 0 || size[i] > 2)
        throw std::invalid_argument("Invalid robot free-space box");
    if (free_boxes_.size() >= 16) free_boxes_.erase(free_boxes_.begin());
    free_boxes_.emplace_back(center, size);
  }

  template <typename Callback>
  void forEach(Callback callback) const {
    if (empty()) return;
    for (uint32_t x = 0; x < shape_[0]; ++x)
      for (uint32_t y = 0; y < shape_[1]; ++y)
        for (uint32_t z = 0; z < shape_[2]; ++z) {
          const Point p{origin_[0] + (x + .5) * resolution_,
                        origin_[1] + (y + .5) * resolution_,
                        origin_[2] + (z + .5) * resolution_};
          callback(p, status(p));
        }
  }

 private:
  static float unknown() { return std::numeric_limits<float>::quiet_NaN(); }
  Point origin_{0, 0, 0};
  double resolution_ = .1;
  std::array<uint32_t, 3> shape_{0, 0, 0};
  std::vector<float> data_;
  std::vector<std::pair<Point, Point>> free_boxes_;
};
}  // namespace dapeng

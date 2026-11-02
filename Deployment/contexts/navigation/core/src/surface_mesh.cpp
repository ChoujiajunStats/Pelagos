#include "dapeng_navigation/surface_mesh.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <wavemap/core/indexing/index_conversions.h>
#include <wavemap/core/map/hashed_wavelet_octree.h>
#include <wavemap/core/map/hashed_chunked_wavelet_octree.h>
#include <wavemap/core/utils/query/occupancy_classifier.h>
namespace dapeng_navigation {
namespace {
MeshKey key(const wavemap::Index3D& i){return {i.x(),i.y(),i.z()};}
wavemap::Index3D index(const MeshKey& k){return {k[0],k[1],k[2]};}
const std::array<MeshKey,7> offsets={MeshKey{0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
MeshKey add(MeshKey a,const MeshKey& b){for(int i=0;i<3;++i)a[i]+=b[i];return a;}
uint64_t hash(uint64_t a,uint64_t b){return a^(b+0x9e3779b97f4a7c15ULL+(a<<6)+(a>>2));}
}
void MeshConfig::validate(float native,int height) const {
  if(!std::isfinite(resolution_m)||resolution_m<native*(1.-1e-5)||!std::isfinite(radius_m)||radius_m<=0||
     !std::isfinite(read_budget_ms)||read_budget_ms<=0||read_budget_ms>50||!max_blocks_per_update||max_blocks_per_update>8||
     !max_source_leaves||!max_triangles||!max_blocks||max_blocks>512||height<1||height>9)
    throw std::invalid_argument("invalid surface mesh geometry or budgets");
  const double p=std::log2(resolution_m/native);
  if(std::abs(p-std::round(p))>1e-4||p>height||((1<<height)/int(std::lround(resolution_m/native)))>128)
    throw std::invalid_argument("mesh resolution must be an aligned native power of two, at most 128 cells per block side");
}
MeshCapture SurfaceTracker::capture(const wavemap::MapBase& map,uint64_t version,const wavemap::Point3D& center,
                                   const MeshConfig& config,float threshold) {
  config.validate(map.getMinCellWidth(),map.getTreeHeight());
  if(!center.allFinite()||!std::isfinite(threshold))throw std::invalid_argument("nonfinite surface capture input");
  const auto began=Clock::now();MeshCapture out;out.version=version;
  const int side=1<<map.getTreeHeight();const float native=map.getMinCellWidth(),width=native*side;
  const wavemap::OccupancyClassifier classifier(threshold);
  const auto visit=[&](const auto& typed){
    std::map<MeshKey,uint64_t> stamps;
    typed.forEachBlock([&](const auto& i,const auto& b){
      stamps[key(i)]=uint64_t(b.getLastUpdatedStamp().time_since_epoch().count());
      const wavemap::Point3D lo=i.template cast<float>()*width,hi=lo+wavemap::Point3D::Constant(width);
      const auto closest=center.cwiseMax(lo).cwiseMin(hi);
      if((closest-center).squaredNorm()<=config.radius_m*config.radius_m)out.retained.insert(key(i));
    });
    if(out.retained.size()>config.max_blocks)throw std::runtime_error("surface display radius exceeds native block budget");
    std::vector<std::pair<MeshKey,uint64_t>> dirty;
    for(const auto& k:out.retained){
      uint64_t f=0;for(const auto& d:offsets){auto it=stamps.find(add(k,d));f=hash(f,it==stamps.end()?0:it->second);}
      auto s=seen_.find(k);if(s==seen_.end()||s->second!=f)dirty.emplace_back(k,f);
    }
    std::stable_sort(dirty.begin(),dirty.end(),[&](const auto& a,const auto& b){return order_[a.first]<order_[b.first];});
    out.pending=dirty.size();
    struct Exhausted{};
    size_t visited=0;
    for(const auto& [k,f]:dirty){
      if(out.changed.size()>=config.max_blocks_per_update||milliseconds(Clock::now()-began)>=config.read_budget_ms){out.budget_limited=true;break;}
      MeshBlockSource source;source.key=k;source.fingerprint=f;source.native_side=side;source.native_resolution=native;
      const wavemap::Point3D lo=index(k).template cast<float>()*width-wavemap::Point3D::Constant(float(config.resolution_m));
      const wavemap::Point3D hi=(index(k).template cast<float>()+wavemap::Point3D::Ones())*width+wavemap::Point3D::Constant(float(config.resolution_m));
      try{
        for(const auto& d:offsets){
          const auto neighbor=add(k,d);const auto* b=typed.getBlock(index(neighbor));if(!b)continue;
          b->forEachLeaf(index(neighbor),[&](const auto& node,float value){
            if(++visited>config.max_source_leaves||((visited%256)==0&&milliseconds(Clock::now()-began)>config.read_budget_ms))throw Exhausted{};
            if(!classifier.is(value,wavemap::Occupancy::kOccupied))return;
            const wavemap::Point3D minimum=wavemap::convert::nodeIndexToMinCornerIndex(node).template cast<float>()*native;
            const float w=float(1<<node.height)*native;
            if(((minimum.array()+w)<=lo.array()).any()||(minimum.array()>=hi.array()).any())return;
            source.occupied.push_back({minimum,w});
          });
        }
      }catch(const Exhausted&){out.budget_limited=true;order_[k]=++sequence_;break;} // Never publish a partial block or an artificial cut surface.
      out.changed.push_back(std::move(source));
    }
  };
  if(auto p=dynamic_cast<const wavemap::HashedWaveletOctree*>(&map))visit(*p);
  else if(auto p=dynamic_cast<const wavemap::HashedChunkedWaveletOctree*>(&map))visit(*p);
  else throw std::invalid_argument("surface meshing supports explicitly adapted HWO and HCWO only");
  for(auto it=seen_.begin();it!=seen_.end();)if(!out.retained.count(it->first)){order_.erase(it->first);it=seen_.erase(it);}else ++it;
  out.read_ms=milliseconds(Clock::now()-began);return out;
}
void SurfaceTracker::acknowledge(const MeshBlockSource& b){seen_[b.key]=b.fingerprint;order_[b.key]=++sequence_;}
SurfaceBlock greedySurface(const MeshBlockSource& s,const MeshConfig& config){
  const auto began=Clock::now();
  const int side=s.native_side,factor=int(std::lround(config.resolution_m/s.native_resolution));
  if(side<=0||factor<1||side%factor||side/factor>128)throw std::invalid_argument("invalid surface block dimensions");
  const int n=side/factor,padded=n+2;const int full=factor*factor*factor;
  std::vector<uint32_t> counts(size_t(padded)*padded*padded,0);
  const auto offset=[&](int x,int y,int z){return size_t((x+1)*padded+y+1)*padded+z+1;};
  // Exact overlap volume, in native cells. Coarsening retains only completely
  // occupied coarse cells; partial occupancy never fills FREE or UNKNOWN space.
  const wavemap::Index3D origin=index(s.key)*side;
  for(const auto& leaf:s.occupied){
    std::array<int,3> low,high,a,b;
    for(int d=0;d<3;++d){low[d]=int(std::lround(leaf.minimum[d]/s.native_resolution))-origin[d];
      high[d]=low[d]+int(std::lround(leaf.width/s.native_resolution));
      a[d]=std::max(-1,int(std::floor(double(low[d])/factor)));b[d]=std::min(n,int(std::ceil(double(high[d])/factor))-1);}
    for(int x=a[0];x<=b[0];++x)for(int y=a[1];y<=b[1];++y)for(int z=a[2];z<=b[2];++z){
      int volume=1;const int c[3]={x,y,z};
      for(int d=0;d<3;++d)volume*=std::max(0,std::min(high[d],(c[d]+1)*factor)-std::max(low[d],c[d]*factor));
      counts[offset(x,y,z)]+=uint32_t(volume);
    }
  }
  const auto occupied=[&](const std::array<int,3>& p){return counts[offset(p[0],p[1],p[2])]==uint32_t(full);};
  SurfaceBlock result;result.key=s.key;result.fingerprint=s.fingerprint;
  const wavemap::Point3D world_origin=origin.cast<float>()*s.native_resolution;
  std::vector<uint8_t> mask(size_t(n)*n);
  for(int d=0;d<3;++d){const int u=(d+1)%3,v=(d+2)%3;
    for(int sign:{-1,1})for(int slice=0;slice<n;++slice){
      for(int j=0;j<n;++j)for(int i=0;i<n;++i){std::array<int,3> p{};p[d]=slice;p[u]=i;p[v]=j;auto q=p;q[d]+=sign;
        mask[size_t(j)*n+i]=occupied(p)&&!occupied(q);}
      for(int j=0;j<n;++j)for(int i=0;i<n;){
        if(!mask[size_t(j)*n+i]){++i;continue;}
        int w=1,h=1;while(i+w<n&&mask[size_t(j)*n+i+w])++w;
        bool grow=true;while(j+h<n&&grow){for(int k=0;k<w;++k)if(!mask[size_t(j+h)*n+i+k]){grow=false;break;}if(grow)++h;}
        if(result.vertices.size()/3+2>config.max_triangles)throw std::runtime_error("surface triangle budget exceeded; block suppressed");
        std::array<wavemap::Point3D,4> p;for(auto& point:p)point=wavemap::Point3D::Zero();
        for(auto& point:p){point[d]=float(slice+(sign>0));point[u]=float(i);point[v]=float(j);}
        p[1][u]+=w;p[2][u]+=w;p[2][v]+=h;p[3][v]+=h;
        for(auto& point:p)point=world_origin+point*float(config.resolution_m);
        const int positive[6]={0,1,2,0,2,3},negative[6]={0,2,1,0,3,2};
        for(int k=0;k<6;++k)result.vertices.push_back(p[sign>0?positive[k]:negative[k]]);
        for(int y=j;y<j+h;++y)for(int x=i;x<i+w;++x)mask[size_t(y)*n+x]=0;
        i+=w;
      }
    }
  }
  result.build_ms=milliseconds(Clock::now()-began);return result;
}
} // namespace dapeng_navigation

#pragma once
#include <Eigen/Geometry>
#include <cmath>
#include <functional>
#include <vector>
namespace pelagos {
// Task constraint over an already-built source tree. No sampling, gain or
// iteration changes. Source getBestEdge/memorizeBestBranch run AFTER selection.
template <typename Node>
Node* selectForwardNode(Node* root,const Eigen::Vector3d& direction,double zero_gain,int* eligible=nullptr,
    const std::function<bool(const Eigen::Vector3d&)>& first_edge_allowed={}) {
  if(eligible)*eligible=0;
  if(!root||!direction.allFinite()||(direction.squaredNorm()==0.&&!first_edge_allowed))return nullptr;
  std::vector<Node*> nodes(root->children_.begin(),root->children_.end());
  Node* best=nullptr;double gain=zero_gain;
  for(size_t i=0;i<nodes.size();++i) {
    auto* node=nodes[i];for(auto* child:node->children_)nodes.push_back(child);
    auto* first=node;while(first->parent_&&first->parent_!=root)first=first->parent_;
    if(first->parent_!=root)continue;
    const Eigen::Vector3d displacement=first->state_.template head<3>()-root->state_.template head<3>();
    // Operator task constrains translation only. Keep the source-sampled yaw
    // and original gain unchanged, including views outside the forward half-plane.
    if(!displacement.allFinite() || (direction.squaredNorm()>0.&&displacement.dot(direction)<=0.) ||
       !std::isfinite(node->gain_) || node->gain_<=zero_gain)continue;
    if(first_edge_allowed&&!first_edge_allowed(displacement))continue;
    if(eligible)++*eligible;
    if(node->gain_>gain) {best=node;gain=node->gain_;}
  }
  return best;
}
}

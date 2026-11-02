#include <iostream>
#include <limits>
#include <stdexcept>
#include "cruise_direction.hpp"
#include "forward_selection.hpp"

struct Node {Eigen::Vector4d state_;Node* parent_=nullptr;std::vector<Node*> children_;double gain_=0;};
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
int main() {
  try {
    const Eigen::Vector3d body(0,0,0),box(.5,.5,.3);
    const std::vector<std::pair<Eigen::Vector3d,double>> at_boundary{{{3.5,0,0},.5}};
    auto exact=pelagos::cruiseDirectionClearance(body,{1,0,1},box,at_boundary,3.);
    require(!exact.valid&&exact.contact.found&&exact.contact.front_clearance_m==3.,"Exactly3m must fail strict forward clearance");
    require(pelagos::cruiseDirectionClearance(body,{1,0,0},box,{{{3.501,0,0},.5}},3.).valid,"Above3m clearance must pass");
    require(!pelagos::cruiseDirectionClearance(body,{1,0,0},box,{{{3.499,0,0},.5}},3.).valid,"Below3m clearance must fail");
    require(pelagos::cruiseDirectionClearance({0,0,1},{1,0,0},box,at_boundary,3.).valid,"Ray must use actual body altitude");
    require(!exact.valid,"A source edge climbing inZ must still test horizontal execution at actual bodyZ");
    const auto unknown=pelagos::cruiseDirectionClearance(body,{0,1,0},box,{},3.);
    require(unknown.valid&&!unknown.contact.found,"No known occupied hit may pass but must remain a no-hit observation");
    require(!pelagos::cruiseDirectionClearance(body,{0,0,1},box,{},3.).valid,"Vertical-only edge has no executable horizontal heading");
    require(!pelagos::cruiseDirectionClearance(body,{1,0,0},box,{},0.).valid,"Invalid threshold must not silently approve a direction");
    require(!pelagos::cruiseDirectionClearance({std::numeric_limits<double>::quiet_NaN(),0,0},{1,0,0},box,{},3.).valid,"Nonfinite actual body must reject");
    const std::vector<std::pair<Eigen::Vector3d,double>> farther{{{4.,0,0},.5}};
    require(pelagos::cruiseDirectionClearance(body,{1,0,0},box,farther,3.).valid,"Original actual body can approve direction");
    require(!pelagos::cruiseDirectionClearance({1,0,0},{1,0,0},box,farther,3.).valid,"Later actual body shift must invalidate formerly approved direction");

    Node root,unsafe,unsafe_descendant,safe,safe_descendant,other,back;
    root.state_<<0,0,0,0;
    unsafe.state_<<.3,0,.95,2.;unsafe.gain_=100;unsafe.parent_=&root;
    unsafe_descendant.state_<<1,1,1,1.;unsafe_descendant.gain_=500;unsafe_descendant.parent_=&unsafe;
    unsafe.children_={&unsafe_descendant};
    safe.state_<<.7,.7,0,-2.;safe.gain_=10;safe.parent_=&root;
    safe_descendant.state_<<1.4,1.4,0,3.;safe_descendant.gain_=50;safe_descendant.parent_=&safe;
    safe.children_={&safe_descendant};
    other.state_<<.7,-.7,0,.2;other.gain_=40;other.parent_=&root;
    back.state_<<-1,0,0,0;back.gain_=2000;back.parent_=&root;
    root.children_={&unsafe,&safe,&other,&back};
    int eligible=0;
    const std::function<bool(const Eigen::Vector3d&)> allowed=[&](const Eigen::Vector3d& d) {
      return pelagos::cruiseDirectionClearance(body,d,box,at_boundary,3.).valid;
    };
    require(pelagos::selectForwardNode(&root,{1,0,0},0.,&eligible,allowed)==&safe_descendant,
      "Choose greatest original gain among actual branches whose first edge is cruisable");
    require(eligible==3,"Unsafe first edge must reject its high-gain descendant too");
    require(safe_descendant.parent_==&safe&&safe.parent_==&root,"Selected result must remain in the actual tree branch");
    require(unsafe_descendant.gain_==500&&safe_descendant.gain_==50&&safe.state_[3]==-2.,"Filtering must not alter source gain or sampled view yaw");
    require(pelagos::selectForwardNode(&root,{1,0,0},0.)==&unsafe_descendant,"Normal selection without turn predicate must retain prior behavior");
    require(pelagos::selectForwardNode(&root,{0,0,0},0.,nullptr,allowed)==&back,"Turn filtering may work without adding a forward task constraint");
    const std::function<bool(const Eigen::Vector3d&)> blocked=[&](const Eigen::Vector3d& d) {
      const Eigen::Vector3d flat(d.x(),d.y(),0.);
      const std::vector<std::pair<Eigen::Vector3d,double>> wall{{2.*flat.normalized(),.5}};
      return pelagos::cruiseDirectionClearance(body,d,box,wall,3.).valid;
    };
    require(!pelagos::selectForwardNode(&root,{1,0,0},0.,nullptr,blocked),"No cruisable actual branch must yield no node, never a synthetic route");
    std::cout<<"PASS: strict3m boundary, actual bodyZ/origin, unknown no-hit distinction, late clearance invalidation, actual first-branch filtering, original gain/yaw, normal selection retention, no-candidate\n";
    return 0;
  }catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}

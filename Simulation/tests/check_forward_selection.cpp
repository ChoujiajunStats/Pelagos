#include <iostream>
#include <stdexcept>
#include "forward_selection.hpp"
struct Node {Eigen::Vector4d state_;Node* parent_=nullptr;std::vector<Node*> children_;double gain_=0;};
void require(bool value,const char* text) {if(!value)throw std::runtime_error(text);}
int main() {
  try {
    Node root,back,reverse_view,forward,descendant,lateral;
    root.state_<<0,0,0,0;
    back.state_<<-1,0,0,0;back.gain_=100;back.parent_=&root;
    reverse_view.state_<<1,0,0,std::acos(-1.);reverse_view.gain_=200;reverse_view.parent_=&root;
    forward.state_<<1,.3,0,.2;forward.gain_=5;forward.parent_=&root;
    descendant.state_<<2,.5,0,.3;descendant.gain_=10;descendant.parent_=&forward;forward.children_={&descendant};
    lateral.state_<<0,1,0,0;lateral.gain_=300;lateral.parent_=&root;
    root.children_={&back,&reverse_view,&forward,&lateral};int eligible=0;
    require(pelagos::selectForwardNode(&root,{1,0,0},0.,&eligible)==&reverse_view,"Choose highest original gain among position-forward branches regardless of sampled view");
    require(eligible==3,"Positive forward translation must retain reverse-view source nodes");
    require(reverse_view.state_[3]==std::acos(-1.),"Task selection must preserve source-sampled yaw");
    root.children_={&back,&lateral};
    require(!pelagos::selectForwardNode(&root,{1,0,0},0.),"No admissible source branch must yield no path");
    require(back.gain_==100&&reverse_view.gain_==200&&descendant.gain_==10,"Task selection must not modify gains");
    Node turned_ahead,old_forward;
    turned_ahead.state_<<-.2,-1,0,1.;turned_ahead.gain_=20;turned_ahead.parent_=&root;
    old_forward.state_<<1,.1,0,0.;old_forward.gain_=1000;old_forward.parent_=&root;
    root.children_={&turned_ahead,&old_forward};
    require(pelagos::selectForwardNode(&root,{0,-1,0},0.)==&turned_ahead,
      "After a turn, forward selection must follow current heading rather than original +X");
    std::cout<<"PASS: forward position filter, reverse-view retention, source yaw/gain unchanged, no-path result\n";
    return 0;
  }catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

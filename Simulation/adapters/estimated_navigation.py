"""Project neutral, pure-turn and NBV intents through the source tracker."""
import copy
import json
import math


def neutral_navigation_tracker(source_class):
    """Retain source leases/guidance, with explicit NBV execution stages."""
    class NeutralNavigationTracker(source_class):
        def _heading_tick(self, goal):
            pose=self.state.pose.pose
            target=goal.target.pose
            values=[getattr(point,key) for point in (pose.position,target.position)
                    for key in ('x','y','z')]
            values += [getattr(q,key) for q in (pose.orientation,target.orientation)
                       for key in ('x','y','z','w')]
            if not all(math.isfinite(value) for value in values):
                return super().tick()
            q=pose.orientation
            actual_yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            # Geometry belongs to the path, not its current authorization.
            # A renewed token must not turn a small endpoint overshoot into a
            # new reverse-facing leg. New paths must get a new real index.
            identity=(goal.run_id,goal.mission_id,goal.epoch,goal.waypoint_index)
            if getattr(self,'_heading_identity',None)!=identity:
                self._heading_identity=identity;self._heading_phase='align'
                dx=target.position.x-pose.position.x;dy=target.position.y-pose.position.y
                self._heading_yaw=(math.atan2(dy,dx) if math.hypot(dx,dy)>
                                   self.p['position_tolerance_m'] else actual_yaw)
                self.dwell=None;self.braking=False
            authority=(goal.generation,goal.token)
            if getattr(self,'_heading_authority',None)!=authority:
                self._heading_authority=authority;self.dwell=None;self.braking=False
            if self._heading_phase=='translate':
                error=math.atan2(math.sin(self._heading_yaw-actual_yaw),
                                 math.cos(self._heading_yaw-actual_yaw))
                if abs(error)>self.p['yaw_tolerance_rad']:
                    self._heading_phase='align';self.dwell=None;self.braking=False
            phase=self._heading_phase
            temporary=copy.deepcopy(goal)
            if phase=='align':
                temporary.waypoint_index=1
                temporary.target.pose.position=copy.deepcopy(pose.position)
            elif phase=='translate':temporary.waypoint_index=2
            if phase!='observe':
                orientation=temporary.target.pose.orientation
                orientation.x=orientation.y=0.
                orientation.z=math.sin(self._heading_yaw/2.)
                orientation.w=math.cos(self._heading_yaw/2.)
            original_status=self.status;reports=[]
            class PhaseStatus:
                def publish(_,message):
                    report=json.loads(message.data);reports.append(report.copy())
                    report.update(execution_phase=phase,source_waypoint_index=goal.waypoint_index,
                                  segment_yaw_rad=self._heading_yaw,
                                  phase_reached=bool(report.get('reached')))
                    # Only the original view pose may finish a real NBV edge.
                    if phase!='observe':report['reached']=False
                    original_status.publish(type(message)(data=json.dumps(report)))
            self.goal=temporary;self.status=PhaseStatus()
            try:
                result=super().tick()
            finally:
                self.status=original_status
                # Respect source invalid-state/exception handling; never revive
                # a goal that the source cleared while processing the clone.
                if self.goal is temporary:self.goal=goal
            if (reports and reports[-1].get('state')=='TRACKING' and reports[-1].get('reached') and
                    reports[-1].get('generation')==goal.generation and
                    reports[-1].get('waypoint_index')==temporary.waypoint_index):
                if phase=='align':self._heading_phase='translate'
                elif phase=='translate':self._heading_phase='observe'
                if phase!='observe':self.dwell=None;self.braking=False
            return result

        def tick(self):
            goal = self.goal
            if (self.cfg.get('planner_profile') != 'ethz_nbvplanner' or
                    goal is None or self.state is None):
                return super().tick()
            if goal.waypoint_index>=12:
                # Reactive cruise goals roll ahead continuously after the bridge
                # has completed its separate pure turn. Retain source guidance,
                # rather than treating each rolling target as a stopped viewpoint.
                if goal.mission_id=='nbv-reactive':return super().tick()
                return self._heading_tick(goal)
            if goal.waypoint_index not in (0,1):return super().tick()
            original_pose = goal.target.pose
            goal.target.pose = copy.deepcopy(self.state.pose.pose)
            if goal.waypoint_index==1:
                # Turn in place at the latest estimated XYZ every source tick;
                # only the externally requested yaw survives pose substitution.
                goal.target.pose.orientation=copy.deepcopy(original_pose.orientation)
            try:
                return super().tick()
            finally:
                # The source may clear self.goal on invalid state; restore the
                # exact original goal object rather than whichever goal is now active.
                goal.target.pose = original_pose

    return NeutralNavigationTracker


def main(args=None):
    import uw_navigation.node as navigation
    navigation.Tracker = neutral_navigation_tracker(navigation.Tracker)
    navigation.main(args=args)


if __name__ == '__main__':
    main()

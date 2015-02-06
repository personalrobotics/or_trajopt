#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015, Carnegie Mellon University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of Carnegie Mellon University nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

###############################################################################
############################## Notes about trajopt ############################
###############################################################################

# 1) The collision cost of type "collision", if the "continous" field is missing or set to True 
#    does not perform self-collision checking. 
#    If one adds the discrete version, it checks the self-collisions but it is slower (approx doubles the planning time!)
#    In case add this to the cost function
#    {
#       "type" : "collision",
#       "params" : {"coeffs" : [20],"dist_pen" : [0.025], "first_step":0, "last_step":19, "continuous":False}
#    }
#    In most of the python examples, there is just the continous term
# 2) In the "checkConstraints" function, we check collisions and pose constraints. 
#    If the planner founds a solution, the pose constraints should be satisfied since it is an hard constraint
#    in the optimization formulation. Regarding the collision, it is a cost term. So, we check just a final refinement.

from prpy.planning.base import (BasePlanner,
                                PlanningError,
                                PlanningMethod)
import logging
import numpy
import time
import openravepy
import prpy.ik_ranking
from . import constraints

logger = logging.getLogger(__name__)


class TrajoptPlanner(BasePlanner):
    def __init__(self):
        """
        Create a PrPy binding to the Trajopt motion optimization package.

        Instantiates a PrPy planner that calls Trajopt to perform various
        planning operations.
        """
        super(TrajoptPlanner, self).__init__()

    def __str__(self):
        return 'Trajopt'

    def _Plan(self, robot, request, interactive=False,
              constraint_threshold=1e-4, **kwargs):
        """
        Plan to a desired configuration with Trajopt.

        This function invokes the Trajopt planner directly on the specified
        JSON request. This can be used to implement custom path optimization
        algorithms.

        @param robot the robot whose active DOFs will be used
        @param request a JSON planning request for Trajopt
        @param interactive pause every iteration, until you press 'p'
                              or press escape to disable further plotting
        @param constraint_threshold acceptable per-constraint violation error
        @return traj a trajectory from current configuration to specified goal
        """
        import json
        import trajoptpy

        # Set up environment.
        env = robot.GetEnv()
        trajoptpy.SetInteractive(interactive)

        # Convert dictionary into json-formatted string and create object that
        # stores optimization problem.
        s = json.dumps(request)
        prob = trajoptpy.ConstructProblem(s, env)

        # Perform trajectory optimization.
        t_start = time.time()
        result = trajoptpy.OptimizeProblem(prob)
        t_elapsed = time.time() - t_start
        logger.info("Optimization took {:.3f} seconds".format(t_elapsed))

        # Check for constraint violations.
        for name, error in result.GetConstraints():
            if error > constraint_threshold:
                raise PlanningError(
                    "Trajectory violates contraint '{:s}': {:f} > {:f}"
                    .format(name, error, constraint_threshold)
                )

        # Check for the returned trajectory.
        waypoints = result.GetTraj()
        if waypoints is None:
            raise PlanningError("Trajectory result was empty.")

        # Verify the trajectory and return it as a result.
        from trajoptpy.check_traj import traj_is_safe
        with env:
            # Set robot DOFs to DOFs in optimization problem
            prob.SetRobotActiveDOFs()

            # Check that trajectory is collision free
            if not traj_is_safe(waypoints, robot):
                raise PlanningError("Result was in collision.")

        # Convert the waypoints to a trajectory.
        return self._WaypointsToTraj(robot, waypoints)

    @PlanningMethod
    def PlanToConfiguration(self, robot, goal, **kwargs):
        """
        Plan to a desired configuration with Trajopt.

        @param robot the robot whose active DOFs will be used
        @param goal the desired robot joint configuration
        @param is_interactive pause every iteration, until you press 'p'
                              or press escape to disable further plotting
        @return traj a trajectory from current configuration to specified goal
        """
        # Auto-cast to numpy array if this was a list.
        goal = numpy.array(goal)

        request = {
            "basic_info": {
                "n_steps": 10,
                "manip": "active",
                "start_fixed": True
            },
            "costs": [
                {
                    "type": "joint_vel",
                    "params": {"coeffs": [1]}
                },
                {
                    "type": "collision",
                    "params": {
                        "coeffs": [20],
                        "dist_pen": [0.025]
                    },
                }
            ],
            "constraints": [
                {
                    "type": "joint",
                    "params": {"vals": goal.tolist()}
                }
            ],
            "init_info": {
                "type": "straight_line",
                "endpoint": goal.tolist()
            }
        }
        return self._Plan(robot, request, **kwargs)

    @PlanningMethod
    def PlanToIK(self, robot, pose, **kwargs):
        """
        Plan to a desired end effector pose with Trajopt.

        An IK ranking function can optionally be specified to select a
        preferred IK solution from those available at the goal pose.

        @param robot the robot whose active manipulator will be used
        @param pose the desired manipulator end effector pose
        @param ranker an IK ranking function to use over the IK solutions
        @param is_interactive pause every iteration, until you press 'p'
                              or press escape to disable further plotting
        @return traj a trajectory from current configuration to specified pose
        """
        self._PlanToIK(robot, pose, **kwargs)

    @PlanningMethod
    def PlanToEndEffector(self, robot, pose, **kwargs):
        """
        Plan to a desired end effector pose with Trajopt.

        This function is internally implemented identically to PlanToIK().

        @param robot the robot whose active manipulator will be used
        @param pose the desired manipulator end effector pose
        @param is_interactive pause every iteration, until you press 'p'
                              or press escape to disable further plotting
        @return traj a trajectory from current configuration to specified pose
        """
        self._PlanToIK(robot, pose, **kwargs)

    def _PlanToIK(self, robot, pose,
                  ranker=prpy.ik_ranking.JointLimitAvoidance, **kwargs):
        # Plan using the active manipulator.
        manipulator = robot.GetActiveManipulator()
        robot.SetActiveDOFs(manipulator.GetArmIndices())

        # Find initial collision-free IK solution.
        from openravepy import (IkFilterOptions,
                                IkParameterization,
                                IkParameterizationType)
        ik_param = IkParameterization(
            pose, IkParameterizationType.Transform6D)
        ik_solutions = manipulator.FindIKSolutions(
            ik_param, IkFilterOptions.CheckEnvCollisions)
        if not len(ik_solutions):
            raise PlanningError('No collision-free IK solution.')

        # Sort the IK solutions in ascending order by the costs returned by the
        # ranker. Lower cost solutions are better and infinite cost solutions
        # are assumed to be infeasible.
        scores = ranker(robot, ik_solutions)
        best_idx = numpy.argmax(scores)
        init_joint_config = ik_solutions[best_idx]

        # Convert IK endpoint transformation to pose.
        goal_position = pose[0:3, 3].tolist()
        goal_rotation = openravepy.quatFromRotationMatrix(pose).tolist()

        # Construct a planning request with these constraints.
        request = {
            "basic_info": {
                "n_steps": 10,
                "manip": "active",
                "start_fixed": True
            },
            "costs": [
                {
                    "type": "joint_vel",
                    "params": {"coeffs": [1]}
                },
                {
                    "type": "collision",
                    "params": {
                        "coeffs": [20],
                        "dist_pen": [0.025]
                    },
                }
            ],
            "constraints": [
                {
                    "type": "pose",
                    "params": {
                        "xyz": goal_position,
                        "wxyz": goal_rotation,
                        "link": manipulator.GetEndEffector().GetName(),
                        "timestep": 9
                    }
                }
            ],
            "init_info": {
                "type": "straight_line",
                "endpoint": init_joint_config.tolist()
            }
        }
        return self._Plan(robot, request, **kwargs)

    @PlanningMethod
    def PlanToTSR(self, robot, tsrchains, is_interactive=False, **kw_args):
        """
        Plan using the given list of TSR chains with TrajOpt.

        @param robot the robot whose active manipulator will be used
        @param tsrchains a list of TSRChain objects to respect during planning
        @param is_interactive pause every iteration, until you press 'p'
                              or press escape to disable further plotting
        @return traj
        """
        import json
        import time
        import trajoptpy

        manipulator = robot.GetActiveManipulator()
        n_steps = 10
        n_dofs = robot.GetActiveDOF()

        # Create separate lists for the goal and trajectory-wide constraints.
        goal_tsrs = [t for t in tsrchains if t.sample_goal]
        traj_tsrs = [t for t in tsrchains if t.constrain]

        # Find an initial collision-free IK solution by sampling goal TSRs.
        from openravepy import (IkFilterOptions,
                                IkParameterization,
                                IkParameterizationType)
        for tsr in self._TsrSampler(goal_tsrs):
            ik_param = IkParameterization(
                tsr.sample(), IkParameterizationType.Transform6D)
            init_joint_config = manipulator.FindIKSolutions(
                ik_param, IkFilterOptions.CheckEnvCollisions)
            if init_joint_config:
                break
        if not init_joint_config:
            raise PlanningError('No collision-free IK solutions.')

        # Construct a planning request with these constraints.
        request = {
            "basic_info": {
                "n_steps": n_steps,
                "manip": "active",
                "start_fixed": True
            },
            "costs": [
                {
                    "type": "joint_vel",
                    "params": {"coeffs": [1]}
                },
                {
                    "type": "collision",
                    "params": {
                        "coeffs": [20],
                        "dist_pen": [0.025]
                    },
                }
            ],
            "constraints": [],
            "init_info": {
                "type": "straight_line",
                "endpoint": init_joint_config.tolist()
            }
        }

        # Set up environment.
        env = robot.GetEnv()
        trajoptpy.SetInteractive(is_interactive)

        # Convert dictionary into json-formatted string and create object that
        # stores optimization problem.
        s = json.dumps(request)
        prob = trajoptpy.ConstructProblem(s, env)
        for t in xrange(1, n_steps):
            prob.AddConstraint(constraints._TsrCostFn(robot, traj_tsrs),
                               [(t, j) for j in xrange(n_dofs)],
                               "EQ", "up{:d}".format(t))
        prob.AddConstraint(constraints._TsrCostFn(robot, goal_tsrs),
                           [(n_steps-1, j) for j in xrange(n_dofs)],
                           "EQ", "up{:d}".format(t))

        # Perform trajectory optimization.
        t_start = time.time()
        result = trajoptpy.OptimizeProblem(prob)
        t_elapsed = time.time() - t_start
        logger.info("Optimization took {:.3f} seconds".format(t_elapsed))

        # Check for constraint violations.
        if result.GetConstraints():
            raise PlanningError("Trajectory did not satisfy constraints: {:s}"
                                .format(str(result.GetConstraints())))

        # Check for the returned trajectory.
        waypoints = result.GetTraj()
        if waypoints is None:
            raise PlanningError("Trajectory result was empty.")

        # Verify the trajectory and return it as a result.
        from trajoptpy.check_traj import traj_is_safe
        with env:
            # Set robot DOFs to DOFs in optimization problem
            prob.SetRobotActiveDOFs()

            # Check that trajectory is collision free
            if not traj_is_safe(waypoints, robot):
                return PlanningError("Result was in collision.")
        return self._WaypointsToTraj(robot, waypoints)

    @PlanningMethod
    def PlanToEndEffectorOffset(self, robot, direction,
                                min_distance, max_distance=None,
                                position_tolerance=0.01,
                                angular_tolerance=0.15, **kwargs):
        """
        Plan to an end-effector offset with a move-hand-straight constraint.

        This function plans a trajectory to purely translate a hand in the
        given direction as far as possible before collision. Movement less
        than min_distance will return failure. The motion will not also move
        further than max_distance.

        @param robot the robot whose active DOFs will be used
        @param direction unit vector in the direction of motion
        @param distance minimum distance in meters
        @param max_distance maximum distance in meters
        @param position_tolerance constraint tolerance in meters
        @param angular_tolerance constraint tolerance in radians
        @return traj a trajectory following specified direction
        """
        # Plan using the active manipulator.
        manipulator = robot.GetActiveManipulator()
        robot.SetActiveDOFs(manipulator.GetArmIndices())
        pose = manipulator.GetEndEffectorTransform()

        # Convert IK endpoint transformation to pose.
        ee_position = pose[0:3, 3].tolist()
        ee_rotation = openravepy.quatFromRotationMatrix(pose).tolist()

        # Create dummy frame that has direction vector along Z.
        z = direction
        y = numpy.cross(z, pose[0:3, 0])  # Cross with EE-frame Y.
        y /= numpy.linalg.norm(y)
        x = numpy.cross(y, z)

        cost_frame = numpy.vstack((x, y, z)).T
        cost_rotation = openravepy.quatFromRotationMatrix(cost_frame).tolist()

        # Construct a planning request with these constraints.
        n_steps = 10
        request = {
            "basic_info": {
                "n_steps": n_steps,
                "manip": "active",
                "start_fixed": True
            },
            "costs": [
                {
                    "type": "joint_vel",
                    "params": {"coeffs": [1]}
                },
                {
                    "type": "collision",
                    "params": {
                        "coeffs": [20],
                        "dist_pen": [0.025]
                    },
                },
                {
                    "type": "pose",
                    "params": {
                        "xyz": [0, 0, 0],
                        "wxyz": cost_rotation,
                        "link": manipulator.GetEndEffector().GetName(),
                        "rot_coeffs": [0, 0, 0],
                        "pos_coeffs": [0, 0, 10]
                    }
                }
            ],
            "constraints": [
                {
                    "type": "pose",
                    "params": {
                        "xyz": ee_position,
                        "wxyz": ee_rotation,
                        "link": manipulator.GetEndEffector().GetName(),
                        # "first_step": 0,
                        # "last_step": n_steps-1,
                    }
                }
            ],
            "init_info": {
                "type": "stationary"
            }
        }
        return self._Plan(robot, request, **kwargs)

    def OptimizeTrajectory(self, robot, traj,
                           distance_penalty=0.025, **kwargs):
        """
        Optimize an existing feasible trajectory using TrajOpt.

        @param robot the robot whose active DOFs will be used
        @param traj the original trajectory that will be optimized
        @param distance_penalty the penalty for approaching obstacles
        @param is_interactive pause every iteration, until you press 'p'
                              or press escape to disable further plotting
        @return traj a trajectory from current configuration to specified goal
        """
        if not traj.GetNumWaypoints():
            raise ValueError("Cannot optimize empty trajectory.")

        # Extract joint positions from trajectory.
        cspec = traj.GetConfigurationSpecification()
        n_waypoints = traj.GetNumWaypoints()
        dofs = robot.GetActiveDOFIndices()
        waypoints = [cspec.ExtractJointValues(traj.GetWaypoint(i),
                                              robot, dofs).tolist()
                     for i in range(n_waypoints)]

        request = {
            "basic_info": {
                "n_steps": n_waypoints,
                "manip": "active",
                "start_fixed": True
            },
            "costs": [
                {
                    "type": "joint_vel",
                    "params": {"coeffs": [1]}
                },
                {
                    "type": "collision",
                    "params": {
                        "coeffs": [20],
                        "dist_pen": [distance_penalty]
                    },
                }
            ],
            "constraints": [
                {
                    "type": "joint",
                    "params": {"vals": waypoints[-1]}
                }
            ],
            "init_info": {
                "type": "given_traj",
                "data": waypoints
            }
        }
        return self._Plan(robot, request, **kwargs)

    def _WaypointsToTraj(self, robot, waypoints):
        """Converts a list of waypoints to an OpenRAVE trajectory."""
        traj = openravepy.RaveCreateTrajectory(robot.GetEnv(), '')
        traj.Init(robot.GetActiveConfigurationSpecification('linear'))

        for (i, waypoint) in enumerate(waypoints):
            traj.Insert(i, waypoint)
        return traj


    ###########################################################################################
    ##################################### MC functions ########################################
    ###########################################################################################
    def _PlanWithTolerance( self, robot, request, time_elapsed = [],
                            ee_pos_vec = [], ee_ori_vec = [], interactive=False,
                            cnt_tolerance=1e-5, position_tolerance=1e-5, angular_tolerance=1e-5, **kwargs):
        """
        Plan to a desired configuration with TrajOpt. This will invoke the
        TrajOpt planner on the specified JSON request.
        @param robot the robot whose active DOFs will be used
        @param request a JSON planning request for TrajOpt
        @param [Optional] time_elapsed list that exports the time needed for the optimization process
        @param [Optional] ee_pos_vec vector of desired position. Needed if one wants to check the position constraint
        @param [Optional] ee_ori_vec vector of desired orientation (quaternion). Needed if one wants to check the orientation constraint
        @param [Optional] cnt_tolerance tolerance for violation of trajopt constraint
        @param [Optional] position_tolerance tolerance for position constraints
        @param [Optional] angular_tolerance tolerance for orientation constraints
        @param interactive pause every iteration, until you press 'p'
                              or press escape to disable further plotting.
        @return traj a trajectory from current configuration to specified goal
        """
        import json
        import time
        import trajoptpy

        # Set up environment.
        env = robot.GetEnv()
        trajoptpy.SetInteractive(interactive)

        # Convert dictionary into json-formatted string and create object that
        # stores optimization problem.
        s = json.dumps(request)
        prob = trajoptpy.ConstructProblem(s, env)

        # Perform trajectory optimization.
        t_start = time.time()
        result = trajoptpy.OptimizeProblem(prob)
        t_elapsed = time.time() - t_start

        # Saving, if needed, time elapsed
        time_elapsed.append(t_elapsed)

        logger.info("Optimization took {:.3f} seconds".format(t_elapsed))

        # Check for the returned trajectory.
        waypoints = result.GetTraj()
        
        if waypoints is None or waypoints.size == 0:
            raise PlanningError("Trajectory result was empty.")

        # Check for constraint violations (for only pose or pose_multiple constraints)
        resultCnt = result.GetConstraints()
        for i in range(0,len(resultCnt)):
            if resultCnt[i][0].find("pose") == 0:
                if resultCnt[i][1] > cnt_tolerance:
                    raise PlanningError("Trajectory did not satisfy constraints: {:s}"
                                    .format(str(resultCnt[i])))
                else:
                    print '[or_trajopt] Warning: a constraint has an error smaller than the tolerance: cnt_name', resultCnt[i][0], " cnt_value: ", resultCnt[i][1], " tolerance: ", cnt_tolerance, "line: ", i, "\n"

        # Check explicitly the contraints (not just seeing if constraint violation is respected)
        if len(ee_pos_vec) > 0 and len(ee_ori_vec) > 0:
            manip = robot.GetActiveManipulator()
            for wpIter in range(0,len(waypoints)):
                # Distinguish the case of single input or vector of inputs
                if len(ee_pos_vec) == 3 and len(ee_ori_vec) == 4:
                    # If it is a single input, the checking is needed only for the final configuration
                    # The previous ones are already checked from previous iterations
                    des_pos = ee_pos_vec
                    des_ori = ee_ori_vec
                    wpIter = len(waypoints)-1
                else:
                    des_pos = ee_pos_vec[wpIter]
                    des_ori = ee_ori_vec[wpIter]
                self.checkConstraints(robot, manip, waypoints[wpIter], des_pos, des_ori, position_tolerance, angular_tolerance)

        # Verify the trajectory and return it as a result.
        from trajoptpy.check_traj import traj_is_safe
        with env:
            # Set robot DOFs to DOFs in optimization problem
            prob.SetRobotActiveDOFs()

            # Check that trajectory is collision free
            if not traj_is_safe(waypoints, robot):
                raise PlanningError("Result was in collision.")

        # Convert the waypoints to a trajectory.
        return self._WaypointsToTraj(robot, waypoints)

    def checkConstraints(self, robot, manip, waypoint, des_pos, des_ori, position_tolerance, angular_tolerance):
        """
        Check the constraints for the angular and the position constraints for a trajopt problem
        @param robot
        @param manip
        @param waypoint robot configuration to be tested
        @param des_pos desired position for the end effector 
        @param des_ori desired orientation for the end effector (quaternion)
        @param position_tolerance constraint tolerance in meters
        @param angular_tolerance constraint tolerance in radians
        @return void. It raises a PlanningError in case of failure
        """

        # Set the robot and get the actual EE-pose (only for active fields)
        # Check if it is robust
        active_dof_indices = robot.GetActiveDOFIndices()
        robot.SetDOFValues(waypoint, active_dof_indices)
        #robot.SetDOFValues(waypoint)

        # Joint limit check
        limits_lower, limits_upper = robot.GetDOFLimits(active_dof_indices)
        check_low_lim = waypoint < limits_lower
        check_up_lim  = waypoint > limits_upper
        if check_low_lim.any() and check_up_lim.any():
            raise PlanningError('Exceeding joint limits.')
        
        # Getting ee information
        ee_pose = manip.GetEndEffectorTransform()
        ee_pos  = ee_pose[0:3,3]

        # Position contraint check
        pos_err = abs(ee_pos - des_pos)
        if numpy.linalg.norm(pos_err) > position_tolerance:
            raise PlanningError('Deviated from a straight line constraint (position).')

        # Orientation constraint check
        currQuat  = openravepy.quatFromRotationMatrix(ee_pose)

        choices_ori = [ currQuat - des_ori, currQuat + des_ori ]
        error_ori1  =  1.0*min(choices_ori, key=lambda q: numpy.linalg.norm(q))
        error_ori2  = -1.0*min(choices_ori, key=lambda q: numpy.linalg.norm(q))

        error_ori1_norm = numpy.linalg.norm(error_ori1)
        error_ori2_norm = numpy.linalg.norm(error_ori2)

        if(error_ori1_norm>angular_tolerance and error_ori2_norm>angular_tolerance):
            raise PlanningError('Deviated from the orientation constraint (orientation).')

    @PlanningMethod
    def PlanToEndEffectorOffsetTO(self, robot, direction, distance, max_distance=None,
                                timelimit=5.0, step_size=0.01,
                                position_tolerance=0.01, angular_tolerance=0.15, cnt_tolerance=1e-6, **kw_args):
        """
        Plan to a desired end-effector offset with move-hand-straight
        constraint. movement less than distance will return failure. The motion
        will not move further than max_distance.
        Note: this method assume that the active robot joints are already set. Check within the function for 
        disabling it internally
        @param robot
        @param direction unit vector in the direction of motion
        @param distance minimum distance in meters
        @param max_distance maximum distance in meters
        @param timelimit timeout in seconds
        @param stepsize step size in meters for the Jacobian pseudoinverse controller
        @param position_tolerance constraint tolerance in meters
        @param angular_tolerance constraint tolerance in radians
        @param cnt_tolerance tolerance on contraint violation from trajopt
        @return traj
        """

        if distance < 0:
            raise ValueError('Distance must be non-negative.')
        elif numpy.linalg.norm(direction) == 0:
            raise ValueError('Direction must be non-zero')
        elif max_distance is not None and max_distance < distance:
            raise ValueError('Max distance is less than minimum distance.')
        elif step_size <= 0:
            raise ValueError('Step size must be positive.')
        elif position_tolerance < 0:
            raise ValueError('Position tolerance must be non-negative.')
        elif angular_tolerance < 0:
            raise ValueError('Angular tolerance must be non-negative.')
        elif cnt_tolerance < 0:
            raise ValueError('Constraint tolerance must be non-negative.')    
        elif timelimit <= 0:
            raise ValueError('Time limit must be positive.')    

        # If the step size exceeds the distance to cover, try to perform a single step 
        if step_size>distance:
            step_size = distance

        # It uses the active manipulator
        # If one wants to set an active manipulator -> robot.SetActiveManipulator('left'/'right'/'head')
        manip = robot.GetActiveManipulator()

        # The following lines constraints the DOF of the robot to the selected manipulator
        #active_dof_indices = manip.GetArmIndices()
        #robot.SetActiveDOFs(active_dof_indices)

        # Here I'm taking statically the right_arm end-effect        
        ee_name      = manip.GetEndEffector().GetName()
        init_ee_pose = manip.GetEndEffectorTransform()
        init_ee_pos  = init_ee_pose[0:3,3]


        # Normalize the direction vector.
        direction  = numpy.array(direction, dtype='float')
        direction /= numpy.linalg.norm(direction)

        # Default to moving an exact distance.
        if max_distance is None:
            max_distance = distance

        n_steps = int(max_distance / step_size) + 1
        print "n_steps: ", n_steps

        import numpy as ny
        ee_pos_vec = init_ee_pos
        start_ori  = openravepy.quatFromRotationMatrix(init_ee_pose)
        ee_ori_vec = start_ori;
        new_ori    = ny.array([0,0,0,0])

        for i in range(0, n_steps-1):
            ee_pos_vec   = ny.vstack((ee_pos_vec, init_ee_pos+(i+1)*direction*step_size))
            # Keep the starting orientation
            ee_ori_vec = ny.vstack((ee_ori_vec, start_ori))

        import json
        ee_pos_vec_json = json.loads(json.dumps(ee_pos_vec.tolist()))
        ee_ori_vec_json = json.loads(json.dumps(ee_ori_vec.tolist()))

        # debug
        #n_steps = 3

        request = {
            "basic_info": {
                "n_steps": n_steps,
                "manip": "active",
                "start_fixed": True,
                "max_time": timelimit
            },
            "costs": [
                {
                    "type": "joint_vel",
                    "params": {"coeffs": [1]}
                },
                {
                    "type": "collision",
                    "params": {
                        "coeffs": [20],
                        "dist_pen": [0.025],
                        "first_step":0, 
                        "last_step":n_steps-1,
                        "continuous":True
                    },
                    
                },
                # Check if one wants to add the following cost: The true does not check for self-collision
                #{
                #    "type": "collision",
                #    "params": {
                #        "coeffs": [20],
                #        "dist_pen": [0.025],
                #        "first_step":0, 
                #        "last_step":n_steps-1,
                #        "continuous":False
                #    },    
                #}
            ],
            "constraints": [
                {
                  "type" : "pose_multiple",
                  "params" : {
                      "first_step" : 0,
                      "last_step"  : n_steps-1, #inclusive (It seems on the last sample it has no effect. Check if to set this checking
                                                # in the C++ code of trajopt)
                      "xyz"  : ee_pos_vec_json,
                      "wxyz" : ee_ori_vec_json,
                      "pos_coeffs" : [1,1,1],
                      "rot_coeffs" : [1,1,1],
                      "link" : ee_name,
                  },
                },
            ],
            "init_info": {
                "type": "stationary"
            }
        }
        return self._PlanWithTolerance(robot, request, ee_pos_vec=ee_pos_vec, ee_ori_vec=ee_ori_vec,
            cnt_tolerance=cnt_tolerance, position_tolerance=position_tolerance,
            angular_tolerance=angular_tolerance,**kw_args)

    @PlanningMethod
    def PlanToEndEffectorOffsetMultiProblem(self, robot, direction, distance, max_distance=None,
                                timelimit=5.0, step_size=0.01,
                                position_tolerance=0.01, angular_tolerance=0.15, cnt_tolerance=1e-5, **kw_args):
        """
        Plan to a desired end-effector offset with move-hand-straight
        constraint. Movement less than distance will return failure. The motion
        will not move further than max_distance.
        This function builts an optimization problem for each step_size.
        @param robot
        @param direction unit vector in the direction of motion
        @param distance minimum distance in meters
        @param max_distance maximum distance in meters
        @param timelimit timeout in seconds
        @param stepsize step size in meters for the Jacobian pseudoinverse controller
        @param position_tolerance constraint tolerance in meters
        @param angular_tolerance constraint tolerance in radians
        @param cnt_tolerance tolerance on contraint violation from trajopt
        @return traj
        """

        if distance < 0:
            raise ValueError('Distance must be non-negative.')
        elif numpy.linalg.norm(direction) == 0:
            raise ValueError('Direction must be non-zero')
        elif max_distance is not None and max_distance < distance:
            raise ValueError('Max distance is less than minimum distance.')
        elif step_size <= 0:
            raise ValueError('Step size must be positive.')
        elif position_tolerance < 0:
            raise ValueError('Position tolerance must be non-negative.')
        elif angular_tolerance < 0:
            raise ValueError('Angular tolerance must be non-negative.')
        elif cnt_tolerance < 0:
            raise ValueError('Constraint tolerance must be non-negative.') 
        elif timelimit <= 0:
            raise ValueError('Time limit must be positive.')    

        # If the step size exceeds the distance to cover, try to perform a single step 
        if step_size>distance:
            step_size = distance

        # It uses the active manipulator
        # If one wants to set an active manipulator -> robot.SetActiveManipulator('left'/'right'/'head')
        #robot.SetActiveManipulator('right')
        manip = robot.GetActiveManipulator()

        # The following lines constraints the DOF of the robot to the selected manipulator
        #active_dof_indices = manip.GetArmIndices()
        #robot.SetActiveDOFs(active_dof_indices)

        # Here I'm taking statically the right_arm end-effect        
        ee_name      = manip.GetEndEffector().GetName()
        init_ee_pose = manip.GetEndEffectorTransform()
        init_ee_pos  = init_ee_pose[0:3,3]


        # Normalize the direction vector.
        direction  = numpy.array(direction, dtype='float')
        direction /= numpy.linalg.norm(direction)

        # Default to moving an exact distance.
        if max_distance is None:
            max_distance = distance

        n_steps = int(max_distance / step_size) + 1
        print "n_steps: ", n_steps
        # Compute the time elapsed
        t_el = []

        import numpy as ny
        import json

        # Initialize final trajectory
        trajTot = openravepy.RaveCreateTrajectory(robot.GetEnv(), '')
        trajTot.Init(robot.GetActiveConfigurationSpecification())

        # debug
        #n_steps = 2
        trajTot.Insert(0,robot.GetDOFValues())
        start_time = time.time()
        start_ori  = openravepy.quatFromRotationMatrix(init_ee_pose)
        new_ori    = ny.array([0,0,0,0])

        # Split the total time for each iteration
        timeIter   = timelimit / n_steps;

        for i in range(0, n_steps-1):
            print "Iteration: ", i
            # Check time condition. There is a time checker also in the trajopt code now. This is a double check.
            current_time = time.time()
            if timelimit is not None and current_time - start_time > timelimit:
                raise PlanningError('Reached time limit.')

            ee_pos_vec = init_ee_pos + (i+1)*step_size*direction

            # Define the new orientation, if needed (it is the easier solution for orientation with the given input)
            #new_ori = ny.hstack((start_ori[0], start_ori[1:4] + (i+1)*direction*step_size))
            #new_ori /= ny.linalg.norm(new_ori)

            # Orientation constrained to the initial value
            ee_ori_vec = start_ori

            ee_pos_vec_json = list(ee_pos_vec)
            ee_ori_vec_json = list(ee_ori_vec)

            request = {
                "basic_info": {
                    "n_steps": 2,
                    "manip": "active",
                    "start_fixed": True,
                    "max_time": timeIter
                },
                "costs": [
                    {
                        "type": "joint_vel",
                        "params": {"coeffs": [1]}
                    },
                    {
                        "type": "collision",
                        "params": {
                            "coeffs": [20],
                            "dist_pen": [0.025]
                        },
                    }
                ],
                "constraints": [
                    {
                      "type" : "pose",
                      "params" : {
                          "xyz"  : ee_pos_vec_json,
                          "wxyz" : ee_ori_vec_json,
                          "pos_coeffs" : [1,1,1],
                          "rot_coeffs" : [1,1,1],
                          "link" : ee_name,
                      },
                    },
                ],
                "init_info": {
                    "type": "stationary"
                }
            }
            #outPlan = self._PlanWithTolerance(robot, request, time_elapsed=t_el, cnt_tolerance=cnt_tolerance,**kw_args)
            outPlan = self._PlanWithTolerance(robot, request, time_elapsed=t_el, ee_pos_vec=ee_pos_vec, ee_ori_vec=ee_ori_vec, 
                                              cnt_tolerance=cnt_tolerance, position_tolerance=position_tolerance, 
                                              angular_tolerance=angular_tolerance,**kw_args)

            numWaypoints    = outPlan.GetNumWaypoints()
            numTotWaypoints = trajTot.GetNumWaypoints()
            for j in range(0,numWaypoints):
                # Check if you need the starting of each waypoint (i.e. if I'm duplicating some connection waypoints)
                # The following "if" skips to duplicate "conjunction" waypoints.
                # The reason for that is that this function launches an optimization problem for each segment.
                # Then the final solution of one problem is identical to the starting solution of the next problem 
                # If one adds all the solution, these are duplicated. The starting configuration is added before the for loop
                if j>0:
                    waypointsTo = outPlan.GetWaypoint(j)
                    trajTot.Insert(numTotWaypoints+j-1, waypointsTo) # In case of including j=0, it should be numTotWaypoints+j
        #for l in range(0,trajTot.GetNumWaypoints()):
        #    print "WAY[", l, "]: ", repr(trajTot.GetWaypoint(l))

        # Computing total time
        tot_time = 0.0
        for k in range(0, len(t_el)):
            tot_time += t_el[k]
        print "Optimization took: ", tot_time
        return trajTot

    @PlanningMethod
    def PlanToEndEffectorPoseTO(self, robot, goal_pose, timelimit=5.0, n_steps = 10,
                              position_tolerance=0.01, angular_tolerance=0.15, cnt_tolerance=1e-6, **kw_args):
        """
        Plan to a desired end-effector offset with move-hand-straight
        constraint. movement less than distance will return failure. The motion
        will not move further than max_distance.
        Note: this method assume that the active robot joints are already set. Check within the function for 
        disabling it internally
        @param robot
        @ param goal_pose goal pose (expressed as a transformation matrix 4x4)
        @param timelimit timeout in seconds
        @param [Optional] n_steps number of steps for trajopt (default = 10)
        @param position_tolerance constraint tolerance in meters
        @param angular_tolerance constraint tolerance in radians
        @param cnt_tolerance tolerance on contraint violation from trajopt
        @return traj
        """

        if position_tolerance < 0:
            raise ValueError('Position tolerance must be non-negative.')
        elif angular_tolerance < 0:
            raise ValueError('Angular tolerance must be non-negative.')
        elif cnt_tolerance < 0:
            raise ValueError('Constraint tolerance must be non-negative.')    
        elif timelimit <= 0:
            raise ValueError('Time limit must be positive.')    

        # It uses the active manipulator
        # If one wants to set an active manipulator -> robot.SetActiveManipulator('left'/'right'/'head')
        manip = robot.GetActiveManipulator()

        # The following lines constraints the DOF of the robot to the selected manipulator
        #active_dof_indices = manip.GetArmIndices()
        #robot.SetActiveDOFs(active_dof_indices)

        # Here I'm taking statically the right_arm end-effect        
        ee_name      = manip.GetEndEffector().GetName()
        init_ee_pose = manip.GetEndEffectorTransform()
        init_ee_pos  = init_ee_pose[0:3,3]

        import numpy as ny
        ee_pos_vec = init_ee_pos
        start_ori  = openravepy.quatFromRotationMatrix(init_ee_pose)
        ee_ori_vec = start_ori;
        new_ori    = ny.array([0,0,0,0])

        goal_pos  = goal_pose[0:3,3]
        goal_quat = openravepy.quatFromRotationMatrix(goal_pose)

        ee_pos_vec_json = list(goal_pos)
        ee_ori_vec_json = list(goal_quat)

        request = {
            "basic_info": {
                "n_steps": n_steps,
                "manip": "active",
                "start_fixed": True,
                "max_time": timelimit
            },
            "costs": [
                {
                    "type": "joint_vel",
                    "params": {"coeffs": [1]}
                },
                {
                    "type": "collision",
                    "params": {
                        "coeffs": [20],
                        "dist_pen": [0.025]
                    },
                }
            ],
            "constraints": [
                {
                  "type" : "pose",
                  "params" : {
                      "xyz"  : ee_pos_vec_json,
                      "wxyz" : ee_ori_vec_json,
                      "pos_coeffs" : [1,1,1],
                      "rot_coeffs" : [1,1,1],
                      "link" : ee_name,
                  },
                },
            ],
            "init_info": {
                "type": "stationary"
            }
        }
        return self._PlanWithTolerance(robot, request, ee_pos_vec=goal_pos, ee_ori_vec=goal_quat,
            cnt_tolerance=cnt_tolerance, position_tolerance=position_tolerance,
            angular_tolerance=angular_tolerance,**kw_args)

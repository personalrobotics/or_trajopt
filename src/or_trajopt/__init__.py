#!/usr/bin/env python
# -*- coding: utf-8 -*-

from prpy.planning.base import (BasePlanner,
                                PlanningError,
                                PlanningMethod)
import logging
import itertools
import numpy
import openravepy
import time

logger = logging.getLogger(__name__)


class TrajOptPlanner(BasePlanner):
    def __init__(self):
        super(TrajOptPlanner, self).__init__()

    def __str__(self):
        return 'TrajOpt'

    def _TsrSampler(self, tsrchains, timelimit=2.0):
        """Returns a generator that samples TSR chains until the timelimit"""
        # Create an iterator that cycles through each TSR chain.
        tsr_cycler = itertools.cycle(tsrchains)

        # Create an iterator that cycles TSR chains until the timelimit.
        tsr_timelimit = time.time() + timelimit
        return itertools.takewhile(
            lambda v: time.time() < tsr_timelimit, tsr_cycler)

    def _TsrCostFn(self, robot, tsrchain_list):
        """ Generates a cost function for a list of TSR chains.

        This function returns the minimum projected Euclidean distance to the
        closest TSR in a TSR set.
        """
        import prpy.tsr.kin as kin
        XYZYPR_TO_XYZRPY = [0, 1, 2, 5, 4, 3]

        # I don't know how to project onto TSR chains...
        for tsrchain in tsrchain_list:
            if len(tsrchain.TSRs) > 0:
                raise ValueError("Cannot project TSR chain.")
        tsrlist = [tsrchain.TSRs[0] for tsrchain in tsrchain_list]

        # Create a joint constraint function over all TSRs.
        def f(x):
            robot.SetActiveDOFValues(x)
            Te = robot.GetActiveManipulator().GetEndEffectorTransform()

            d = []
            for tsr in tsrlist:
                # Compute the transform from Tw to Tw_target. This is the
                # residual in the task frame, which is constrained by Bw.
                Tw_target = numpy.dot(Te, numpy.linalg.inv(tsr.Tw_e))
                Tw_relative = numpy.dot(numpy.linalg.inv(tsr.T0_w), Tw_target)

                # Compute distance from the Bw AABB.
                xyzypr = kin.pose_to_xyzypr(Tw_relative)
                xyzrpy = xyzypr[XYZYPR_TO_XYZRPY]

                distance_vector = numpy.maximum(
                    numpy.maximum(xyzrpy - tsr.Bw[:, 1], numpy.zeros(6)),
                    numpy.maximum(tsr.Bw[:, 1] - xyzrpy, numpy.zeros(6))
                )
                d.append(numpy.linalg.norm(distance_vector, ord=2))

            return numpy.min(d)

        # Return the constraint function itself.
        return f

    def _WaypointsToTraj(self, robot, waypoints):
        """Converts a list of waypoints to an OpenRAVE trajectory."""
        traj = openravepy.RaveCreateTrajectory(robot.GetEnv(), '')
        traj.Init(robot.GetActiveConfigurationSpecification())

        for (i, waypoint) in enumerate(waypoints):
            traj.Insert(i, waypoint)
        return traj

    def _Plan(self, robot, request, interactive=False, **kwargs):
        """
        Plan to a desired configuration with TrajOpt. This will invoke the
        TrajOpt planner on the specified JSON request.
        @param robot the robot whose active DOFs will be used
        @param request a JSON planning request for TrajOpt
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
                raise PlanningError("Result was in collision.")

        # Convert the waypoints to a trajectory.
        return self._WaypointsToTraj(robot, waypoints)

    @PlanningMethod
    def PlanToConfiguration(self, robot, goal, **kwargs):
        """
        Plan to a desired configuration with TrajOpt.
        @param robot the robot whose active DOFs will be used
        @param goal the desired robot joint configuration
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
        Plan to a desired end effector pose with TrajOpt.
        @param robot the robot whose active manipulator will be used
        @param pose the desired manipulator end effector pose
        @param ranker an IK ranking function to use over the IK solutions
        @return traj a trajectory from current configuration to specified pose
        """
        # Plan using the active manipulator.
        manipulator = robot.GetActiveManipulator()

        # Find initial collision-free IK solution.
        from openravepy import (IkFilterOptions,
                                IkParameterization,
                                IkParameterizationType)
        ik_param = IkParameterization(
            pose, IkParameterizationType.Transform6D)
        init_joint_config = manipulator.FindIKSolution(
            ik_param, IkFilterOptions.CheckEnvCollisions)
        if init_joint_config is None:
            raise PlanningError('No collision-free IK solution.')

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
        Plan using the given TSR chains with OMPL.
        @param robot
        @param tsrchains A list of TSRChain objects to respect during planning
        @param is_interactive pause every iteration, until you press 'p'
                              or press escape to disable further plotting.
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
            prob.AddConstraint(self._TsrCostFn(robot, traj_tsrs),
                               [(t, j) for j in xrange(n_dofs)],
                               "EQ", "up{:d}".format(t))
        prob.AddConstraint(self._TsrCostFn(robot, goal_tsrs),
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

    def OptimizeTrajectory(self, robot, traj,
                           distance_penalty=0.025, **kwargs):
        """
        Optimize an existing feasible trajectory.
        @param robot the robot whose active DOFs will be used
        @param traj the original trajectory that will be optimized
        @param distance_penalty the penalty for approaching obstacles
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

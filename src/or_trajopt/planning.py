#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

    def _Plan(self, robot, request, interactive=False, **kwargs):
        """
        Plan to a desired configuration with Trajopt.

        This function invokes the Trajopt planner directly on the specified
        JSON request.  This can be used to implement custom path optimization
        algorithms.

        @param robot the robot whose active DOFs will be used
        @param request a JSON planning request for Trajopt
        @param interactive pause every iteration, until you press 'p'
                              or press escape to disable further plotting.
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
        Plan to a desired configuration with Trajopt.

        @param robot the robot whose active DOFs will be used
        @param goal the desired robot joint configuration
        @param is_interactive pause every iteration, until you press 'p'
                              or press escape to disable further plotting.
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
    def PlanToIK(self, robot, pose,
                 ranker=prpy.ik_ranking.JointLimitAvoidance, **kwargs):
        """
        Plan to a desired end effector pose with Trajopt.

        An IK ranking function can optionally be specified to select a
        preferred IK solution from those available at the goal pose.

        @param robot the robot whose active manipulator will be used
        @param pose the desired manipulator end effector pose
        @param ranker an IK ranking function to use over the IK solutions
        @param is_interactive pause every iteration, until you press 'p'
                              or press escape to disable further plotting.
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
        Plan using the given list of TSR chains with TrajOpt.

        @param robot the robot whose active manipulator will be used
        @param tsrchains a list of TSRChain objects to respect during planning
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

    def OptimizeTrajectory(self, robot, traj,
                           distance_penalty=0.025, **kwargs):
        """
        Optimize an existing feasible trajectory using TrajOpt.

        @param robot the robot whose active DOFs will be used
        @param traj the original trajectory that will be optimized
        @param distance_penalty the penalty for approaching obstacles
        @param is_interactive pause every iteration, until you press 'p'
                              or press escape to disable further plotting.
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
        traj.Init(robot.GetActiveConfigurationSpecification())

        for (i, waypoint) in enumerate(waypoints):
            traj.Insert(i, waypoint)
        return traj

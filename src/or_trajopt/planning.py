#!/usr/bin/env python
# -*- coding: utf-8 -*-

from prpy.planning.base import (BasePlanner,
                                MetaPlanner,
                                PlanningError,
                                PlanningMethod)
import logging
import numpy
import time
import openravepy
import os
from . import constraints

logger = logging.getLogger(__name__)
os.environ['TRAJOPT_LOG_THRESH'] = 'WARN'


class TrajoptWrapper(MetaPlanner):
    def __init__(self, planner):
        """
        Create a PrPy binding that wraps an existing planner and calls
        its planning methods followed by Trajopt's OptimizeTrajectory.

        @param planner the PrPy plan wrapper that will be wrapped
        """
        assert planner
        # TODO: this should be revisited once the MetaPlanners are not assuming
        #       self._planners must exist.
        self._planners = (planner,)
        self._trajopt = TrajoptPlanner()

    def __str__(self):
        return 'TrajoptWrapper({0:s})'.format(self._planners[0])

    def get_planners(self, method_name):
        return [planner for planner in self._planners
                if hasattr(planner, method_name)]

    def plan(self, method, args, kwargs):
        # Can't handle deferred planning yet.
        assert not kwargs['defer']

        # According to prpy spec, the first positional argument is 'robot'.
        robot = args[0]

        # Call the wrapped planner to get the seed trajectory.
        planner_method = getattr(self._planners[0], method)
        traj = planner_method(*args, **kwargs)

        # Simplify redundant waypoints in the trajectory.
        from prpy.util import SimplifyTrajectory
        traj = SimplifyTrajectory(traj, robot)

        # Run path shortcutting on the RRT path.
        import openravepy
        params = openravepy.Planner.PlannerParameters()
        params.SetExtraParameters('<time_limit>0.25</time_limit>')
        simplifier = openravepy.RaveCreatePlanner(robot.GetEnv(),
                                                  'OMPL_Simplifier')
        simplifier.InitPlan(robot, params)

        with robot:
            result = simplifier.PlanPath(traj)
            assert result == openravepy.PlannerStatus.HasSolution

        # Call Trajopt to optimize the seed trajectory.
        # Try different distance penalties until out of collision.
        penalties = numpy.logspace(numpy.log(0.05), numpy.log(0.5), num=4)
        for distance_penalty in penalties:
            try:
                return self._trajopt.OptimizeTrajectory(
                    robot, traj,
                    **kwargs
                )
            except PlanningError:
                logger.warn("Failed to optimize trajectory "
                            "with distance penalty: {:f}"
                            .format(distance_penalty))

        # If all of the optimizations ended in collision,
        # just return the original.
        logger.warn("Failed to optimize trajectory, "
                    "returning unoptimized solution.")
        return traj


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

        # Trajopt's UserData gets confused if the same environment
        # is cloned into multiple times, so create a fresh env here.
        from prpy import Clone
        with Clone(env) as internal_env:
            internal_robot = internal_env.Cloned(robot)

            # Convert dictionary into json-formatted string and create object
            # that stores optimization problem.
            s = json.dumps(request)
            prob = trajoptpy.ConstructProblem(s, internal_env)

            # Perform trajectory optimization.
            t_start = time.time()
            result = trajoptpy.OptimizeProblem(prob)
            t_elapsed = time.time() - t_start
            logger.debug("Optimization took {:.3f} seconds".format(t_elapsed))

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
            with internal_robot:
                # Set robot DOFs to match DOFs in optimization problem
                prob.SetRobotActiveDOFs()

                # Check that trajectory is collision free
                if not traj_is_safe(waypoints, internal_robot):
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
        num_steps = 10

        request = {
            "basic_info": {
                "n_steps": num_steps,
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
    def PlanToEndEffectorPose(self, robot, pose, **kwargs):
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
                  ranker=None, **kwargs):

        # Plan using the active manipulator.
        with robot.GetEnv():
            manipulator = robot.GetActiveManipulator()

            # Distance from current configuration is default ranking.
            if ranker is None:
                from prpy.ik_ranking import NominalConfiguration
                ranker = NominalConfiguration(manipulator.GetArmDOFValues())

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

        # Settings for TrajOpt
        num_steps = 10

        # Construct a planning request with these constraints.
        request = {
            "basic_info": {
                "n_steps": num_steps,
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
                        "timestep": num_steps-1
                    }
                }
            ],
            "init_info": {
                "type": "straight_line",
                "endpoint": init_joint_config.tolist()
            }
        }

        # Set active DOFs to match active manipulator and plan.
        p = openravepy.KinBody.SaveParameters
        with robot.CreateRobotStateSaver(p.ActiveDOF):
            robot.SetActiveDOFs(manipulator.GetArmIndices())
            return self._Plan(robot, request, **kwargs)

    # @PlanningMethod
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
        n_steps = 20
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

    # @PlanningMethod
    def PlanToEndEffectorOffset(self, robot, direction, distance,
                                max_distance=None,
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
        with robot.GetEnv():
            manipulator = robot.GetActiveManipulator()
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

        # Set active DOFs to match active manipulator and plan.
        p = openravepy.KinBody.SaveParameters
        with robot.CreateRobotStateSaver(p.ActiveDOF):
            robot.SetActiveDOFs(manipulator.GetArmIndices())
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

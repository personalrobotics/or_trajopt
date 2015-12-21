#!/usr/bin/env python
# -*- coding: utf-8 -*-

import enum
import logging
import numpy
import time
import openravepy
import os
import prpy.util
from . import constraints
from prpy.planning.base import (BasePlanner,
                                MetaPlanner,
                                PlanningError,
                                PlanningMethod,
                                Tags)
from prpy.planning.exceptions import (CollisionPlanningError,
                                      SelfCollisionPlanningError,
                                      ConstraintViolationPlanningError,
                                      JointLimitError)

logger = logging.getLogger(__name__)
os.environ['TRAJOPT_LOG_THRESH'] = 'WARN'

# Keys under which Trajopt stores custom UserData.
TRAJOPT_USERDATA_KEYS = ['trajopt_cc', 'bt_use_trimesh', 'osg', 'bt']

# Environment objects within which Trajopt stores custom UserData.
TRAJOPT_ENV_USERDATA = '__trajopt_data__'


@enum.unique
class ConstraintType(enum.Enum):
    """
    Constraint function types supported by TrajOpt.

    * EQ: an equality constraint. `f(x) == 0` in a valid solution.
    * INEQ: an inequality constraint. `f(x) >= `0` in a valid solution.
    """
    EQ = 'EQ'
    INEQ = 'INEQ'


@enum.unique
class CostType(enum.Enum):
    """
    Cost function types supported by TrajOpt.

    * SQUARED: minimize `f(x)^2`
    * ABS: minimize `abs(f(x))`
    * HINGE: minimize `f(x)` while `f(x) > 0`
    """
    SQUARED = 'SQUARED'
    ABS = 'ABS'
    HINGE = 'HINGE'


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

    @staticmethod
    def _addFunction(problem, timestep, i_dofs, n_dofs, fnargs, wide=False):
        """ Converts dict of function parameters into cost or constraint. """
        f = fnargs.get('f')
        assert f is not None

        fntype = fnargs.get('type')
        assert fntype is not None

        fnname = "{:s}{:d}".format(str(fnargs['f']), timestep)
        dfdx = fnargs.get('dfdx')
        dofs = fnargs.get('dofs')
        inds = ([list(i_dofs).index(dof) for dof in dofs]
                if dofs is not None else range(n_dofs))
        

        # Trajopt problem function signatures:
        # - AddConstraint(f, [df], ijs, typestr, name)
        # - AddErrCost(f, [df], ijs, typestr, name)
        if wide == True:
           if isinstance(fntype, CostType) and dfdx is None:
               problem.AddErrorCost(f, [(t, i) for i in inds for t in range(timestep)], fntype.value, fnname)
           elif isinstance(fntype, ConstraintType) and dfdx is None: 
               problem.AddConstraint(f, [(timestep, i) for i in inds for timestep in range(timestep)],
                     fntype.value, fnname)
           else:
               ValueError('Trajectory wide constraints and those with dfdx are not supported')
        elif isinstance(fntype, ConstraintType):
            if dfdx is not None:
                problem.AddConstraint(f, dfdx, [(timestep, i) for i in inds],
                                      fntype.value, fnname)
            else:
                problem.AddConstraint(f, [(timestep, i) for i in inds],
                                      fntype.value, fnname)
        elif isinstance(fntype, CostType):
            if dfdx is not None:
                problem.AddErrorCost(f, dfdx, [(timestep, i) for i in inds],
                                     fntype.value, fnname)
            else:
                problem.AddErrorCost(f, [(timestep, i) for i in inds],
                                     fntype.value, fnname)
        else:
            ValueError('Invalid cost or constraint type: {:s}'
                       .format(str(fntype)))

    def _Plan(self, robot, request,
              traj_constraints=(), goal_constraints=(),
              traj_costs=(), goal_costs=(),
              traj_wide_constraints=(), traj_wide_costs=(),
              interactive=False, constraint_threshold=1e-4,
              **kwargs):
        """
        Plan to a desired configuration with Trajopt.

        This function invokes the Trajopt planner directly on the specified
        JSON request. This can be used to implement custom path optimization
        algorithms.

        Constraints and costs are specified as dicts of:
            ```
            {
                'f': [float] -> [float],
                'dfdx': [float] -> [float],
                'type': ConstraintType or CostType
                'dofs': [int]
            }
            ```

        The input to f(x) and dfdx(x) is a vector of active DOF values used in
        the planning problem.  The output is a vector of costs, where the
        value *increases* as a constraint or a cost function is violated or
        unsatisfied.

        See ConstraintType and CostType for descriptions of the various
        function specifications and their expected behavior.

        The `dofs` parameter can be used to specify a subset of the robot's
        DOF indices that should be used. A ValueError is thrown if these
        indices are not entirely contained in the current active DOFs of the
        robot.

        @param robot: the robot whose active DOFs will be used
        @param request: a JSON planning request for Trajopt
        @param traj_constraints: list of dicts of constraints that should be
                                 applied over the whole trajectory
        @param goal_constraints: list of dicts of constraints that should be
                                 applied only at the last waypoint
        @param traj_costs: list of dicts of costs that should be applied over
                           the whole trajectory
        @param goal_costs: list of dicts of costs that should be applied only
                           at the last waypoint
        @param interactive: pause every iteration, until you press 'p' or press
                           escape to disable further plotting
        @param constraint_threshold: acceptable per-constraint violation error
        @returns traj: trajectory from current configuration to specified goal
        """
        import json
        import trajoptpy
        from prpy import util

        # Set up environment.
        env = robot.GetEnv()
        trajoptpy.SetInteractive(interactive)

        # Trajopt's UserData gets confused if the same environment
        # is cloned into multiple times, so create a scope to later
        # remove all TrajOpt UserData keys.
        try:
            # Convert dictionary into json-formatted string and create object
            # that stores optimization problem.
            s = json.dumps(request)
            prob = trajoptpy.ConstructProblem(s, env)
            assert(request['basic_info']['manip'] == 'active')
            assert(request['basic_info']['n_steps'] is not None)
            n_steps = request['basic_info']['n_steps']
            n_dofs = robot.GetActiveDOF()
            i_dofs = robot.GetActiveDOFIndices()
            #print 'Active DOF', n_dofs
            #print s

            # Add trajectory-wide costs and constraints to each timestep.
            for t in xrange(1, n_steps):
                for constraint in traj_constraints:
                    self._addFunction(prob, t, i_dofs, n_dofs, constraint)
                for cost in traj_costs:
                    self._addFunction(prob, t, i_dofs, n_dofs, cost)

            # Add trajectory wide costs and constrains
            for constraint in traj_wide_constraints:
                self._addFunction(prob, n_steps, i_dofs, n_dofs, constraint, wide=True)

            for cost in traj_wide_costs:
                self._addFunction(prob, n_steps, i_dofs, n_dofs, cost, wide=True)

            # Add goal costs and constraints.
            for constraint in goal_constraints:
                self._addFunction(prob, n_steps-1, i_dofs, n_dofs, constraint)

            for cost in goal_costs:
                self._addFunction(prob, n_steps-1, i_dofs, n_dofs, cost)

            # Perform trajectory optimization.
            t_start = time.time()
            result = trajoptpy.OptimizeProblem(prob)
            t_elapsed = time.time() - t_start
            logger.debug("Optimization took {:.3f} seconds".format(t_elapsed))

            # Check for constraint violations.
            for name, error in result.GetConstraints():
                if error > constraint_threshold:
                    raise ConstraintViolationPlanningError(
                        name,
                        threshold = constraint_threshold,
                        violation_by = error)

            # Check for the returned trajectory.
            waypoints = result.GetTraj()
            if waypoints is None:
                raise PlanningError("Trajectory result was empty.")

            # Convert the trajectory to OpenRAVE format.
            traj = self._WaypointsToTraj(robot, waypoints)
            
            # Generate unit-timed trajectory, required for GetCollisionCheckPts.
            timed_traj = util.ComputeUnitTiming(robot, traj)
            
            # Check that trajectory is collision free.
            p = openravepy.KinBody.SaveParameters
            with robot.CreateRobotStateSaver(p.ActiveDOF):
                # Set robot DOFs to DOFs in optimization problem.
                prob.SetRobotActiveDOFs()
                checkpoints = util.GetCollisionCheckPts(robot, timed_traj,
                                                        include_start=True)
                for _, q_check in checkpoints:
                    self._checkCollisionForIKSolutions(robot, [q_check])

            # Convert the waypoints to a trajectory.
            prpy.util.SetTrajectoryTags(traj, {
                    Tags.SMOOTH: True
                }, append=True)
            return traj
        finally:
            for body in env.GetBodies():
                for key in TRAJOPT_USERDATA_KEYS:
                    body.RemoveUserData(key)

            trajopt_env_userdata = env.GetKinBody('__trajopt_data__')
            if trajopt_env_userdata is not None:
                env.Remove(trajopt_env_userdata)

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
        return self._PlanToIK(robot, pose, **kwargs)

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
        return self._PlanToIK(robot, pose, **kwargs)

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
            # Identify collision and raise error.
            self._raiseCollisionErrorForPose(robot, pose)
           

        # Sort the IK solutions in ascending order by the costs returned by the
        # ranker. Lower cost solutions are better and infinite cost solutions
        # are assumed to be infeasible.
        scores = ranker(robot, ik_solutions)
        best_idx = numpy.argmin(scores)
        init_joint_config = ik_solutions[best_idx]

        # Convert IK endpoint transformation to pose. OpenRAVE operates on
        # GetEndEffectorTransform(), which is equivalent to:
        #
        #   GetEndEffector().GetTransform() * GetLocalToolTransform()
        #
        link_pose = numpy.dot(pose, numpy.linalg.inv(manipulator.GetLocalToolTransform()))
        goal_position = link_pose[0:3, 3].tolist()
        goal_rotation = openravepy.quatFromRotationMatrix(link_pose).tolist()

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
        import prpy.util

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
            pose = tsr.sample()
            ik_param = IkParameterization(
                pose, IkParameterizationType.Transform6D)
            init_joint_config = manipulator.FindIKSolutions(
                ik_param, IkFilterOptions.CheckEnvCollisions)
            if init_joint_config:
                break
        if not init_joint_config:
            # Use last pose to report collision error.
            self._raiseCollisionErrorForPose(robot,  pose) 

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
            # Pick the first violated constraint.
            name = result.GetConstraints()[0][0]
            error = result.GetConstraints()[0][1]
            raise ConstraintViolationPlanningError(name, violation_by=error)

        # Check for the returned trajectory.
        waypoints = result.GetTraj()
        if waypoints is None:
            raise PlanningError("Trajectory result was empty.")

        # Convert the trajectory to OpenRAVE format.
        traj = self._WaypointsToTraj(robot, waypoints)

        # Generate unit-timed trajectory, required for GetCollisionCheckPts.
        timed_traj = util.ComputeUnitTiming(robot, traj) 

        # Check that trajectory is collision free.
        p = openravepy.KinBody.SaveParameters
        with robot.CreateRobotStateSaver(p.ActiveDOF):
            # Set robot DOFs to DOFs in optimization problem.
            prob.SetRobotActiveDOFs()
            checkpoints = util.GetCollisionCheckPts(robot, timed_traj,
                                                    include_start=True)
            for _, q_check in checkpoints:
                self._checkCollisionForIKSolutions(robot, [q_check])

        # Return the generated trajectory.
        return traj

    def OptimizeTrajectory(self, robot, traj,
                           distance_penalty=0.050, **kwargs):
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
        goal = traj.GetWaypoint(traj.GetNumWaypoints()-1)[0:7]
        print 'DOFS in Optimize', dofs
        waypoints = [cspec.ExtractJointValues(traj.GetWaypoint(i),
                                              robot, dofs).tolist()
                     for i in range(n_waypoints)]
        print 'Waypoints', len(waypoints)

        request = {
            "basic_info": {
                "n_steps": n_waypoints,
                "manip": "active",
                "start_fixed": True
            },
            #"costs": [
                #{
                #    "type": "joint_vel",
                #    "params": {"coeffs": [1]}
                #},
            #    {
            #        "type": "collision",
            #        "params": {
            #            "coeffs": [75], #was 20
            #            "dist_pen": [distance_penalty]
            #        },
            #    }
            #],
            #"constraints": [
            #    {
            #        "type": "joint",
            #        "params": {"vals": waypoints[-1]}
            #    }
            #],
            "constraints": [
                {
                    "type": "joint",
                    "params": {"vals": goal.tolist()}
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

    def _raiseCollisionErrorForPose(self, robot, pose):
        """ Identify collision for pose and raise error.
        It should be called only when there is no IK solution and collision is expected. 
        """
        from openravepy import (IkFilterOptions,
                                IkParameterization,
                                IkParameterizationType)

        manipulator = robot.GetActiveManipulator()

        ik_param = IkParameterization(
            pose, IkParameterizationType.Transform6D)

        ik_returns = manipulator.FindIKSolutions(
            ik_param,
            openravepy.IkFilterOptions.IgnoreSelfCollisions,
            ikreturn=True,
            releasegil=True)

        if len(ik_returns) == 0: 
            # This is a hack. Sometimse findIKSolutions fails to find a solution
            # and claims that it's due to JointLimit, but IgnoreJointLimit cant't find 
            # solution either. 
            ik_return = manipulator.FindIKSolution(
                ik_param, 
                openravepy.IkFilterOptions.CheckEnvCollisions,
                ikreturn = True, 
                releasegil = True)
            raise PlanningError(str(ik_return.GetAction())) #this most likely is JointLimit
        
        self._checkCollisionForIKSolutions(robot, map(lambda x: x.GetSolution(), ik_returns))
        raise Exception('Collision/JointLimit error expected but not found.')

    def _checkCollisionForIKSolutions(self, robot, ik_solutions): 
        """ Raise collision/joint limit  error if there is one in ik_solutions
        Should be called while saving robot's current state 
        """
        from openravepy import CollisionReport
        manipulator = robot.GetActiveManipulator()  
        p = openravepy.KinBody.SaveParameters

        with robot.CreateRobotStateSaver(p.LinkTransformation):
            for q in ik_solutions: 
                robot.SetActiveDOFValues(q)
                report = CollisionReport() 
                env = robot.GetEnv()
                if env.CheckCollision(robot, report=report): 
                    raise CollisionPlanningError.FromReport(report)
                elif robot.CheckSelfCollision(report=report):
                    raise SelfCollisionPlanningError.FromReport(report)

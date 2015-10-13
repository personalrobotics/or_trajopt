^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package or_trajopt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* Added 'smooth' tag to all trajopt trajectories (`#9 <https://github.com/personalrobotics/or_trajopt/issues/9>`_)
* Updated package.xml file to format 2 (`#7 <https://github.com/personalrobotics/or_trajopt/issues/7>`_)
* Removed extra clone operation inside Trajopt (`#4 <https://github.com/personalrobotics/or_trajopt/issues/4>`_)
* Exported constraint and cost types in the main Python module
* Contributors: Michael Koval, Pras Velagapudi

1.0.0 (2015-05-01)
------------------
* Fixed `return` instead of `raise` PlanningError.
* Fixes bug where robot active DOFs were changed.
  This adds state savers around all ActiveDOF calls (one was missing) to make sure that none of the trajopt planning calls have persistent effects on the robot.
* Doubled default collision penalty.
* Add clone inside planning function.
  This isolates the TrajOpt user data between multiple calls, which
  fixes some userdata issues with collision geometry not being
  initialized correctly in subsequent calls.
* Implemented bugfixes and more robust planner wrapper.
* Added TrajoptWrapper metaplanner that wraps a planner with trajopt.
  This enables the output of other planners to be postprocessed with
  trajopt's OptimizeTrajectory function.
* Fixed some bugs in the TrajOpt binding with active manipulators.
* Cleaned up arguments to match prpy spec and aborted unfinished calls.
* Added boilerplate for PlanToEndEffectorOffset.
* Added PlanToEndEffector/Offset stubs, and cleaned up docstrings.
* Added fix for some slack in constraint violation.
* Added a note about building TrajOpt in Parallels.
* Added linear interpolation flag to waypoints.
* Added ranking function to PlanToIK.
* Refactored planner and constraints into separate python files.
* Update README.md
* Created initial commit of python hooks for trajopt.
* Initial commit
* Contributors: Jennifer King, Michael Koval, Pras, Pras Velagapudi

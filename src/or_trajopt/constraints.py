import itertools
import time
import numpy
import openravepy


def _TsrSampler(tsrchains, timelimit=2.0):
    """Returns a generator that samples TSR chains until the timelimit"""
    # Create an iterator that cycles through each TSR chain.
    tsr_cycler = itertools.cycle(tsrchains)

    # Create an iterator that cycles TSR chains until the timelimit.
    tsr_timelimit = time.time() + timelimit
    return itertools.takewhile(
        lambda v: time.time() < tsr_timelimit, tsr_cycler)


def _TsrCostFn(robot, tsrchain_list):
    """ Generates a cost function for a list of TSR chains.

    This function returns the minimum projected Euclidean distance to the
    closest TSR in a TSR set.
    """
    import prpy.tsr.kin as kin
    XYZYPR_TO_XYZRPY = [0, 1, 2, 5, 4, 3]

    # I don't know how to project onto TSR chains...
    for tsrchain in tsrchain_list:
        if len(tsrchain.TSRs) != 1:
            raise ValueError("Cannot project TSR chain: {:s}"
                             .format(str(tsrchain.TSRs)))
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
            xyzypr = kin.pose_to_xyzypr(openravepy.poseFromMatrix(Tw_relative))
            xyzrpy = xyzypr[XYZYPR_TO_XYZRPY]

            distance_vector = numpy.maximum(
                numpy.maximum(xyzrpy - tsr.Bw[:, 1], numpy.zeros(6)),
                numpy.maximum(tsr.Bw[:, 1] - xyzrpy, numpy.zeros(6))
            )
            d.append(numpy.linalg.norm(distance_vector, ord=2))

        # Return the minimum cost or 0.0 if there are no constraints.
        return numpy.min(d) if len(d) else 0.0

    # Return the constraint function itself.
    return f

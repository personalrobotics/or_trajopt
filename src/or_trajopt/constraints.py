import itertools
import time
import numpy


def xyzrpy_from_H(H):
    """Converts from homogeneous transform to translation and roll-pitch-yaw"""
    xyzrpy = numpy.zeros(6)
    xyzrpy[0:3] = H[:3, 3]

    # Compute roll, pitch, yaw from Lavalle 2006.
    # See: http://planning.cs.uiuc.edu/node103.html
    xyzrpy[3] = numpy.arctan2(H[1, 0], H[0, 0])
    xyzrpy[4] = numpy.arctan2(-H[2, 0], numpy.linalg.norm(H[2, 1:3], ord=2))
    xyzrpy[5] = numpy.arctan2(H[2, 1], H[2, 2])
    return xyzrpy


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
            xyzrpy = xyzrpy_from_H(Tw_relative)

            distance_vector = numpy.maximum(
                # Nonzero distance if we are greater than the max Bw.
                numpy.maximum(xyzrpy - tsr.Bw[:, 1], numpy.zeros(6)),
                # Nonzero distance if we are less than the min Bw.
                numpy.maximum(tsr.Bw[:, 0] - xyzrpy, numpy.zeros(6))
            )
            d.append(numpy.linalg.norm(distance_vector, ord=2))

        # Return the minimum cost or 0.0 if there are no constraints.
        return d if len(d) else 0.0

    # Return the constraint function itself.
    return f

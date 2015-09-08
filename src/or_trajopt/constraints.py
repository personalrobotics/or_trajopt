import itertools
import time
import numpy


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
    # Create a joint constraint function over all TSR chains.
    def f(x):
        robot.SetActiveDOFValues(x)
        Te = robot.GetActiveManipulator().GetEndEffectorTransform()
        d = [tsrchain.distance(Te)[0] for tsrchain in tsrchain_list]
        return numpy.min(d)

    # Return the constraint function itself.
    return f

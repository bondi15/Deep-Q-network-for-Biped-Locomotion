import numpy as np
import itertools


class State:

    def __init__(self):
        self._jointPos = [0] * 12
        self._jointVel = [0] * 12
        self._idx = 0
        self._scale = [1, 1, 1]

    def SetPosVel(self, pos, vel):
        self._jointPos = pos
        self._jointVel = vel

    def SetState(self, pos, vel, ind, sca):
        self._jointPos = pos
        self._jointVel = vel
        self._idx = ind
        self._scale = sca

    def GetState(self):
        mergedState = list(itertools.chain.from_iterable([self._jointPos,
                                                          self._jointVel,
                                                          [self._idx],
                                                          self._scale]))
       # mergedState.append(self._idx)
        return np.asarray(mergedState)

    def Reset(self):
        self._jointPos = [0] * 12
        self._jointVel = [0] * 12
        self._idx = 0
        self._scale = [1, 1, 1]



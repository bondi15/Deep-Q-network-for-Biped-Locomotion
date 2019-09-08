import numpy as np
import itertools
from math import sqrt

class ReferenceZMP:

    def __init__(self):

        self._LeftFootCenter_Y = 0.15882
        self._RightFootCenter_Y = -0.15796
        self._StepSize = 0.2
        self._StepTime = 45
        self._ZMP_MotionUnderSole = 0.015
        self._DoubleSupportTime = 60
        self._savex = []
        self._savey = []

        # LEFT LEG SUPPORT -> DOUBLE PHASE -> RIGHT LEG SUPPORT -> DOUBLE PHASE - 20

        self._X = [[0] * 30 +
                  list(np.linspace(start=0.0, stop=self._ZMP_MotionUnderSole, num=self._StepTime)) +
                  list(np.linspace(start=self._ZMP_MotionUnderSole, stop=self._StepSize - self._ZMP_MotionUnderSole/2, num=self._DoubleSupportTime)) +
                  list(np.linspace(start=self._StepSize - self._ZMP_MotionUnderSole/2, stop=self._StepSize + self._ZMP_MotionUnderSole/2, num=self._StepTime)) +
                  list(np.linspace(start=self._StepSize + self._ZMP_MotionUnderSole/2, stop=2*self._StepSize - self._StepSize/2, num=self._DoubleSupportTime/2))]
        self._X = list(itertools.chain(*self._X))
        self._X = np.asarray(self._X) + 0.17470423     # CALIBRATION

        self._Y = [list(np.linspace(start=0.0, stop=self._LeftFootCenter_Y, num=self._DoubleSupportTime/2)) +
                   [self._LeftFootCenter_Y] * self._StepTime +
                   list(np.linspace(start=self._LeftFootCenter_Y, stop=self._RightFootCenter_Y, num=self._DoubleSupportTime)) +
                   [self._RightFootCenter_Y]*self._StepTime +
                   list(np.linspace(start=self._RightFootCenter_Y, stop=0.0, num=self._DoubleSupportTime/2))]
        self._Y = list(itertools.chain(*self._Y))
        self._Y = np.asarray(self._Y) + 0.0225431802    # CALIBRATION

        self._RealZmpX = []
        self._RealZmpY = []

    def GetReward(self, sphereheight, RealZMP_X, RealZMP_Y, idx, totreward):
        reward = 0.0
        quit = False
        done = False
        RealZMP_X = RealZMP_X/100
        RealZMP_Y = RealZMP_Y/100

        dist_X = abs(self._X[idx - 15] - RealZMP_X)
        dist_Y = abs(self._Y[idx - 15] - RealZMP_Y)

        reward += -sqrt(pow(dist_X, 2) + pow(dist_Y, 2))*10
        # print("reward before falling", reward)

        self._RealZmpX.append(RealZMP_X)
        self._RealZmpY.append(RealZMP_Y)

        if sphereheight < 0.82:
            quit = True
            reward += -250 + idx

        if all([idx == len(self._X)-1, abs(totreward) < 5]):
            done = True
            reward += 200

        return reward, quit, done



import vrep
import sys
import pandas as pd
import numpy as np
from math import pi
import itertools
from State import State
from ZMP import ReferenceZMP


def _DataFromCSV(filename, order, offset):
    df = pd.read_csv(filename, header=None)
    np_mat = df.values
    data = {}
    for joint, val, off in zip(order, np_mat, offset):
        data[joint] = [sval + off for sval in val[170:380]]
    return data


def _AddInitialPos(data, offset, order):
    _data = {}
    for k, v in data.items():
        ini = list(np.linspace(0, v[0] + offset[order.index(k)], num=35))
        _data[k] = ini + list(data[k])
    return _data


def _GetJointValues(clientID, handles, joints, lastposition):

    position = []
    velocity = []

    for el, prevpos in zip(joints, lastposition):
        errorCode, pos = vrep.simxGetJointPosition(clientID, handles[el], vrep.simx_opmode_streaming)
        vel = (pos - prevpos) / 0.01   # angle/second

        position.append(pos)
        velocity.append(vel)

    return position, velocity


def _GetSensorValues(clientID, right, left, handles):
        rightSensors = []
        leftSensors = []


        for senR, senL in zip(right,left):
            errorCode, trash, values, trash2 = vrep.simxReadForceSensor(clientID, handles[senR],
                                                                            vrep.simx_opmode_streaming)
            rightSensors.append(values[2])

            errorCode, trash, values, trash2 = vrep.simxReadForceSensor(clientID, handles[senL],
                                                                        vrep.simx_opmode_streaming)
            leftSensors.append(values[2])

        return rightSensors, leftSensors

def _GetZmp(clientID, handles, floor, rightval, leftval, rightsum, leftsum):

    W = 0.1
    L = 0.2

    errorCode, rightFootPos = vrep.simxGetObjectPosition(clientID, handles['RFootRespond'], handles[floor], vrep.simx_opmode_streaming)
    rightFootPosX = rightFootPos[0]
    rightFootPosY = rightFootPos[1]
    errorCode, leftFootPos = vrep.simxGetObjectPosition(clientID, handles['LFootRespond'], handles[floor], vrep.simx_opmode_streaming)
    leftFootPosX = leftFootPos[0]
    leftFootPosY = leftFootPos[1]

    rightFootZMP_X = ((L/2)/rightsum) * ((rightval[0] + rightval[1]) - (rightval[2] + rightval[3]))
    rightFootZMP_Y = ((W/2)/rightsum) * ((rightval[0] + rightval[2]) - (rightval[1] + rightval[3]))
    rightFootAbsZMP_X = rightFootZMP_X + rightFootPosX
    rightFootAbsZMP_Y = rightFootZMP_Y + rightFootPosY

    leftFootZMP_X = ((L/2)/leftsum) * ((leftval[0] + leftval[1]) - (leftval[2] + leftval[3]))
    leftFootZMP_Y = ((W/2)/leftsum) * ((leftval[0] + leftval[2]) - (leftval[1] + leftval[3]))
    leftFootAbsZmp_X = leftFootZMP_X + leftFootPosX
    leftFootAbsZmp_Y = leftFootZMP_Y + leftFootPosY

    TotalZMP_X = (leftFootAbsZmp_X * leftsum + rightFootAbsZMP_X * rightsum)
    TotalZmp_Y = (leftFootAbsZmp_Y * leftsum + rightFootAbsZMP_Y * rightsum)

    return TotalZMP_X, TotalZmp_Y #, leftFootAbsZmp_X, leftFootAbsZmp_Y, rightFootAbsZMP_X, rightFootAbsZMP_Y

class Environment:

    def __init__(self, portnumber, scene, trajectory_file, order, action_values):

        self._portnumber = portnumber

        self._scene = scene
        self._order = order

        self._handles_dict = {}
        self._rad = lambda x: x * (pi / 180)

        self._action_values = action_values
        self._actions = list(itertools.product([self._action_values[0], self._action_values[1], self._action_values[2]],
                                               repeat=3))
        self._num_actions = len(self._actions)
        #print(self._actions.index((0.0, 0.0, 0.0)))

        self._state = State()
        self._num_states = 28

        self._refZMP = ReferenceZMP()

        self._reward = 0
        self._done = False
        self._quit = False
        self._sphere_height = 0.0

        self._firstSim = True
        self._range = 5

        self._floor = 'ResizableFloor_5_25'
        self._rightSen = ['Right_br', 'Right_bl', 'Right_fr', 'Right_fl']
        self._leftSen = ['Left_br', 'Left_bl', 'Left_fr', 'Left_fl']

        self._JointsSca1 = ['rAnkleP6', 'lAnkleP6', 'lHipPit2', 'rHipPit2']
        self._JointsSca2 = ['rKneeee4', 'lKneeee4']
        self._JointsSca3 = ['lAnkleR5', 'rAnkleR5', 'lHipRol3', 'rHipRol3']

#        self._offset = [0, -9, -15, 10, 0, -20,
#                         0, -9, -15, 10, 0, -20]

        self._offset = [0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0]

        self._walking_data = _DataFromCSV(trajectory_file, order, self._offset)
        self._walking_data = _AddInitialPos(self._walking_data, self._offset, self._order)

        self.clientID = -1


    def _connect(self):
        vrep.simxFinish(-1)
        self.clientID = vrep.simxStart('127.0.0.1', self._portnumber, True, True, 5000, 5)

        if self.clientID != -1:
            print("Connected to remote API server!")
            vrep.simxSetIntegerSignal(self.clientID, 'asdf', 1, vrep.simx_opmode_oneshot)

        else:
            sys.exit('Could not connect!')

    def _reset(self):
        if self._firstSim:
            self._connect()
            vrep.simxLoadScene(self.clientID, self._scene, 0, vrep.simx_opmode_blocking)

        else:
            vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)

            number_of_iteration_not_stopped = 0
            while True:
                vrep.simxGetIntegerSignal(self.clientID, 'asdf', vrep.simx_opmode_blocking)
                e = vrep.simxGetInMessageInfo(self.clientID, vrep.simx_headeroffset_server_state)
                not_stopped = e[1] & 1
                if not not_stopped:
                    print("STOPPED")
                    break
                else:
                    if number_of_iteration_not_stopped % 10 == 0:
                        print(number_of_iteration_not_stopped, ': not stopped')

                number_of_iteration_not_stopped += 1

                if number_of_iteration_not_stopped > 100:
                    self._firstSim = True
                    self._connect()
                    vrep.simxLoadScene(self.clientID, self._scene, 0, vrep.simx_opmode_blocking)

        errorCode, handles, intData, floatData, stringData = vrep.simxGetObjectGroupData(self.clientID,
                                                                                         vrep.sim_appobj_object_type,
                                                                                         0, vrep.simx_opmode_blocking)
        self._handles_dict = dict(zip(stringData, handles))

        tmp = []
        self.new_trajectory = {key: list(tmp) for key in self._order}

        for el in self._order:
            vrep.simxSetJointTargetPosition(self.clientID, self._handles_dict[el], 0, vrep.simx_opmode_oneshot)
            vrep.simxGetJointPosition(self.clientID, self._handles_dict[el], vrep.simx_opmode_streaming)
        #
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)

        for i in range(self._range):
            if self._firstSim:
                print("FirStSim iteration:  ", i)
                for el in self._order:
                    vrep.simxSynchronousTrigger(self.clientID)
                    vrep.simxSetJointTargetPosition(self.clientID, self._handles_dict[el], self._rad(self._walking_data[el][i]),
                                                    vrep.simx_opmode_streaming)
            else:
                print("Reset iteration:  ", i)
                for el in self._order:
                    vrep.simxSynchronousTrigger(self.clientID)
                    vrep.simxSetJointTargetPosition(self.clientID, self._handles_dict[el], self._rad(self._walking_data[el][i]),
                                                    vrep.simx_opmode_streaming)
            if i == (self._range - 2):
                lastposlist = []
                for jo in self._order:
                    errorCode, lastpostmp = vrep.simxGetJointPosition(self.clientID, self._handles_dict[jo], vrep.simx_opmode_streaming)
                    lastposlist.append(lastpostmp)



        rightSenValues, leftSenValues = _GetSensorValues(self.clientID, self._rightSen, self._leftSen,
                                                         self._handles_dict)
        # print("RightSensors: ", rightSenValues)
        # print("LeftSensors: ", leftSenValues)

        rightSenSum = np.sum(np.asarray(rightSenValues))
        leftSenSum = np.sum(np.asarray(leftSenValues))

        zmpX, zmpY = _GetZmp(self.clientID, self._handles_dict, self._floor, rightSenValues, leftSenValues, rightSenSum, leftSenSum)

        # print("ZMPX: ", zmpX)
        # print("ZMPy: ", zmpY)

        errorCode, sphere = vrep.simxGetObjectPosition(self.clientID, self._handles_dict['Sphere'], self._handles_dict[self._floor], vrep.simx_opmode_streaming)
        self._sphere_height = sphere[2]
        # print("Sphere height: ", self._sphere_height)

        CurrentPos, CurrentVel = _GetJointValues(self.clientID, self._handles_dict, self._order, lastposlist)
        # print("Position: ", CurrentPos)
        # print("Velocity: ", CurrentVel)

        self._state.Reset()
        self._state.SetPosVel(CurrentPos, CurrentVel)


        if self._firstSim:
            self._firstSim = False
            self._range = 15
            self._reset()

        self._done = False
        self._quit = False

        return self._state.GetState()

    def _step(self, action, index, totreward):
        next_state = State()
        self._reward = 0
        current_action = list(self._actions[action])
        # print("Current action", current_action)
        # print("Current state:", len(self._state.GetState()))

        for el in ['rHipRol3', 'lHipRol3', 'rHipPit2', 'lHipPit2', 'rHipYaw1', 'lHipYaw1', 'rKneeee4', 'lKneeee4', 'rAnkleP6', 'lAnkleP6', 'rAnkleR5', 'lAnkleR5']: #self._order:
            if el in self._JointsSca1:
                vrep.simxSynchronousTrigger(self.clientID)
                vrep.simxSetJointTargetPosition(self.clientID, self._handles_dict[el],
                                                self._rad((self._walking_data[el][index]) * (self._state.GetState()[-3] + current_action[0])),
                                                vrep.simx_opmode_streaming)
                #print("error: ", e,"ToJoint: ", self._rad((self._walking_data[el][index]) * (self._state.GetState()[-3] + current_action[0])))
                # print("Joint - sca1: ", el)
                # print("Offset: ", self._offset[self._order.index(el)])
                # print("Scale: ", self._state.GetState()[-3])

            elif el in self._JointsSca2:
                vrep.simxSynchronousTrigger(self.clientID)
                vrep.simxSetJointTargetPosition(self.clientID, self._handles_dict[el],
                                                self._rad((self._walking_data[el][index]) * (self._state.GetState()[-2]+current_action[1])),
                                                vrep.simx_opmode_streaming)
                #print("error: ", e, "ToJoint: ", self._rad((self._walking_data[el][index]) * (self._state.GetState()[-2]+current_action[1])))
                # print("Joint - sca2: ", el)
                # print("Offset: ", self._offset[self._order.index(el)])
                # print("Scale: ", self._state.GetState()[-3])

            elif el in self._JointsSca3:
                vrep.simxSynchronousTrigger(self.clientID)
                vrep.simxSetJointTargetPosition(self.clientID, self._handles_dict[el],
                                                self._rad((self._walking_data[el][index]) * (self._state.GetState()[-1] + current_action[2])),
                                                vrep.simx_opmode_streaming)
                #print("error: ", e,"ToJoint: ", self._rad((self._walking_data[el][index]) * (self._state.GetState()[-1] + current_action[2])))
                # print("Joint - sca3: ", el)
                # print("Offset: ", self._offset[self._order.index(el)])
                # print("Scale: ", self._state.GetState()[-3])

            else:
                vrep.simxSynchronousTrigger(self.clientID)
                vrep.simxSetJointTargetPosition(self.clientID, self._handles_dict[el],
                                                self._rad(self._walking_data[el][index]),
                                                vrep.simx_opmode_streaming)
                #print("error: ", e,"ToJoint: ", self._rad(self._walking_data[el][index]))
                # print("Joint - no sca: ", el)
                # print("Offset: ", self._offset[self._order.index(el)])
                # print("Scale: no scale")



        CurrentPos, CurrentVel = _GetJointValues(self.clientID, self._handles_dict, self._order,
                                                 self._state.GetState()[0:12])
        # print(self._state.GetState()[-4:])
        next_state.SetState(CurrentPos, CurrentVel, index, self._state.GetState()[-3:] + current_action)
        # print("next state:", len(next_state.GetState()))
        # print(next_state.GetState()[-4:])

        for key, val in next_state.__dict__.items():
            self._state.__dict__.__setitem__(key, val)

        rightSenValues, leftSenValues = _GetSensorValues(self.clientID, self._rightSen, self._leftSen,
                                                         self._handles_dict)

        rightSenSum = np.sum(np.asarray(rightSenValues))
        leftSenSum = np.sum(np.asarray(leftSenValues))
        CurrentZmpX, CurrentZmpY = _GetZmp(self.clientID, self._handles_dict, self._floor, rightSenValues,
                                           leftSenValues, rightSenSum, leftSenSum)

        errorCode, sphere = vrep.simxGetObjectPosition(self.clientID, self._handles_dict['Sphere'], self._handles_dict[self._floor], vrep.simx_opmode_streaming)
        self._sphere_height = sphere[2]

        self._reward, self._quit, self._done = self._refZMP.GetReward(self._sphere_height, CurrentZmpX, CurrentZmpY, index, totreward)

        return next_state.GetState(), self._reward, self._done, self._quit



# file = '/home/bondi/Documents/MLP_RL_V2/walk500.csv'
# scene = '/home/bondi/Documents/MLP_RL_V2/NUHumanoid_V3_12.5kg_DisturbancesForce_Sensors.ttt'
# portnumber = 19990
# order = ['rHipYaw1', 'rHipRol3', 'rHipPit2', 'rKneeee4', 'rAnkleR5', 'rAnkleP6',
#            'lHipYaw1', 'lHipRol3', 'lHipPit2', 'lKneeee4', 'lAnkleR5', 'lAnkleP6']
#
# env = Environment(portnumber, scene, file, order, [-0.1, 0.0, 0.1])





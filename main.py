import tensorflow as tf
import matplotlib.pyplot as plt
import time
import vrep

from EnvironmentV2 import Environment
from Model import Model
from memory import Memory
from SimRunner import SimRunner
from Visu_JSON import VisuJSON

file = '/home/nuhumanoid/Documents/MLP_RL_V2/walk500.csv'
scene = '/home/nuhumanoid/Documents/MLP_RL_V2/NUHumanoid_V3_12.5kg_DisturbancesForce_Sensors.ttt'
portnumber = 19997
order = ['rHipYaw1', 'rHipRol3', 'rHipPit2', 'rKneeee4', 'rAnkleR5', 'rAnkleP6',
          'lHipYaw1', 'lHipRol3', 'lHipPit2', 'lKneeee4', 'lAnkleR5', 'lAnkleP6']
action_values = [-0.1, 0.0, 0.1]

env = Environment(portnumber, scene, file, order, action_values)

num_states = env._num_states
num_actions = env._num_actions


BATCHSIZE = 128

MAX_EPS = 1
MIN_EPS = 0.001
DECAY = 0.0004
GAMMA = 0.8

NUM_EPISODE = 10000

model = Model(num_states, num_actions, BATCHSIZE)
mem = Memory(700000)
vis = VisuJSON(env._refZMP._X, env._refZMP._Y)

trajectory = {}

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(model._var_init)
    sr = SimRunner(sess, model, env, mem, MAX_EPS, MIN_EPS, DECAY, GAMMA)
    cnt = 0
    while cnt < NUM_EPISODE+1:
        start = time.time()

        # if cnt % 10 == 0:
        #     print('------------------------------------------------------------------------------')
        #     print('Episode {} of {}'.format(cnt+1, NUM_EPISODE))

        print('------------------------------------------------------------------------------')
        print('Episode {} of {}'.format(cnt+1, NUM_EPISODE))

        env._refZMP._RealZmpX = []
        env._refZMP._RealZmpY = []

        sr.run()

        if cnt % 50 == 0:
            vis.savefigZMP(env._refZMP._RealZmpX, env._refZMP._RealZmpY, sr._reward_store[-1], cnt)
            vis.savefigReward(sr._reward_store, cnt)
            saver.save(sess, 'my_test_model')

        cnt = cnt + 1

        end = time.time()
        t = end - start

        print('Time of the episode:', t)
        # plt.plot(env._refZMP._X, env._refZMP._Y, 'b')
        # plt.plot(env._refZMP._savex, env._refZMP._savey, 'r')
        # plt.show()

    vrep.simxStopSimulation(env.clientID, vrep.simx_opmode_oneshot)
    plt.close("all")






import numpy as np
import random
import math


class SimRunner:

    def __init__(self, sess, model, env, memory, max_eps, min_eps, decay, gamma):
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._decay = decay
        self._gamma = gamma
        self._eps = self._max_eps
        self._eps_steps = 0

        self._steps = 0
        self._reward_store = []

    def run(self,):
        state = self._env._reset()
        tot_reward = 0
        self._steps = 15

        while True:

            if self._steps > 224:
                self._reward_store.append(tot_reward)
                self._eps_steps += 1
                print("Finished at iteration: ", self._steps)
                break

            action = self._choose_action(state)
            next_state, reward, done, fall = self._env._step(action, self._steps, tot_reward)

            if done:
                next_state = None

            self._memory.add_sample((state, action, reward, next_state))
            self._replay()
            
            if all([self._eps_steps !=0, self._eps_steps % 50 == 0]):
                self._eps = 0
            else:
                self._eps = self._min_eps + (self._max_eps - self._min_eps) \
                                * math.exp(-self._decay * self._eps_steps)

            state = next_state

            tot_reward += reward

            if done:
                self._reward_store.append(tot_reward)
                self._eps_steps += 1
                print("Finished at iteration: ", self._steps)
                break

            if fall:
                self._reward_store.append(tot_reward)
                self._eps_steps += 1
                print("Finished at iteration: ", self._steps)
                break
#            print("=========================================")
#            print("EPS steps: ", self._eps_steps)
#            print("EPSILON: ", self._eps)
#            print("=========================================")
            self._steps = self._steps + 1

        print("Total reward: {}, Eps: {}".format(tot_reward, self._eps))


    def _choose_action(self, state):
        if random.random() < self._eps:
            return random.randint(0, self._model._num_actions - 1)
        else:
            return np.argmax(self._model.predict_one(state, self._sess))


    def _replay(self):
        batch = self._memory.sample(self._model._batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([np.zeros(self._model._num_states) if val[3] is None else val[3]
                                for val in batch])

        q_s_a = self._model.predict_batch(states, self._sess)
        q_s_a_d = self._model.predict_batch(next_states, self._sess)

        x = np.zeros((len(batch), self._model._num_states))
        y = np.zeros((len(batch), self._model._num_actions))

        for i, b in enumerate(batch):

            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            current_q = q_s_a[i]

            if next_state is None:
                current_q[action] = reward

            else:
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])

            x[i] = state
            y[i] = current_q

        self._model.train_batch(self._sess, x, y)



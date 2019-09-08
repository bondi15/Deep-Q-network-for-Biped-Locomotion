import matplotlib.pyplot as plt

class VisuJSON:

    def __init__(self, RefZmpX, RefZmpY):
        self.RefZmpX = RefZmpX
        self.RefZmpY = RefZmpY

        self.maxReward = -100

    def getMaxReward(self):
        return self.maxReward

    def savefigZMP(self, TrueZmpX, TrueZmpY, reward, epnumber):
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax.plot(self.RefZmpX, self.RefZmpY, label = 'Reference ZMP' )
        ax.plot(TrueZmpX, TrueZmpY, 'r', label = 'True Zmp')
        ax.legend()
        ax.text(0.9, 0.1, 'reward: '+str(reward), horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        if len(TrueZmpY) < 215:
            ax.text(0.8, 0.5, 'FALL DOWN - '+str(len(TrueZmpY)), horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        fig.savefig('/home/nuhumanoid/Documents/MLP_RL_V2/Figures/ZMP_Plot_Ep_'+str(epnumber)+'.png')  # save the figure to file
        plt.close(fig)

    def savefigReward(self, all_reward, epnumber):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(all_reward)
        ax.text(0.9, 0.1, 'Episode: ' + str(epnumber), horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        fig.savefig('/home/nuhumanoid/Documents/MLP_RL_V2/Figures/Reward_Plot_Ep_' + str(epnumber) + '.png')  # save the figure to file
        plt.close(fig)

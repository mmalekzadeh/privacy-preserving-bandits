import numpy as np
import matplotlib.pyplot as plt

class Plotter():
    def __init__(self):
        self.scale = 1
        
    def plot_pms(self, p_mat, scale=1, save_fig=False, file_name="prior_pms.png"):
        """
        To plot a Probaility Mass Function.
        p_mat is a PMS each row sums to 1.
        scale is to get different image and font size
        """
        
        plt.rcParams["figure.figsize"] = p_mat.shape[1]*scale+2, p_mat.shape[0]*scale+1
        current_left = np.zeros(p_mat.shape[0])
        for p in range(p_mat.shape[1]):
            plt.barh(range(p_mat.shape[0]), p_mat[:,p], left = current_left, label="p"+str(p))
            current_left += p_mat[:,p]
            
            plt.yticks(range(p_mat.shape[0]+1), fontsize=p_mat.shape[0]*scale)
            plt.ylabel("users", fontsize=p_mat.shape[0]*scale)
            plt.xticks(fontsize=p_mat.shape[0]*scale)
            plt.xlim(0,1)
            plt.xlabel("preference probabilities", fontsize=p_mat.shape[0]*scale)
            

        plt.legend(loc = 'upper left', prop={'size': p_mat.shape[1]+scale}, ncol=p_mat.shape[1], fancybox=True, shadow=True)
        
        if save_fig:
            plt.savefig(file_name)
            plt.clf() 
        
    def plot_history(self, history, K_val=1, scale=1, save_fig=False, file_name="action_hist.png"):
        """
        To plot a history of 
        content browsed by users
        or
        actions taken by learners
        
        history is a matrix each row shows a user
        scale is to get different image and font size
        """
        
        plt.rcParams["figure.figsize"] = history.shape[1]+2, K_val+1
        for u in range(history.shape[0]):
            plt.plot(range(history.shape[1]), history[u],'o-', label="u"+str(u))
            
            plt.yticks(range(K_val+1), fontsize=history.shape[0]*scale)
            plt.ylabel("preferences", fontsize=history.shape[0]*scale)
            plt.xticks(fontsize=history.shape[0]*scale)
            plt.xlabel("timepoints", fontsize=history.shape[0]*scale)
            
        if(len(history)!=1):
            plt.legend(loc = 'upper left', prop={'size': len(np.unique(history))+2+scale}, ncol=history.shape[0], fancybox=True, shadow=True)
        
        if save_fig:
            plt.savefig(file_name)
            plt.clf()

    def get_ctr(self, hist):
        ctrs = list()
        for k in hist:
            tmp = np.array(hist[k][:,2], dtype=int)
            tmp = tmp.sum()
            ctrs.append(round((tmp/len(hist[k])),5))
        return [round(np.mean(ctrs),5) , round(np.std(ctrs),5)]
        
    def plot_rec_rew(self, hist, K_val, sp_response):
        #plt.rcParams["figure.figsize"] = 30, 15
        u = np.random.randint(len(hist))
        width = 0.25

        tot_rew = np.bincount(hist[u][:,1].astype(int), minlength=K_val)
        tot_rew = (tot_rew/tot_rew.sum())*100
        plt.bar(np.arange(K_val)-width, tot_rew, width, label="Recommendations")

        tmp = ((hist[u][:,1]+1)*(hist[u][:,2]*2-1)).astype(int)
        tmp = tmp[tmp>0]-1
        ach_rev = np.bincount(tmp, minlength=K_val)
        ach_rev = ((ach_rev/ach_rev.sum())*100)
        plt.bar(np.arange(K_val), ach_rev, width, label="Clicked", alpha=0.75)


        resps = np.asarray(sp_response[u].sum(axis=0))
        resps = (resps/resps.sum())*100
        plt.bar(np.arange(K_val)+width, resps, width, label = "True Rewards Distribution")

        plt.xlabel("Ads. Categories",  size=32)
        plt.ylabel("Percentage (%)",  size=32)
        plt.xticks(np.arange(K_val), size=30)
        plt.yticks(np.arange(0, ach_rev.max()+1, (ach_rev.max()//10)+1), size=30)
        plt.legend(prop={'size': 32})
        plt.grid()
        plt.show()
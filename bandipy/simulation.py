## Basics
import os
import sys
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
## To save a trained ML Models
import pickle
from sklearn.cluster import KMeans
## To calculate possible combinations of a histograms
from scipy.special import comb
## To round a vector without changing its sum
import iteround
## Our Library
#sys.path.append("../")
from bandipy import environment
from bandipy import experiment
from bandipy import prior
from bandipy import policy
from bandipy import plotter
from bandipy import datasets
## To reduce categorical product features into a unique id.
from category_encoders.hashing import HashingEncoder
## To encode pairs of integers as single integer values using the Cantor pairing algorithm
import pairing as pf
## For synthetic data generation
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras import backend as K

## Class instances
plotter = plotter.Plotter()
mldata = datasets.MultiLabelDatasets()
criteodata = datasets.CriteoDatasets()

## To show where exactly a warning has happend
# import warnings
# warnings.filterwarnings('error')
## For the sake of reproducibility
random.seed(0)
np.random.seed(0)
# tf.set_random_seed(0)
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)
tf.random.set_seed(0)
os.environ['PYTHONHASHSEED']=str(0)

class Simulation():
    def __init__(self, data_type, bandit_algorithm, privacy_model, sim_sig):
        self.data_type = data_type
        self.bandit_algorithm = bandit_algorithm
        self.privacy_model = privacy_model
        self.sim_sig = sim_sig
    
    def run_a_simulation(self, users, n_samples, n_actions, context_size,
                            contexts, responses, rec_policy, alpha,
                            given_agent, cb_sampling_rate, cb_neg_rew_sam_rate):
            print("\n___________________________________________________\n")
            history = dict()
            shared_data = dict()
            for u in users:
                if u%500 == 0:
                    print((int)(((u-users[0])/len(users))*100), end=" - ")
                    if u%5000 == 0:
                        print("*")
                env = environment.ContextualBanditEnv(context_s=contexts[u],
                                                    context_ns=None,
                                                    rewards=responses[u],
                                                    #rnd_seed=np.random.randint(len(users)),
                                                    rand_gen=np.random)
                if rec_policy == "LinUCB":
                    agent = policy.LinUCB(n_actions, context_size)
                elif rec_policy == "given":
                    agent = given_agent

                exp = experiment.ExperimentWithSampling(env, agent)
                exp.run_bandit(n_samples, alpha, cb_sampling_rate, cb_neg_rew_sam_rate)
                history[u] = np.array(exp.con_act_rew_hist)
                shared_data[u] = exp.shared_data

            return history, shared_data 

    def update_on_data(self, shared_contexts, shared_actions, shared_responses,
                       n_actions, context_size, rec_policy):

            contexts = np.zeros((1, context_size))
            responses = np.zeros((1, n_actions))
            env = environment.ContextualBanditEnv(context_s = contexts,
                                                  context_ns = None,
                                                  rewards = responses,
                                                  rand_gen = np.random)
            if rec_policy == "LinUCB":
                agent = policy.LinUCB(n_actions, context_size)

            exp = experiment.ExperimentWithSampling(env, agent)
            exp.run_bandit_on_data(shared_contexts, shared_actions, shared_responses)

            shared_model= agent.get_model()

            return shared_model 

    def prepare_shared_data(self, shared_data, context_size, n_actions):
        shared_contexts = np.zeros((0, context_size))
        shared_actions = np.zeros(0)
        shared_responses = np.zeros(0)
        for d in shared_data:
            tmp = np.array(shared_data[d])
            if len(tmp)!=0:
                s, e = 0, context_size
                shared_contexts = np.append(shared_contexts, tmp[:, s:e], axis=0)
                shared_actions = np.append(shared_actions, tmp[:, -2], axis=0)
                shared_responses = np.append(shared_responses, tmp[:, -1], axis=0)
        return shared_contexts, shared_actions ,shared_responses


    def thresholding(self, shared_data, priv_sc_threshold, bin_size, n_actions):
        freqs_contexts = np.zeros(2**bin_size)
        th_shared_data = {k: v for k, v in shared_data.items() if len(v)!=0}
        for u in th_shared_data:
            for d in th_shared_data[u]:
                freqs_contexts[int(d[0])] +=1
        n_removed = 0
        for u in th_shared_data:
            for i, d in enumerate(th_shared_data[u]):
                if freqs_contexts[int(d[0])] < priv_sc_threshold:
                    th_shared_data[u].pop(i)
                    n_removed+=1
        print("Number of Removed on Shuffler: ", n_removed)
        return th_shared_data

    def normalize_and_bound(self, data, dec_digits=1):
        data = data.copy()
        for i in range(len(data)):
            nd = None
            if data[i].sum() != 0:
                nd = data[i]/data[i].sum()
            else:
                nd = np.array([1/len(data[i])]*len(data[i])) 
            data[i] = iteround.saferound(nd, dec_digits)
        return data    
    

    """
    data_type:
        - 'syn' for synthetic data,
        - 'mlc' for multi-labe classification data, 
        - 'ads' for ad recommendations data.
    """
    def run_simulation(self, n_users, early_frac,
                        n_samples, n_actions, context_size, with_data=False, **kwargs):
        if with_data == False:    
            if self.data_type == 'syn':
                ctr_scaling_factor = kwargs['ctr_scaling_factor']
                resp_noise_level = kwargs['resp_noise_level']
                mapping_function = kwargs['mapping_function']
                data_builder = datasets.Synthetic(mapping_function)
                contexts , responses = data_builder.generate_data(n_users, n_samples, 
                                                                n_actions, context_size,
                                                                ctr_scaling_factor, resp_noise_level)
            elif self.data_type == 'mlc':
                if kwargs['mlc_dataset'] == "mediamill":
                    if  n_samples*n_users > 43000:
                        print("This dataset does not include that much data. Pelase choose 'n_samples'*'n_users' < 43000") 
                        return None
                    if n_actions > 40 or context_size > 40: 
                        print("Please choose 'n_actions' and 'context_size' <= 40! (Or contribute by modifying the code :))") 
                        return None
                    contexts_mm1, contexts_mm2, responses = mldata.splitted_mediamill(N=n_users,
                                                                                    red_K= n_actions, 
                                                                                    shuffle=False,
                                                                                    verbose=False,
                                                                                    focus="context")
                    contexts = list()
                    for u in range(len(contexts_mm1)):
                        if context_size > 10:
                            contexts.append(np.concatenate((contexts_mm1[u],
                                                            contexts_mm2[u][:,:context_size-10]),
                                                            axis=1))
                        else:
                            contexts.append(np.array(contexts_mm1[u][:,:context_size]))
                    
                    for i in range(len(contexts)):
                        contexts[i] = self.normalize_and_bound(contexts[i])
                    
                elif kwargs['mlc_dataset'] == "tmc":
                    if  n_samples*n_users > 28000:
                        print("This dataset does not include that much data. Pelase choose 'n_samples'*'n_users' < 28000") 
                        return None
                    if n_actions > 22: 
                        print("Please choose 'n_actions' <= 22 (The dataset does not support more than this)") 
                        return None
                    contexts, _, responses = mldata.splitted_tmc(N=n_users, Km= context_size,
                                                                Ksm=1,
                                                                shuffle=False,
                                                                verbose=False,
                                                                focus="context")
                    if n_actions < 22:
                        for u in range(len(responses)):
                            responses[u] = responses[u][:,:n_actions]
                    
                    for i in range(len(contexts)):
                        contexts[i] = self.normalize_and_bound(contexts[i])

            elif self.data_type == 'ads':
                if kwargs['ads_dataset'] == "criteo_kaggle":
                    if context_size != 10:
                        print("For this dataset you can only have 'context_size'==10.")
                        return None
                    cr_f_name = kwargs['ads_f_name']
                    
                    if kwargs['ads_build'] == True:
                        prc = n_actions
                        data_file = pd.read_csv("Criteo/"+cr_f_name+".csv")
                        X, y = criteodata.create_tabular_dataset(data_file,
                                                                prc=prc,
                                                                verbose=False)
                        X.to_csv("Criteo/X_"+cr_f_name+".csv", index=False)
                        pd.DataFrame(y).to_csv("Criteo/y_"+cr_f_name+".csv", index=False)
                    
                    X = pd.read_csv("Criteo/X_"+cr_f_name+".csv")
                    y = pd.read_csv("Criteo/y_"+cr_f_name+".csv")
                    # ############
                    y = pd.get_dummies(y["prc"])                
                    contexts, _, responses = criteodata.splitted_criteo(X= X.values,
                                                                        y= y.values,
                                                                        N=n_users,
                                                                        Kp = 0, 
                                                                        shuffle = False, 
                                                                        verbose = False, 
                                                                        focus="context")

                    for i in range(len(contexts)):
                        contexts[i] = self.normalize_and_bound(contexts[i])
            else:
                print("Wrong Data Type")
                return None 
            
            print("Data Info:")
            print("Shapes:", contexts[0].shape, responses[0].shape)
            print("Context_Tail:", contexts[0][-2:])
            print("Response_Tail:", responses[0][-2:])
            print("Sum of Responses:", responses[0].sum(axis=0))
        else:
            print("Continue With the Same Data")
            contexts = kwargs['contexts']
            responses = kwargs['responses']

        

        if self.bandit_algorithm == 'contextual_linear_ucb':
            alpha = kwargs['alpha'] ## Explore_Exploit parameter
            exp_algo = "LinUCB"
        
        if self.privacy_model == 'crowd_blending_with_sampling':
            ## Sampling from all the users with this probability
            cb_sampling_rate = kwargs['cb_sampling_rate']
            cb_neg_rew_sam_rate = kwargs['neg_rew_sam_rate']
            ## Remove shared data with frequency less than these values
            cb_context_threshold = kwargs['cb_context_threshold']
            #cb_response_threshold = kwargs['cb_response_threshold']
        

        """
        reports :         mean    std
                        ---------------
        cold_non_pri    |  0,0 |  0,1 |
                        ---------------
        warm_non_pri    |  1,0 |  1,1 |
                        ---------------
        warm_private    |  2,0 |  3,1 |
                        ---------------
        """
        reports = np.zeros((3,2))
        
        ## Dividing users into early and late users
        U_early = np.arange(0, (int)(early_frac * n_users))
        U_late = np.arange((int)(early_frac * n_users), n_users)
        
       
        ### Early ###
        _, early_shared_data = self.run_a_simulation( U_early, 
                                                    n_samples, n_actions, context_size,
                                                    contexts, responses, exp_algo, alpha, given_agent=None,
                                                    cb_sampling_rate = cb_sampling_rate,
                                                    cb_neg_rew_sam_rate = cb_neg_rew_sam_rate)

        ### Server ###    
        shared_contexts, shared_actions, shared_responses = self.prepare_shared_data(early_shared_data, context_size, n_actions)        
        if len(shared_contexts) == 0:
            print("\n---!!!No data received at the Server side!!!---\n")
            return reports
        else:
            print("Dimension of received data at the Server side:")
            print("Shapes:",shared_contexts.shape, shared_actions.shape, shared_responses.shape)
            # print("Shared Action Count:",np.bincount(shared_actions.astype(int)))
        
        shared_model = self.update_on_data(shared_contexts, shared_actions, shared_responses,
                                           n_actions, context_size, exp_algo)

        ### Late Cold ###
        late_cold_history, _ = self.run_a_simulation(U_late,
                                                n_samples, n_actions, context_size,
                                                contexts, responses, exp_algo, alpha,
                                                given_agent=None,
                                                cb_sampling_rate = 0.0,
                                                cb_neg_rew_sam_rate = 0.0)
        reports[0] = plotter.get_ctr(late_cold_history)
        
        ### Late Warm ###
        if self.bandit_algorithm == 'contextual_linear_ucb':
            warm_agent = policy.LinUCB(n_actions, context_size)
            warm_agent.A = shared_model[0]
            warm_agent.b = shared_model[1]
        late_warm_history, _ = self.run_a_simulation(U_late,
                                                        n_samples, n_actions, context_size,
                                                        contexts, responses, "given", alpha,
                                                        given_agent = warm_agent,
                                                        cb_sampling_rate = 0.0,
                                                        cb_neg_rew_sam_rate = 0.0) 
        reports[1] = plotter.get_ctr(late_warm_history)                  

        dec_digits = kwargs['dec_digits'] ## Number of decimal points for rounding
        bin_size = kwargs['bin_size'] ## Size of the binary code
        save_dir = "encoders_repo"
        f_name = "hasher_"+str(context_size)+"_"+str(dec_digits)+"_"+str(bin_size)
        re_indexer = np.load(save_dir+"/re_indexer_"+f_name+".npy")
        with open(save_dir+"/kmeans_"+f_name+".pkl", 'rb') as fid:
            kmeans = pickle.load(fid)

        enc_contexts = list()
        for u in range(n_users):
            tmp_contexts = np.zeros((len(contexts[u]), 1))
            for d in range(len(contexts[u])):
                tmp_contexts[d] = re_indexer[kmeans.predict([contexts[u][d]])[0]]
            enc_contexts.append(tmp_contexts)
        
        #### Private ####
        enc_context_size = 1

        ### Early Encoded ###
        _, early_shared_enc_data = self.run_a_simulation( U_early, 
                                                    n_samples, n_actions, enc_context_size,
                                                    enc_contexts, responses, exp_algo, alpha,
                                                    given_agent=None,
                                                    cb_sampling_rate = cb_sampling_rate,
                                                    cb_neg_rew_sam_rate = cb_neg_rew_sam_rate)

        ### Server Encoded ###
        th_shared_data = self.thresholding(early_shared_enc_data,cb_context_threshold, bin_size, n_actions)
        th_enc_shared_contexts, th_enc_shared_actions,  th_enc_shared_responses = self.prepare_shared_data(th_shared_data, 
                                                                                    enc_context_size, n_actions)        
        if len(th_enc_shared_contexts) == 0:
            print("\n---!!!No encoded data received at the Server side!!!---\n")
            return reports
        else:
            print("Dimension of received encoded data at the Server side:")
            print("Shapes encoded:",th_enc_shared_contexts.shape, th_enc_shared_actions.shape, th_enc_shared_responses.shape)
            print("Shared Encoded Action Count:", np.bincount(th_enc_shared_actions.astype(int)))
        
        enc_shared_model = self.update_on_data(th_enc_shared_contexts, th_enc_shared_actions, th_enc_shared_responses,
                                            n_actions, enc_context_size, exp_algo)
        # ### Encoded Late Cold ###
        # enc_late_cold_history, _ = self.run_a_simulation(U_late,
        #                                         n_samples, n_actions, enc_context_size,
        #                                         enc_contexts, responses, exp_algo, alpha,
        #                                         given_agent=None,
        #                                         cb_sampling_rate = 0.0,
        #                                         cb_neg_rew_sam_rate = 0.0)
        # reports[2] = plotter.get_ctr(enc_late_cold_history)
        
        ### Encoded Late Warm ###
        if self.bandit_algorithm == 'contextual_linear_ucb':
            enc_warm_agent = policy.LinUCB(n_actions, enc_context_size)
            enc_warm_agent.A = enc_shared_model[0]
            enc_warm_agent.b = enc_shared_model[1]

        enc_late_warm_history, _ = self.run_a_simulation(U_late,
                                                        n_samples, n_actions, enc_context_size,
                                                        enc_contexts, responses, "given", alpha,
                                                        given_agent = enc_warm_agent,
                                                        cb_sampling_rate = 0.0,
                                                        cb_neg_rew_sam_rate = 0.0) 
        reports[2] = plotter.get_ctr(enc_late_warm_history)

        if self.data_type != 'syn':
            return reports, contexts, responses
        return reports
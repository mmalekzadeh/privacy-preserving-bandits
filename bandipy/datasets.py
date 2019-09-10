import numpy as np
import matplotlib.pyplot as plt
import pairing as pf

from skmultilearn.cluster import MatrixLabelSpaceClusterer
from sklearn.cluster import KMeans
from skmultilearn.dataset import load_dataset
from category_encoders.hashing import HashingEncoder
from sklearn.feature_selection import VarianceThreshold

###____________  CriteoDatasets ____________###
class CriteoDatasets():
    def __init__(self):
        self.datasets_list = ['criteo_kaggle']

    def create_tabular_dataset(self, train, prc = 10, verbose=False):
        
        def pairit(data):
            p1 = pf.pair(data["col_0"],data["col_1"])
            p2 = pf.pair(p1,data["col_2"])
            return pf.pair(p1,p2)

        def code_by_freq(data):
            code = np.where(top_val_count[:,1] == data['col'])[0][0]
            return code
        
        ce_hash = HashingEncoder(cols = list(train.columns[12:]), n_components=3, verbose=1, drop_invariant=True,  hash_method='md5')
        tmp_data = ce_hash.fit_transform(train[train.columns[2:]])
        tmp_data["Label"] = train["Label"] 
        train = tmp_data

        train["col"] =  train.apply(pairit, axis=1)

        bc = np.bincount(train["col"].values.astype(int))
        nz = np.nonzero(bc)[0]
        val_count = np.sort(np.array([list(a) for a in zip(nz,bc[nz])]))
        val_count = val_count[val_count[:,0].argsort()]
        vc_mean = val_count[:,0].mean()
        top_val_count = (val_count[val_count[:,0] > vc_mean][::-1])

        train = train.drop(columns=["col_0", "col_1", "col_2"])
        train = train[train["col"].isin(top_val_count[:,1][:prc])]


        train["prc"] =  train.apply(code_by_freq, axis=1)
        train =  train.drop(columns=["col"])
        if verbose:
            print(train["prc"].value_counts())

        tmp = list(train.columns[:-2])
        tmp.extend(["prc","Label"])
        train = train.reindex(columns=tmp)
        train = train.rename(index=str, columns={"I2":"f1",
                                    "I3":"f2",
                                    "I4":"f3",
                                    "I5":"f4",
                                    "I6":"f5",
                                    "I7":"f6",
                                    "I8":"f7",
                                    "I9":"f8",
                                    "I11":"f9",
                                    "I13":"f10"})

        if verbose:
            print(train.head())

        train = train.drop(columns=["Label"])
        train = train.reset_index(drop=True)
        train.head()

        X = train[train.columns[:-1]]
        y = train[train.columns[-1]]

        return X, y

        
    def partition_features(self, K, X, y):
        
        #selector = SelectKBest(mutual_info_classif, k='all')
        #selector.fit(X, y)
        #selected_features = list(selector.scores_)
        #most_feat = set((np.argpartition(selected_features, -K)[-K:]))
        #least_feat = set(range(X.shape[1])).difference(most_feat)
        #most_feat = np.array(sorted(list(most_feat)))
        #least_feat = np.array(sorted(list(least_feat)))
        vt = VarianceThreshold()
        vt.fit(X)
        idx = np.argpartition(vt.variances_, K)
        most_feat = idx[K:]
        least_feat = idx[:K]
        return most_feat, least_feat

    def splitted_criteo(self, X, y, N, Kp, shuffle = False, verbose = True, focus="pref"):
        K = X.shape[1]
        if verbose:
            print("K = ", Kp)
        most_feat, least_feat = self.partition_features(Kp, X, y)
        if verbose:
            print("\nMost Relevant ",Kp, " Features:", most_feat)
            print("\nLeast Relevant ", K-Kp ," Features:", least_feat)
            print("Focus on: ", focus)
        if focus == "pref":
            pref = X[:,most_feat]
            context = X[:,least_feat]

        elif focus == "context":
            context = X[:,most_feat] 
            pref = X[:,least_feat]
        
        if shuffle:
            c = list(zip(context, pref, X, y))
            np.random.shuffle(c)
            context, pref, y = zip(*c)
            context = np.array(context).squeeze(axis=1)
            pref = np.array(pref).squeeze(axis=1)
            y = np.array(y).squeeze(axis=1)
        else:
            context = np.array(context)
            pref = np.array(pref)
            y = np.array(y)
        if verbose:
            print("\n Contexts Shape: ", context.shape)
            print("\n Preferences Shape: ", pref.shape)
            print("\n Responses Shape: ", y.shape)

    
        if verbose:
            plt.rcParams["figure.figsize"] = 16,4
            plt.bar(range(context.shape[1]), np.asarray(context.sum(axis=0)/context.sum()), label = "Contexts")
            plt.legend(prop={'size': 20})
            plt.show()
            plt.bar(range(pref.shape[1]), np.asarray(pref.sum(axis=0)/pref.sum()), label = "Preferences")
            plt.legend(prop={'size': 20})
            plt.show()


        sp_context = np.array_split(context, N)
        sp_pref = np.array_split(pref, N)
        sp_response = np.array_split(y, N)

        return sp_context, sp_pref, sp_response


###____________  MultiLabelDatasets ____________###
class MultiLabelDatasets():
    def __init__(self):
        self.datasets_list = ['mediamill']

    def partition_features(self, K, X, y):
        from sklearn.feature_selection import chi2, SelectKBest
        selected_features = [] 
        for label in range(y.shape[1]):
            selector = SelectKBest(chi2, k='all')
            selector.fit(X, y[:,label])
            selected_features.append(list(selector.scores_))
        avg_selected_features = np.mean(selected_features, axis=0) 
        most_feat = set((np.argpartition(avg_selected_features, -K)[-K:]))
        least_feat = set(range(X.shape[1])).difference(most_feat)
        most_feat = np.array(sorted(list(most_feat)))
        least_feat = np.array(sorted(list(least_feat)))
        return most_feat, least_feat

    def splitted_mediamill(self, N, red_K, shuffle = False, verbose=True, focus="pref"):
        X_tr, y_tr, feature_names, label_names = load_dataset('mediamill', 'train')
        X_te, y_te, _, _ = load_dataset('mediamill', 'test')

        X_tr = X_tr.todense()
        X_te = X_te.todense()
        y_tr = y_tr.todense()
        y_te = y_te.todense()
        if verbose:
            print("Shape of Train Data: ", X_tr.shape)
            print("Shape of Test Data: ", X_te.shape)
            print("Shape of Train Labels: ", y_tr.shape)
            print("Shape of Test Data: ", y_te.shape)
        
        X = np.concatenate((X_tr, X_te), axis=0)
        y = np.concatenate((y_tr, y_te), axis=0)
        y = y[:,np.asarray(y.sum(axis=0) > 100)[0]]
        if verbose:
            print("Shape of All Data:", X.shape)
            print("Shape of All Labels:", y.shape)
        
        K= y.shape[1]
        most_feat, least_feat = self.partition_features(K, X, y)
        if verbose:
            print("\nMost Relevant ",K, " Features:", most_feat)
            print("\nLeast Relevant ", X.shape[1]-K ," Features:", least_feat)

        if focus == "pref":
            pref = (X[:,most_feat] > 0.45).astype(float)
            context = (X[:,least_feat] > 0.45).astype(float)
            if verbose:
                print("\n Preferences Shape: ", pref.shape)
                print("\n Contexts Shape: ", context.shape)
            
            pref = pref[:,np.asarray(pref.sum(axis=0) > 2400)[0]]
            y = y[:,np.asarray(y.sum(axis=0) > 450)[0]]
            context = context[:,np.asarray(np.logical_and(context.sum(axis=0) > 2000 , context.sum(axis=0) < 40000))[0]]

        
        elif focus == "context":
            context = (X[:,most_feat] > 0.45).astype(float)
            pref = (X[:,least_feat] > 0.45).astype(float)
            if verbose:
                print("\n Preferences Shape: ", pref.shape)
                print("\n Contexts Shape: ", context.shape)
            
            pref = pref[:,np.asarray(pref.sum(axis=0) > 500)[0]]
            y = y[:,np.asarray(y.sum(axis=0) > 450)[0]]
            context = context[:,np.asarray(np.logical_and(context.sum(axis=0) > 9999 , context.sum(axis=0) < 29000))[0]]
            
        if shuffle:
            c = list(zip(context, pref, y))
            np.random.shuffle(c)
            context, pref, y = zip(*c)
            context = np.array(context).squeeze(axis=1)
            pref = np.array(pref).squeeze(axis=1)
            y = np.array(y).squeeze(axis=1)
        else:
            context = np.array(context)
            pref = np.array(pref)
            y = np.array(y)
        if verbose:
            print("\n Contexts Shape: ", context.shape)
            print("\n Preferences Shape: ", pref.shape)
            print("\n Actions Shape: ", y.shape)
        
        
        matrix_clusterer = MatrixLabelSpaceClusterer(clusterer=KMeans(n_clusters=red_K))
        similar_ys = matrix_clusterer.fit_predict(context, y)
        if verbose:
            print("Silimar Labeles: ", similar_ys)
        
        y_red = np.zeros((y.shape[0],red_K))
        for k, lbs in enumerate(similar_ys):
            for lb in lbs:
                y_red[:,k] += y[:,lb]
        y_red = (y_red >=1).astype(float)
    
        if verbose:
            plt.rcParams["figure.figsize"] = 16,4
            plt.bar(range(context.shape[1]), np.asarray(context.sum(axis=0)), label = "Contexts")
            plt.legend(prop={'size': 20})
            plt.show()
            plt.bar(range(pref.shape[1]), np.asarray(pref.sum(axis=0)), label = "Preferences")
            plt.legend(prop={'size': 20})
            plt.show()
            plt.bar(range(y_red.shape[1]), np.asarray(y_red.sum(axis=0)), label = "Responses")
            plt.legend(prop={'size': 20})
            plt.show()

        sp_context = np.array_split(context, N)
        sp_pref = np.array_split(pref, N)
        sp_response = np.array_split(y_red, N)
            
        return sp_context, sp_pref, sp_response
    

    def splitted_tmc(self, N, Km, Ksm, shuffle = False, verbose=True, focus="pref"):
        X_tr, y_tr, feature_names, label_names = load_dataset('tmc2007_500', 'train')
        X_te, y_te, _, _ = load_dataset('tmc2007_500', 'test')

        X_tr = X_tr.todense()
        X_te = X_te.todense()
        y_tr = y_tr.todense()
        y_te = y_te.todense()
        if verbose:
            print("Shape of Train Data: ", X_tr.shape)
            print("Shape of Test Data: ", X_te.shape)
            print("Shape of Train Labels: ", y_tr.shape)
            print("Shape of Test Data: ", y_te.shape)

        X = np.concatenate((X_tr, X_te), axis=0)
        y = np.concatenate((y_tr, y_te), axis=0)

        most_feat, least_feat = self.partition_features(Km, X, y)
        if verbose:
            print("\nMost Relevant ",Km, " Features:", most_feat)
            #print("\nLeast Relevant ", X.shape[1]-Km ," Features:", least_feat)

        red_X = X.copy()
        red_X[:, most_feat] = 1

        red_most_feat, red_least_feat = self.partition_features(Ksm, red_X, y)
        if verbose:
            print("\nSecond Most Relevant ",Ksm, " Features:", red_most_feat)
            #print("\nLeast Relevant ", X.shape[1]-Ksm ," Features:", red_least_feat)

        if focus == "pref":
            pref = X[:,most_feat]
            context = X[:,red_most_feat]
        elif focus == "context":
            context = X[:,most_feat] 
            pref = X[:,red_most_feat]
            
        if verbose:
            print("\n Preferences Shape: ", pref.shape)
            print("\n Contexts Shape: ", context.shape)


        if shuffle:
            c = list(zip(context, pref, y))
            np.random.shuffle(c)
            context, pref, y = zip(*c)
            context = np.array(context).squeeze(axis=1)
            pref = np.array(pref).squeeze(axis=1)
            y = np.array(y).squeeze(axis=1)
        else:
            context = np.array(context)
            pref = np.array(pref)
            y = np.array(y)
        if verbose:
            print("\n Contexts Shape: ", context.shape)
            print("\n Preferences Shape: ", pref.shape)
            print("\n Actions Shape: ", y.shape)

        if verbose:
            plt.rcParams["figure.figsize"] = 16,4
            plt.bar(range(context.shape[1]), np.asarray(context.sum(axis=0)), label = "Contexts")
            plt.legend(prop={'size': 20})
            plt.show()
            plt.bar(range(pref.shape[1]), np.asarray(pref.sum(axis=0)), label = "Preferences")
            plt.legend(prop={'size': 20})
            plt.show()
            plt.bar(range(y.shape[1]), np.asarray(y.sum(axis=0)), label = "Responses")
            plt.legend(prop={'size': 20})
            plt.show()

        sp_context = np.array_split(context, N)
        sp_pref = np.array_split(pref, N)
        sp_response = np.array_split(y, N)

        return sp_context, sp_pref, sp_response

class Synthetic():
    def __init__(self, mapping_function):
        self.mapping_function = mapping_function
    
    def make_hists(self, n_samples, hist_size):
        hists = np.random.rand(n_samples, hist_size)
        hists_sum = np.sum(hists, axis=1 )
        for i in range(len(hists)):
            hists[i,:] = hists[i,:]/hists_sum[i]
        return hists

    def generate_data(self, n_users, n_samples, n_actions, context_size, 
                            ctr_scaling_factor=1., noise_level=.01):
        contexts = list()
        responses = list()
        for u in range(n_users):
            hists = self.make_hists(n_samples, context_size)
            resps_probs = self.mapping_function.predict(hists)
            resps_probs += noise_level * np.random.normal(0,1, (n_samples, n_actions))
            resps_probs = ctr_scaling_factor * resps_probs
            resps_probs = np.clip(resps_probs,0,1)
            resps = np.zeros((n_samples, n_actions), dtype=int)
            for s in range(n_samples):
                for a in range(n_actions):
                    resps[s, a] = (np.random.rand() <= resps_probs[s, a]).astype(int)
            contexts.append(hists)
            responses.append(resps)
        
        return contexts, responses
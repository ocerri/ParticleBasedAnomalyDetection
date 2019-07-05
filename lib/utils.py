#['EvtId', 'Energy', 'Px', 'Py', 'Pz', 'Pt', 'Eta', 'Phi', 'vtxX', 'vtxY', 'vtxZ', 'ChPFIso', 'GammaPFIso', 'NeuPFIso', 'isChHad', 'isNeuHad', 'isGamma', 'isEle', 'isMu', 'Charge']
import numpy as np

class Bauble():
    def __init__(self):
        pass

class EarlyStopping():
    def __init__(self, patient=5):
        self.best = None
        self.NoImprovement = 0
        self.patient = patient

    def check(self, val):
        if self.best is None:
            self.best = val
            self.NoImprovement = 0
        else:
            if self.best > val:
                self.best = val
                self.NoImprovement = 0
            else:
                self.NoImprovement += 1

        if self.NoImprovement > self.patient:
            print('Early Stopping triggered')
            return False
        else:
            return True

def createROC_curve(dataset):
    loss_dic = dataset.loss
    weight_dic = dict(zip(dataset.SM_names, dataset.SM_val_weights))

    p_SM = np.logspace(base=10, start=-7, stop=0, num=100)
    p_SM[-1] = 0.999

    t_SM = np.concatenate((loss_dic['Wlnu'],
                           loss_dic['qcd'],
                           loss_dic['Zll'],
                           loss_dic['ttbar']
                          ))

    w_SM = np.concatenate((np.full_like(loss_dic['Wlnu'], weight_dic['Wlnu'], np.float128),
                           np.full_like(loss_dic['qcd'], weight_dic['qcd'], np.float128),
                           np.full_like(loss_dic['Zll'], weight_dic['Zll'], np.float128),
                           np.full_like(loss_dic['ttbar'], weight_dic['ttbar'], np.float128)
                          ))

    i_sort = np.argsort(t_SM)

    t_SM = t_SM[i_sort]
    w_SM = w_SM[i_sort]

    l = np.zeros(4)
    for i,n in enumerate(dataset.SM_names):
            l[i] = dataset.valSamples[n].shape[0]
    i_min = np.argmin(l/dataset.SM_fraction)

    cum_sum = np.cumsum(w_SM, dtype=np.float128)/np.float128(l[i_min]/dataset.SM_fraction[i_min])
    print('CumSum accuracy:', cum_sum[-1])

    idx_q = np.argmax(cum_sum>np.atleast_2d(1-p_SM).T, axis=1)
    q_SM = t_SM[idx_q]

    dic_ROC = {}
    for n in dataset.BSM_names:
        out = loss_dic[n] > np.atleast_2d(q_SM).T
        p_BSM = np.float64(np.sum(out, axis=1, dtype=np.float128)/loss_dic[n].shape[0])

        roc_auc = np.trapz(p_BSM, p_SM)

        dic_ROC[n] = {'eff_BSM':p_BSM, 'eff_SM':p_SM, 'roc_auc':roc_auc}

    return dic_ROC

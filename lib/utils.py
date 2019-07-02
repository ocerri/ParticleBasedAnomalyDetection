#['EvtId', 'Energy', 'Px', 'Py', 'Pz', 'Pt', 'Eta', 'Phi', 'vtxX', 'vtxY', 'vtxZ', 'ChPFIso', 'GammaPFIso', 'NeuPFIso', 'isChHad', 'isNeuHad', 'isGamma', 'isEle', 'isMu', 'Charge']

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
            return False
        else:
            return True

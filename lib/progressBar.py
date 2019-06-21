import time, sys

class ProgressBar():
    def __init__(self, maxEntry, percentPrecision = 5):
        self.maxEntry = maxEntry
        self.percentPrecision = percentPrecision

        nStep = int(100/percentPrecision)
        self.nStep = nStep if nStep <= maxEntry else maxEntry
        self.setpSize = int(maxEntry/self.nStep)

    def show(self, entry):
        if entry%self.setpSize==0:
            if entry>0:
                sys.stdout.write('\r')
            else:
                self.startTime = time.time()

            Progress = float(entry)/self.maxEntry
            nStepDone = int(Progress*self.nStep)

            outLine = '['+'#'*nStepDone + '-'*(self.nStep-nStepDone) +']'+'  {}%'.format(int(100*Progress))

            if entry>0:
                timeleft = (self.maxEntry - float(entry))*(time.time() - self.startTime)/float(entry)
                if timeleft<181:
                    outLine += " - ETA:{:5.0f} s   ".format(timeleft)
                elif timeleft<10801:
                    timeleft/=60
                    outLine += " - ETA:{:5.1f} min ".format(timeleft)
                else:
                    timeleft/=3600
                    outLine += " - ETA:{:5.1f} h   ".format(timeleft)

            sys.stdout.write(outLine)
            sys.stdout.flush()

        if entry==self.maxEntry-1:
            outLine = '\r['+ '#'*self.nStep +']  100%'+'- Tot. time: {:.1f} s'.format(time.time() - self.startTime)
            sys.stdout.write(outLine)
            sys.stdout.flush()

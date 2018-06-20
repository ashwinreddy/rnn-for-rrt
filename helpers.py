import numpy as np

numSteps = lambda seq: seq[:,2].tolist().index(1)

seqToPath = lambda seq: np.cumsum(np.delete(seq, 2, 1), axis=0)
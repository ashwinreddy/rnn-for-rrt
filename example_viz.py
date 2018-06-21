import matplotlib.pyplot as plt
from helpers import processData
import numpy as np

trainingInputs, trainingOutputs = processData("training")

# print(trainingInputs.shape)
# print(trainingOutputs.shape)

exampleNumber = 0

start = trainingInputs[:,0,:2][exampleNumber]
end = trainingInputs[:,0,2:-1][exampleNumber]

samples = np.vstack((start, np.cumsum(trainingOutputs[:,:,:-1], axis=1)[exampleNumber]+start))

print(samples)

# plt.plot(samples[:,0], samples[:,1])
# plt.show()
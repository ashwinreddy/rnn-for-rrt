import numpy as np

def padSequence(sequence, Nmax):
    steps = len(sequence)

    sequence = np.hstack(
        (
            sequence,
            np.zeros(
                (steps, 1)
            )
        )
    )

    no_steps = np.tile(  [0,0,1], (Nmax - steps, 1) )

    return np.vstack(   (sequence, no_steps)  )

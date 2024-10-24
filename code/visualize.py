import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def drawMFCC(data,idx):
    fig, ax = plt.subplots()
    mfcc_data = np.swapaxes(data, 0, 1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    ax.set_title('MFCC')
    plt.savefig(f'fig/{idx}.png')
    #plt.show()
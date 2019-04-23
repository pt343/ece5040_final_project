import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from math import sqrt

#==================================================#
# Global variables
#==================================================#
fs          = 5000           # sampling frequency
frame_time  = 1              # frame of 1 second
frame_size  = frame_time*fs  # points in one frame

#==================================================#
# Plotting functions
#==================================================#
def plot_time(arr):
    """
    plots the given array vs. time
    """
    num_data_pts = len(arr)
    time = np.array([t/fs for t in range(num_data_pts)])

    plt.plot(time, arr)
    plt.show()


def plot_algorithm(arr):
    """
    plots data from a given algorithm
    """
    num_data_pts = len(arr)
    time = np.array([t for t in range(num_data_pts)])

    plt.plot(time,arr)
    plt.show()

def plot_algorithms(channel, alg_dict):
    """
    Channel = channel number
    alg_dict contains key=algorithm name and value=array

    Plots an array with each algorithm as a subplot
    """
    f, ax = plt.subplots(len(alg_dict),sharex=True)
    f.suptitle('Channel {}'.format(channel), fontsize=16)
    j = 0
    for alg,arr in alg_dict.items():
        ax[j].plot(arr)
        ax[j].set_title(alg, fontsize=10)
        j+=1
    ax[j-1].set_xlabel('Time (s)')
    plt.subplots_adjust(hspace=0.6)
    plt.show()

#==================================================#
# Feature characterization algorithms
#==================================================#
def all_algorithms(arr):
    """
    In order to avoid wasting computing time running through a for loop for a given array multiple times,
    this function applies all of the algorithms for a given array

    ll: line length
    en: energy
    var: variance
    beta: spectral power beta
    hfo: spectral power hfo
    """
    num_data_pts = len(arr)

    # allocate memory for each of the algorithm arrays
    ll_arr       = np.empty([num_data_pts/frame_size, 1])
    en_arr       = np.empty([num_data_pts/frame_size, 1])
    var_arr      = np.empty([num_data_pts/frame_size, 1])
    beta_arr     = np.empty([num_data_pts/frame_size, 1])
    hfo_arr      = np.empty([num_data_pts/frame_size, 1])

    algorithms = {
        '1. Line Length': ll_arr,
        '2. Energy': en_arr,
        '3. Variance': var_arr,
        '4. Spectral Power (Beta)': beta_arr,
        '4. Spectral Power (HFO)': hfo_arr
    }

    # iterate through each frame, truncating at the last one
    for i in range(0,num_data_pts-frame_size,frame_size):
        ll  = 0
        en  = 0
        mu  = 0
        var = 0

        # iterate through individual frame
        for j in range(i,i+frame_size):
            if j!=i: ll += abs(arr[j]-arr[j-1])     # line length
            en += (arr[j]/sqrt(frame_size))**2      # energy
            mu += arr[j]                            # mean for variance calculation

        # need to iterate through array again to calculate variance
        mu /= frame_size
        for j in range(i+1,i+frame_size):
            var += (arr[j]-mu)**2

        # fft for power calculations
        freq_power = np.fft.fft(arr[i:i+frame_size])
        freq_power = abs(freq_power)**2

        # update arrays
        ll_arr[i/frame_size]    = ll
        en_arr[i/frame_size]    = en
        var_arr[i/frame_size]   = var
        beta_arr[i/frame_size]  = np.sum(freq_power[12:30])
        hfo_arr[i/frame_size]   = np.sum(freq_power[100:600])
    return algorithms



def line_length(arr):
    """
    The line length algorithm sums the absolute value of the difference between
    adjacent points for a given frame
    """

    num_data_pts = len(arr)
    ll_arr = np.empty([num_data_pts/frame_size, 1])

    # iterate through each frame
    for i in range(0,num_data_pts-frame_size,frame_size):
        ll = 0
        # calculate line length for each individual frame, truncating the frame if the array is too short
        #for j in range(i+1,min(i+frame_size,num_data_pts)):
        for j in range(i+1,i+frame_size):
            ll += abs(arr[j]-arr[j-1])
        ll_arr[i/frame_size] = ll

        #np.insert(ll_arr,i,ll)
    return ll_arr


def energy(arr):
    """
    Energy algorithm sums the squared value of all points in a frame
    """
    num_data_pts = len(arr)
    en_arr = np.empty([num_data_pts/frame_size, 1])

    # iterate through each frame
    for i in range(num_data_pts/frame_size):
        en = 0
        # calculate energy for each frame, truncating if the array is too long
        for j in range(i,min(i+frame_size,num_data_pts)):
            en += (arr[j]/sqrt(frame_size))**2
        en_arr[i] = en
    return en_arr

def variance(arr):
    """
    Variance for a given frame
    """
    num_data_pts = len(arr)
    var_arr = np.empty([num_data_pts/frame_size, 1])

    # iterate through each frame
    for i in range(num_data_pts/frame_size):
        mu = 0
        var = 0

        # first calculate the mean of the sample window
        for j in range(i,min(i+frame_size,num_data_pts)):
            mu += arr[j]
        mu /= frame_size

        # now calculate the variance
        for j in range(i,min(i+frame_size,num_data_pts)):
            var += (arr[j]-mu)**2
        var /= frame_size

        var_arr[i] = var
    return var_arr

def spectral_power_beta(arr):
    """
    Calculate spectral power in 12-30Hz band
    """
    num_data_pts = len(arr)
    beta_arr = np.empty([num_data_pts/frame_size, 1])

    # iterate through each frame
    for i in range(0,num_data_pts-frame_size,frame_size):
        power_spec = np.fft.fft(arr[i:i+frame_size])
        power_spec = abs(power_spec)**2

        # add together all frequencies from 12-30Hz
        beta_arr[i/frame_size] = np.sum(power_spec[12:30])

    return beta_arr

def spectral_power_hfo(arr):
    """
    Calculate spectral power in 100-600Hz band
    """

    num_data_pts = len(arr)
    hfo_arr = np.empty([num_data_pts/frame_size, 1])

    # iterate through each frame
    for i in range(0,num_data_pts-frame_size,frame_size):
        power_spec = np.fft.fft(arr[i:i+frame_size])
        power_spec = abs(power_spec)**2

        freq = np.fft.fftfreq(frame_size, d=1.0/fs)
        #print(freq)

        # add together all frequencies from 12-30Hz

        #hfo_arr[i/frame_size] = np.sum(power_spec[])

    return hfo_arr


if __name__ == '__main__':
    # load data
    data = loadmat('data.mat')
    data = data['data']

    # array of most interesting channels
    interesting_channels = [4, 19, 21, 24]

    for i in interesting_channels:
        algs = all_algorithms(data[:,i])
        plot_algorithms(i,algs)

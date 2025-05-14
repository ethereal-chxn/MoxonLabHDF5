import h5py
import numpy as np

# Creates an EEG dataset datatype with specified number of channels
# EEG dataset structure:
# (channel name, float32 numpy array that contains channel's eeg data)
def eeg_dt(num_chs):
    return np.dtype([('channel_name', h5py.string_dtype()), ('channel_data', np.float32, num_chs)])


# Creates a spike times datatype with specified number of single unit neurons
# Datatype consists of:
# Name (string)
# Spike Times (ragged array of floats)
def spks_dt():
    return np.dtype([('single_unit_name', h5py.string_dtype()), ('times', h5py.vlen_dtype('float64'))])


def spk_waveforms_dt():
    return np.dtype([('single_unit_neuron', h5py.string_dtype()), ('waveforms', h5py.vlen_dtype(np.array(dtype=np.int32)))])


def events_dt():
    return np.dtype([('time', np.float32), ('type', h5py.string_dtype()), ('value', h5py.string_dtype())])
import h5py
import numpy as np

# Generate a HDF5 file name from a given raw data file's path regardless of if given
# absolute or relative path.
def gen_hdf5_name_from_filepath(data_fp):
    data_fp_dirs = data_fp.split('/')
    file_name = data_fp_dirs[len(data_fp_dirs) - 1]
    return file_name.split('.')[0] + ".hdf5"


# Attempts to access a specified subgroup. If subgroup doesn't exist,
# creates that subgroup.
# ===
def get_subgroup(hdf5_obj, subgroup):
    try:
        return hdf5_obj[subgroup]
    except:
        hdf5_obj.create_group(subgroup)
        return hdf5_obj[subgroup]
    

# Attempts to access a specified dataset in HDF5 file.
def get_dataset(hdf5_obj, dataset_path):
    try:
        return hdf5_obj[dataset_path]
    except:
        print("Error: {dataset_path} could not be found")
        return

def get_spike_ch(spike_name):
    electrode_ch = spike_name.split('.')[0]
    return electrode_ch


# Converts the Plexon spike name from "SPK(electrode ch#).(single unit #)" 
# format to "SPK#.(single unit letter)"
# Note: SPK#.0 -> unsorted spikes in that channel
# e.g.: SPK189.4 => SPK189.d
def format_spike_name(spike_name):
    # split spike name string into the electrode channel number and the single unit number
    electrode_ch = get_spike_ch(spike_name)
    single_unit_num = int(spike_name.split('.')[1])

    # convert single_unit_num to letter
    if single_unit_num == 0:
        single_unit = "unsorted"
    else:
        # note: 97 is 'a''s ascii number
        single_unit = chr(97 + single_unit_num - 1)

    return electrode_ch + '.' + single_unit


# Takes in a raw file's spike data and returns a hashmap    
def get_num_single_units_in_chs(spiketrains):
    single_units_hash = {}

    for spiketrain in spiketrains:
        ch_num = get_spike_ch(spiketrain.name)

        if ch_num in single_units_hash:
            single_units_hash[ch_num] += 1
        else:
            single_units_hash[ch_num] = 1
    
    return single_units_hash

# Gets number of single unit neurons with recorded spikes.
def get_num_spike_neurons(spiketrains):
    num_spk_neurons = 0

    for spiketrain in spiketrains:
        # Must check if single unit neuron contains any spike times because
        # NEO reads in every spike channel regardless of if there is
        # any actual spikes recorded in it.
        if spiketrain.magnitude.size == 0:
            continue
        num_spk_neurons += 1
    
    return num_spk_neurons

def get_max_spiketimes_len(spiketrains):
     spiketrains_lens = [spiketrain.magnitude.size for spiketrain in spiketrains]
     return np.max(np.array(spiketrains_lens))
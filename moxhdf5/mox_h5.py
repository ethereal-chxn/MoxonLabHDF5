import h5py
import numpy as np
import neo
import sys
import modules.datatype_creator as datatype_creator

# ======================
# || GLOBAL VARIABLES ||
# ======================



# Defines numpy datatype for single unit spikes.
# Datatype consists of:
# Name (string)
# Spike Times (ragged array of floats)
single_unit_dt = np.dtype([('name', h5py.string_dtype()), ('spike_times', h5py.vlen_dtype('float64'))])

# Defines the composite type for elements in the SpikeTrains dataset
spk_ch_dt = np.dtype([('ch_name', h5py.string_dtype()), ('single_unit_spks', single_unit_dt)])


# Creates an EEG dataset datatype with specified number of channels
# EEG dataset structure:
# (channel name, float32 numpy array that contains channel's eeg data)
def create_eeg_dt(num_chs):
    return np.dtype([('channel_name', h5py.string_dtype()), ('channel_data', np.float32, num_chs)])


# Creates a spike times datatype with specified number of single unit neurons
# Datatype consists of:
# Name (string)
# Spike Times (ragged array of floats)
def create_spks_dt():
    return np.dtype([('single_unit_name', h5py.string_dtype()), ('times', h5py.vlen_dtype('float64'))])


def create_spk_waveforms_dt():
    return np.dtype([('single_unit_neuron', h5py.string_dtype()), ('waveforms', h5py.vlen_dtype(np.array(dtype=np.int32)))])

# ======================
# || HELPER FUNCTIONS ||
# ======================

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


def append_to_dataset(dataset, data):
    pass

# =======================
# || STORAGE FUNCTIONS ||
# =======================

# Stores raw continuous data within the raw EEG data file's NEO object form into
# the specified HDF5 file object's Neural group.
# ===
# Note: Consult the Block, Segment, and AnalogSignal sections in 
# https://neo.readthedocs.io/en/latest/api_reference.html# to understand
# how EEG data is structured in an NEO object.
def store_raw_cont(raw_data, hdf5_obj):
    neural_group = get_subgroup(hdf5_obj, "Neural")

    print("== STORING RAW CONTINUOUS DATA ==")

    # The NEO reader extracts 3 AnalogSignals. The third one is from the FPSeparator device 
    # in the OmniPlex (AKA the raw continuous neural data).
    raw_cont = raw_data[0].segments[0].analogsignals[2]

    # Turns raw continuous signal into a HDF5-storable numpy array.
    raw_cont_np = raw_cont.magnitude
    
    num_chs = raw_cont_np.shape[1]

    raw_cont_ds = neural_group.create_dataset("RawContinuous", num_chs, dtype=datatype_creator.eeg_dt(num_chs))

    # Store each channel into the RawContinuous dataset as a tuple consisting
    # of the channel name and the channel EEG data
    ch_index = 0
    for ch in raw_cont_np:
        ch_number = ch_index + 1
        ch_name = f"Channel {ch_number}"
        ch_data = ch

        print(f"Storing {ch_name}...")

        raw_cont_ds[ch_index] = (ch_name, ch_data)
        ch_index += 1
        if (ch_index == num_chs):
            break

    # Extract metadata
    raw_cont_ds.attrs["name"] = str(raw_cont.name)
    raw_cont_ds.attrs["sample_rate"] = raw_cont.sampling_rate
    raw_cont_ds.attrs["start_time"] = raw_cont.t_start
    raw_cont_ds.attrs["duration"] = f"{float(raw_cont.duration.magnitude)} s"    


# Stores episodic data within the raw EEG data file's NEO object form into
# the specified HDF5 file object's Episodic group.
# ===
# Note: Consult the Block, Segment, and AnalogSignal sections in 
# https://neo.readthedocs.io/en/latest/api_reference.html# to understand
# how EEG data is structured in an NEO object.
def store_behavioral_data(raw_data, hdf5_obj):
    block_idx = 0
    segment_idx = 0

    behavorial_group = get_subgroup(hdf5_obj, "Behavorial")

    print("== STORING EPISODIC DATA ==")
    for block in raw_data:
        segment_idx = 0
        for segment in block.segments:
            # The NEO reader extracts 3 AnalogSignals. The first one is from an Auxillary Analog Input (AuxAI)
            # source which is episodic, non-neural data. The second one is from the AuxAI Filter device which
            # is a filtered AI source. 
            aux_ai = segment.analogsignals[0]
            aux_ai_filter = segment.analogsignals[1]
            print(f"shape of AuxAI signal in block {block_idx}, segment {segment_idx}: {aux_ai.shape}")
            print(f"shape of AuxAI Filter signal in block {block_idx}, segment {segment_idx}: {aux_ai_filter.shape}")

            aux_ai_np = aux_ai.magnitude
            aux_ai_filter_np = aux_ai_filter.magnitude

            behavorial_group["AuxAI"] = aux_ai_np
            behavorial_group["AuxAIFilter"] = aux_ai_filter_np

            # extract metadata of the raw EEG signals
            behavorial_group["AuxAI"].attrs["name"] = str(aux_ai.name)
            behavorial_group["AuxAI"].attrs['sample_rate'] = aux_ai.sampling_rate
            behavorial_group["AuxAIFilter"].attrs['name'] = str(aux_ai_filter.name)
            behavorial_group["AuxAIFilter"].attrs['sample_rate'] = aux_ai_filter.sampling_rate
            
            segment_idx += 1
        block_idx += 1
        

def store_spike_times(raw_data, hdf5_obj):
    print("== STORING SPIKE TIME ARRAYS ==")
    neural_group = get_subgroup(hdf5_obj, "Neural")

    spiketrains = raw_data[0].segments[0].spiketrains
    num_single_units = len(spiketrains)

    spiketimes_dataset = neural_group.create_dataset("SpikeTimes", num_single_units, dtype=datatype_creator.spks_dt())

    single_unit_index = 0
    for spiketrain in spiketrains:
        single_unit_name = format_spike_name(spiketrain.name)
        spike_times = spiketrain.magnitude
        spiketimes_dataset[single_unit_index] = (single_unit_name, spike_times)
        single_unit_index += 1


def store_spike_waveforms(raw_data, hdf5_obj):
    print("== STORING SPIKE WAVEFORMS ==")
    neural_group = get_subgroup(hdf5_obj, "Neural")
    waveforms_group = neural_group.create_group("SpikeWaveforms")

    spiketrains = raw_data[0].segments[0].spiketrains

    for spiketrain in spiketrains:
        single_unit_name = format_spike_name(spiketrain.name)
        spike_waveforms_dataset = waveforms_group.create_dataset(single_unit_name, \
                                                                spiketrain.waveforms.shape[0], \
                                                                dtype=h5py.vlen_dtype('float32'))
        
        print(f"Storing {single_unit_name}...")
        waveform_idx = 0
        for waveform in spiketrain.waveforms:
            spike_waveforms_dataset[waveform_idx] = waveform[0]
            waveform_idx += 1


# TODO: implement storing events
def store_events_classes(raw_data, hdf5_obj):
    print("==STORING EVENT DATA==")

    events_timestamps = get_subgroup(hdf5_obj, "events_classes/events_timestamps")

    for block in raw_data:
        for segment in block.segments:
            for event in segment.events:
                # extract the event's timestamps and put into HDF5 container
                event_timestamps = event.magnitude
                events_timestamps[f"{event.name}"] = event_timestamps
                # extract the event's metadata
                events_timestamps[f"{event.name}"].attrs['units'] = str(event.units).split()[1]
                events_timestamps[f"{event.name}"].attrs['num_occurences'] = event.shape[0]


# Stores a raw EEG file's data into the Neural and Episodic groups.
# ===
# Command format: store_data [data_fp] [hdf5_fp]

def store_data(data_fp, hdf5_fp):
    # read in raw EEG file as a NEO object
    reader = neo.io.get_io(data_fp)
    raw_data = reader.read(lazy=False, load_waveforms=True)

    # FOR DEBUGGING PURPOSES
    print(f"num of blocks in raw data: {len(raw_data)}")

    # open/create hdf5 file at given path, hdf5_fp
    hdf5_obj = h5py.File(hdf5_fp, "a")

    store_raw_cont(raw_data, hdf5_obj)
    store_spike_times(raw_data, hdf5_obj)
    store_spike_waveforms(raw_data, hdf5_obj)
    store_behavioral_data(raw_data, hdf5_obj)
    store_events_classes(raw_data, hdf5_obj)


# ====================
# || MOX_H5 METHODS ||
# ====================

# Stores a specified recording session data file into a HDF5 file.
# ===
# Command format: store_rs [data_path] [OPTIONAL: hdf5_file_name]

# TODO: get store_recording_session to work with the DRP003 txt files
def store_recording_session(params):
    data_path = params[0]

    if len(params) == 1:
        hdf5_file_name = gen_hdf5_name_from_filepath(data_path)
    else:
        hdf5_file_name = params[1]
    
    store_data(data_path, hdf5_file_name)


# To run this program:
# py ./src/mox_h5.py [command] [params]
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: py ./src/mox_h5.py [command] [params]")
        exit(1)
    
    cmd_name = sys.argv[1]
    params = sys.argv[2:]

    if cmd_name == "store_rs":
        if len(params) < 1:
            print("Usage: py ./src/mox_h5.py store_rs [data_fp] [hdf5_fp]")
            exit(1)
        store_recording_session(params)
    else:
        print(f"Error: command {cmd_name} not found")
        exit(1)
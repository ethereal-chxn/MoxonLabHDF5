import h5py
import numpy as np
import neo
import sys

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
# Note: call this function like: ___.get_subgroup(subgroup)
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


# Stores raw continuous data within the raw EEG data file's NEO object form into
# the specified HDF5 file object's Neural group.
# ===
# Note: Consult the Block, Segment, and AnalogSignal sections in 
# https://neo.readthedocs.io/en/latest/api_reference.html# to understand
# how EEG data is structured in an NEO object.
def store_raw_cont(raw_data, hdf5_obj):
    block_idx = 0
    segment_idx = 0

    neural_group = get_subgroup(hdf5_obj, "Neural")

    print("== STORING RAW CONTINUOUS DATA ==")
    for block in raw_data:
        segment_idx = 0

        # here for debugging purposes. TO BE DELETED LATER
        # print(f"num of segments in block {block_idx}: {len(block.segments)}")
        for segment in block.segments:
            # The NEO reader extracts 3 AnalogSignals. The third one is from the FPSeparator device 
            # in the OmniPlex (AKA the raw continuous neural data).
            raw_cont = segment.analogsignals[2]
            print(f"shape of RawContinuous in block {block_idx}, segment {segment_idx}: {raw_cont.shape}")

            # Turns raw continuous signal into a HDF5-storable numpy array.
            raw_cont_np = raw_cont.magnitude

            neural_group["RawContinuous"] = raw_cont_np

            segment_idx += 1
        block_idx += 1    


# Stores episodic data within the raw EEG data file's NEO object form into
# the specified HDF5 file object's Episodic group.
# ===
# Note: Consult the Block, Segment, and AnalogSignal sections in 
# https://neo.readthedocs.io/en/latest/api_reference.html# to understand
# how EEG data is structured in an NEO object.
def store_episodic_data(raw_data, hdf5_obj):
    block_idx = 0
    segment_idx = 0

    episodic_group = get_subgroup(hdf5_obj, "Episodic")

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

            episodic_group["AuxAI"] = aux_ai_np
            episodic_group["AuxAIFilter"] = aux_ai_filter_np

            segment_idx += 1
        block_idx += 1
        

def store_spikes(raw_data, hdf5_obj):
    print("== STORING SPIKES ==")
    block_idx = 0
    segment_idx = 0
    spiketrain_idx = 0

    neural_group = get_subgroup(hdf5_obj, "Neural")

    for block in raw_data:
        segment_idx = 0
        print(f"num of segments in block {block_idx}: {len(block.segments)}")
        for segment in block.segments:
            spiketrain_idx = 0
            print(f"num of SpikeTrains in block {block_idx}, segment {segment_idx}: {len(segment.spiketrains)}")
            for spiketrain in segment.spiketrains:
                # print(f"shape of SpikeTrains {spiketrain_idx} in block {block_idx}, segment {segment_idx}: {signal.shape}")
                signal_np_arr = spiketrain.magnitude
                neural_group[f"{spiketrain.name}"] = signal_np_arr
                spiketrain_idx += 1
            segment_idx += 1
        block_idx += 1


# Stores a raw EEG file's data into the Neural and Episodic groups.
# ===
# Command format: store_data [data_fp] [hdf5_fp]

def store_data(data_fp, hdf5_fp):
    # read in raw EEG file as a NEO object
    reader = neo.io.get_io(data_fp)
    raw_data = reader.read(lazy=False)

    # FOR DEBUGGING PURPOSES
    print(f"num of blocks in raw data: {len(raw_data)}")

    # open/create hdf5 file at given path, hdf5_fp
    hdf5_obj = h5py.File(hdf5_fp, "a")

    store_raw_cont(raw_data, hdf5_obj)
    store_spikes(raw_data, hdf5_obj)
    store_episodic_data(raw_data, hdf5_obj)


# TODO: implement storing events and metadata
def store_events_classes():
    pass


def store_meta():
    pass

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
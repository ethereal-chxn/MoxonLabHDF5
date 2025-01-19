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
    print("== STORING RAW CONTINUOUS DATA ==")
    block_idx = 0
    segment_idx = 0
    signal_idx = 0

    neural_group = get_subgroup(hdf5_obj, "Neural")

    for block in raw_data:
        segment_idx = 0
        print(f"num of segments in block {block_idx}: {len(block.segments)}")
        for segment in block.segments:
            signal_idx = 0
            print(f"num of AnalogSignals in block {block_idx}, segment {segment_idx}: {len(segment.analogsignals)}")
            for signal in segment.analogsignals:
                print(f"shape of AnalogSignal {signal_idx} in block {block_idx}, segment {segment_idx}: {signal.shape}")
                signal_np_arr = signal.magnitude
                neural_group[f"RawContinuous{signal_idx}"] = signal_np_arr
                signal_idx += 1
            segment_idx += 1
        block_idx += 1    

# TODO: implement storing spike data into HDF5 file
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
            for signal in segment.spiketrains:
                print(f"shape of SpikeTrains {spiketrain_idx} in block {block_idx}, segment {segment_idx}: {signal.shape}")
                signal_np_arr = signal.magnitude
                neural_group[f"Spikes{spiketrain_idx}"] = signal_np_arr
                spiketrain_idx += 1
            segment_idx += 1
        block_idx += 1


# ====================
# || MOX_H5 METHODS ||
# ====================

# Stores a raw EEG file's data into the Neural group.
# ===
# Command format: store_neural [data_fp] [hdf5_fp] [subgroup]

def store_neural(data_fp, hdf5_fp):
    # extract data_fp and hdf5_fp from user input
    # data_fp = params[0]
    # hdf5_fp = params[1]

    # read in raw EEG file as a NEO object
    reader = neo.io.get_io(data_fp)
    raw_data = reader.read(lazy=False)

    # FOR DEBUGGING PURPOSES
    print(f"num of blocks in raw data: {len(raw_data)}")

    # open/create hdf5 file at given path, hdf5_fp
    hdf5_obj = h5py.File(hdf5_fp, "a")

    store_raw_cont(raw_data, hdf5_obj)
    store_spikes(raw_data, hdf5_obj)


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
    
    store_neural(data_path, hdf5_file_name)


# TODO: refactor program to create HDF5 files for each recording session
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
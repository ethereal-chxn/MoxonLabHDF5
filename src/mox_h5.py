import h5py
import numpy as np
import neo
import sys

# ======================
# || HELPER FUNCTIONS ||
# ======================

# Generate a dataset name from a given raw data file's path regardless of if given
# absolute or relative path.
def gen_dataset_name_from_filepath(data_fp):
    data_fp_dirs = data_fp.split('/')
    file_name = data_fp_dirs[len(data_fp_dirs) - 1]
    return file_name.split('.')[0]


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


# Stores analog signals within the raw EEG data file's NEO object form into
# a specified subgroup in the specified HDF5 file object.
# ===
# Note: Consult the Block, Segment, and AnalogSignal sections in 
# https://neo.readthedocs.io/en/latest/api_reference.html# to understand
# how EEG data is structured in an NEO object.
def store_analog_signals(raw_data, hdf5_obj, subgroup_path, dataset_name):
    block_idx = 0
    segment_idx = 0
    signal_idx = 0

    for block in raw_data:
        segment_idx = 0
        print(f"num of segments in block {block_idx}: {len(block.segments)}")
        for segment in block.segments:
            signal_idx = 0
            print(f"num of AnalogSignals in block {block_idx}, segment {segment_idx}: {len(segment.analogsignals)}")
            curr_segment_signals_path = f"{subgroup_path}/{dataset_name}/Block{block_idx}/Segment{segment_idx}/AnalogSignals"
            curr_segment_signals = get_subgroup(hdf5_obj, curr_segment_signals_path)
            for signal in segment.analogsignals:
                print(f"shape of AnalogSignal {signal_idx} in block {block_idx}, segment {segment_idx}: {signal.shape}")
                signal_np_arr = signal.magnitude
                curr_segment_signals[f"AnalogSignal{signal_idx}"] = signal_np_arr
                signal_idx += 1
            segment_idx += 1
        block_idx += 1    


# ====================
# || MOX_H5 METHODS ||
# ====================

# Stores a raw EEG data file into the HDF5 file's Neural group.
# ===
# Command format: store_neural [data_fp] [hdf5_fp] [subgroup]

# TODO: get store_neural to work with the DRP003 txt files
def store_neural(params):
    # extract data_fp, hdf5_fp, and subgroup from user input
    data_fp = params[0]
    hdf5_fp = params[1]
    subgroup = params[2]

    # read in raw EEG file as a NEO object
    reader = neo.io.get_io(data_fp)
    raw_data = reader.read(lazy=False)
    print(f"num of blocks in raw data: {len(raw_data)}")

    # open/create hdf5 file at given path, hdf5_fp
    hdf5_obj = h5py.File(hdf5_fp, "a")

    subgroup_path = "Neural/" + subgroup
    dataset_name = gen_dataset_name_from_filepath(data_fp)

    store_analog_signals(raw_data, hdf5_obj, subgroup_path, dataset_name)


# Deletes a specified dataset from a specified HDF5 file's Neural group.
# ===
# Command format: delete_neural_ds [hdf5_obj] [dataset_path]
def delete_neural_dataset(params):
    hdf5_obj = params[0]
    dataset_path = params[1]

    # # Attempt to access dataset
    # try:
    #     specified_dataset = get_dataset(hdf5_obj, dataset_path)
    # except:
    #     print("Error in deleting neural dataset: Specified dataset does not exist.")
    #     return
    try:
        del hdf5_obj[dataset_path]
    except:
        print("Error occurred in deleting neural dataset")


# To run this program:
# py ./src/mox_h5.py [command] [params]
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: py ./src/mox_h5.py [command] [params]")
        exit(1)
    
    cmd_name = sys.argv[1]
    params = sys.argv[2:]

    if cmd_name == "store_neural":
        if len(params) != 3:
            print("Usage: py ./src/mox_h5.py store_neural [data_fp] [hdf5_fp] [subgroup]")
            exit(1)
        store_neural(params)  
    else:
        print(f"Error: command {cmd_name} not found")
        exit(1)
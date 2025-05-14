import h5py
import numpy as np
import neo
import sys
import modules.datatype_creator as datatype_creator
import modules.utils as utils
import json

def get_events_from_file(events_fp):
    with open(events_fp, 'r') as events_file:
        events_json = events_file.read()
    events_json = json.loads(events_json)
    events = events_json["events"]

    return events

def store_events(events_fp, hdf5_obj):
    print("== STORING EVENTS ==")

    events = get_events_from_file(events_fp)

    behavioral_group = utils.get_subgroup(hdf5_obj, "Behavioral")
    events_ds = behavioral_group.create_dataset("Events", len(events), datatype_creator.events_dt())

    for event in events:
        event_idx = event["id"]
        event_time = event["time_m"]
        event_type = event["name"]
        event_value = "N/A"

        events_ds[event_idx] = (event_time, event_type, event_value)
    
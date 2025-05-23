# Read the LSL data from an XDF file and the capnograph data from CSV files

import os
import glob
import pyxdf
import numpy as np
import pandas as pd
import xarray as xr
import cedalion
import mne 

from cedalion.dataclasses import PointType

def read_stream(stream):
    """
    Read a stream from an XDF file and return the data, timestamps, and metadata.
    """

    # Extract the data and timestamps from the stream
    data = stream['time_series']
    timestamps = stream['time_stamps']
    metadata = stream['info']['desc'][0]
    
    return data, timestamps, metadata

def read_od(fnirs_data):

    fnirs_data = fnirs_data[:, 1:]  # Omit first "channel" that has not data
    fnirs_n_channels = (fnirs_data.shape[1])//4
    # Read OD data only (omit concentration changes data)
    fnirs_data_raw_wl1 = fnirs_data[:, :fnirs_n_channels]
    fnirs_data_raw_wl2 = fnirs_data[:, fnirs_n_channels:2*fnirs_n_channels]
    fnirs_data_od = np.concatenate([fnirs_data_raw_wl1.reshape(-1, 1, fnirs_n_channels), 
                                    fnirs_data_raw_wl2.reshape(-1, 1, fnirs_n_channels)], axis=1)
    
    return fnirs_data_od.T

def read_eeg_metadata(eeg_metadata, montage_name='easycap-M1'):

    # Read channel names
    eeg_channels_all = eeg_metadata['channels'][0]['channel']

    # Read EEG montage information from LSL metadata
    montage_mne = mne.channels.make_standard_montage(montage_name)
    
    channels_row = []
    for item in eeg_channels_all:
        label = item['label'][0]
        if 'EOG' in label:
            pos = None
            type = 'eog'
        elif 'EMG' in label:
            pos = None
            type = 'emg'
        else:
            try:
                pos = montage_mne.get_positions()['ch_pos'][label] * 1000
                type = 'eeg'
            except KeyError:
                # If the label is not found in the montage, we assume it's an auxiliary channel
                pos = None
                type = 'aux'

        channels_row.append([label, pos, type])

    # Add standard landmarks
    # channels_row.append(['Nz', montage_mne.get_positions()['nasion'] * 1000, PointType.LANDMARK])
    # channels_row.append(['Iz', montage_mne.get_positions()['ch_pos']['Iz'] * 1000, PointType.LANDMARK])
    # channels_row.append(['LPA', montage_mne.get_positions()['lpa'] * 1000, PointType.LANDMARK])
    # channels_row.append(['RPA', montage_mne.get_positions()['rpa'] * 1000, PointType.LANDMARK])

    liveamp_channels = pd.DataFrame(channels_row, columns=['label', 'pos', 'type'])

    return liveamp_channels

def read_fnirs_montage(fnirs_metadata, landmarks_pos_dir=None):
    """
    Read fnirs montage from LSL metadata.
    """

    # Read source and detector positions
    meta_montage = fnirs_metadata['montage'][0]
    sources = meta_montage['optodes'][0]['sources'][0]['source']
    detectors = meta_montage['optodes'][0]['detectors'][0]['detector']
    sources_pos = np.array([np.array(list(o['location'][0].values()), dtype=float).squeeze()[:3] for o in sources])
    detectors_pos = np.array([np.array(list(o['location'][0].values()), dtype=float).squeeze()[:3] for o in detectors])

    # Build pandas DataFrame
    optodes_rows = []
    for i, pos in enumerate(sources_pos):
        label = f"S{i + 1}"
        type = PointType.SOURCE
        pos = pos.tolist()
        optodes_rows.append([label, pos, type])

    for i, pos in enumerate(detectors_pos):
        label = f"D{i + 1}"
        type = PointType.DETECTOR
        pos = pos.tolist()
        optodes_rows.append([label, pos, type])

    # Add standard landmarks
    # These values are taken from the first entries of the digpts.txt file that Aurora generates
    # If such a file is not provided, we use some hardcoded values that came from one of these files,
    # and they should work well in all cases using the ICBM152 heamodel.
    if landmarks_pos_dir is not None:
        landmarks = np.loadtxt(landmarks_pos_dir, dtype=str)
        for lk in landmarks:
            if lk[0] == 'nz:':
                label = 'Nz'
            elif lk[0] == 'ar:':
                label = 'RPA'
            elif lk[0] == 'al:':
                label = 'LPA'
            elif lk[0] == 'iz:':
                label = 'Iz'
            else:
                continue
            
            pos = np.array([float(lk[1]), float(lk[2]), float(lk[3])])
            type = PointType.LANDMARK
            optodes_rows.append([label, pos, type])
    else:
        # Hardcoded values (probably the same for all configuration files with  ICBM15 headmodel)
        optodes_rows.append(['Nz', [0.400, 85.900, -47.600], PointType.LANDMARK])
        optodes_rows.append(['Iz', [0.200, -120.500, -25.800], PointType.LANDMARK])
        optodes_rows.append(['LPA', [-83.800, -18.600, -57.200], PointType.LANDMARK])
        optodes_rows.append(['RPA', [83.900, -16.600, -56.700], PointType.LANDMARK])


    fnirs_montage_pd = pd.DataFrame(optodes_rows, columns=['label', 'pos', 'type'])

    return fnirs_montage_pd

def read_fnirs_metadata(fnirs_metadata, landmarks_pos_dir=None):

    # Restrict to OD data only
    fnirs_meta_all = fnirs_metadata['channels'][0]['channel'][1:]
    n_channels = len(fnirs_meta_all)//4
    fnirs_meta_od = fnirs_meta_all[:n_channels]

    # Read channels
    fnirs_channels = [c['custom_name'][0].replace('-', '') for c in fnirs_meta_od]

    # Read wavelengths
    wl1 = float(fnirs_meta_all[0]['wavelength'][0])
    wl2 = float(fnirs_meta_all[n_channels]['wavelength'][0])

    # Read montage
    fnirs_montage_pd = read_fnirs_montage(fnirs_metadata, landmarks_pos_dir)
    # Convert to xarray
    fnirs_montage = xr.DataArray(data=list(fnirs_montage_pd.pos),
                            dims=['label', 'pos'],
                            coords={'label': list(fnirs_montage_pd.label),
                                    'type': ('label', fnirs_montage_pd.type)})
    # Add units and invert normals for later alignement with head
    fnirs_montage.data = fnirs_montage.data * cedalion.units.mm

    return fnirs_channels, wl1, wl2, fnirs_montage

def build_fnirs_xr(fnirs_data, fnirs_time, fnirs_channels, wl1, wl2):
    """
    Build xDataArray object in Cedalion-compatible format.
    """

    # Build xDataArray object in Cedalion-compatible format
    fnirs_data_xr = xr.DataArray(data=fnirs_data, 
                                dims=['channel', 'wavelength', 'time'],
                                coords={'time': fnirs_time,
                                        'samples': ('time', np.arange(0, len(fnirs_time)).tolist()),
                                        'wavelength': [wl1, wl2],
                                        'channel': fnirs_channels,
                                        'source': ('channel', [c.split('D')[0] for c in fnirs_channels]),
                                        'detector': ('channel', ['D' + c.split('D')[1] for c in fnirs_channels])}
                                )
    
    # Add units
    fnirs_data_xr.data = fnirs_data_xr.data * cedalion.units.V
    time_with_units = fnirs_data_xr.time.assign_attrs({'units': cedalion.units.s})
    fnirs_data_xr = fnirs_data_xr.assign_coords({'time': time_with_units})

    return fnirs_data_xr

def build_eeg_mne(liveamp_data, electrodes_time, liveamp_channels, eeg_montage_name='easycap-M1'):
    """
    Build xDataArray object in Cedalion-compatible format.
    """

    # Exclude auxiliary signals
    electrodes_channels = liveamp_channels[liveamp_channels.type!='aux']
    electrodes_names = electrodes_channels.label.tolist()
    electrodes_types = electrodes_channels.type.tolist()
    electrodes_data = liveamp_data[liveamp_channels.type!='aux']

    # Get sampling rate using the median difference between consecutive timestamps for robustness
    eeg_dt = np.median(np.diff(electrodes_time))
    eeg_rate = 1 / eeg_dt
    # Compute the first_samp such that raw.first_time == ti
    ti = electrodes_time[0]
    first_samp = int(round(ti * eeg_rate))

    # Create the MNE info dictionary
    eeg_info = mne.create_info(ch_names=electrodes_names, sfreq=eeg_rate, ch_types=electrodes_types)

    # Create the Raw object (MNE expects data of shape (n_channels, n_samples))
    eeg_mne = mne.io.RawArray(data=electrodes_data, 
                              info=eeg_info, 
                              first_samp=first_samp)
    
    # Add montage, a subset of the standard 10-20 system
    eeg_montage = mne.channels.make_standard_montage(eeg_montage_name)
    eeg_mne.set_montage(eeg_montage)

    return eeg_mne

def build_liveamp_aux(liveamp_data, liveamp_time, liveamp_channels):

    liveamp_aux_data = liveamp_data[liveamp_channels.type=='aux']
    liveamp_aux_channels = liveamp_channels[liveamp_channels.type=='aux'].label.tolist()

    # Build xDataArray object in Cedalion-compatible format
    liveamp_aux_xr = xr.DataArray(data=liveamp_aux_data, 
                                dims=['channel', 'time'],
                                coords={'time': liveamp_time,
                                        'samples': ('time', np.arange(0, len(liveamp_time)).tolist()),
                                        'channel': liveamp_aux_channels})
    
    return liveamp_aux_xr

def rename_markers(markers_data):
    # Rename markers (avoid this if possible by giving them the right names in the first place)
    markers_labels = []
    finger_list = ['thumb', 'index', 'middle', 'ring', 'pinkie', 'all']
    for m in markers_data:
        label = m.split('_')
        if label[0] in ['sitting', 'walking']:
            condition = label[0]
            finger = [f for f in finger_list if f in label[1]][0]
            if 'task' in label[1]:
                on_off = 'on'
            elif 'rest' in label[1]:
                on_off = 'off'
            label = '_'.join([condition, finger, on_off])
        elif label[0] in ['pre', 'post', 'after']:    
            label = '_'.join([label[0], label[1]])
        else:
            label = m
        markers_labels.append(label)

    markers_labels = np.array(markers_labels)

    return markers_labels

def markers_to_pandas(markers_data, markers_ts):
    
    # Convert to numpy arrays and remove extra dimension
    markers_data = np.array(markers_data).squeeze()
    # markers_data = rename_markers(markers_data) # Used in some cases with bad LSL naming
    # Convert to Pandas
    markers_pd = pd.DataFrame({'marker': markers_data, 'time': markers_ts})

    return markers_pd

def buttons_to_pandas(buttons_data, buttons_ts):
    
    # Convert to numpy arrays and remove extra dimension
    buttons_data = np.array(buttons_data).squeeze()
    
    # Assign values to buttons
    finger_map = {'Th_P': 1, 'In_P': 2, 'Md_P': 3, 'Lt_P': 4,
              'Th_R': .5, 'In_R': 1.5, 'Md_R': 2.5, 'Lt_R': 3.5}
    buttons_values = np.array([finger_map[b] for b in buttons_data])
    
    # Convert to Pandas
    buttons_pd = pd.DataFrame({'button': buttons_data, 
                               'values': buttons_values, 
                               'time': buttons_ts})

    return buttons_pd

def load_lifesense_csvs(folder):
    """
    Load LifeSense II capnograph CSV exports into xarray objects with Unix time.
    
    Args:
    folder : str
        Path to directory containing files matching *_gd.csv, *_cw.csv, *_pt.csv.
    
    Returns:
        - gd_ds (xarray.Dataset) : 1 Hz trend data (SpO2, PR, EtCO2, RR, …)
        - cw_xr (xarray.DataArray) : 10 Hz capnogram waveform (CO2)
        - pt_xr (xarray.DataArray) : event-driven pulse intervals (Pulse_Time)
    
    Each has a coord `"time"` giving the Unix timestamp in seconds (UTC).
    """
    
    # find the three files
    gd_file = glob.glob(os.path.join(folder, '*_gd.csv'))[0]
    cw_file = glob.glob(os.path.join(folder, '*_cw.csv'))[0]
    pt_file = glob.glob(os.path.join(folder, '*_pt.csv'))[0]
    
    # helper to parse Date+Time → UTC datetime → Unix seconds
    def build_unix_time(df, time_format):
        # Read time in datetime format
        time = pd.to_datetime(df['Date'] + ' ' + df['Time'], format=time_format)
        # Localize and convert to UTC
        time = time.dt.tz_localize('Europe/Berlin').dt.tz_convert('UTC')
        # Build unix time
        time = np.array([t.timestamp() for t in time])
        
        return time
    
    # Load gd file
    gd = pd.read_csv(gd_file)
    # Build unix time
    gd_unix = build_unix_time(gd, time_format="%Y-%m-%d %H:%M:%S")
    # Select main signals
    gd_vars = ['SpO2', 'PR', 'EtCO2(mmHg)', 'RR']
    gd_ds = xr.Dataset(
        {var: ('time', gd[var].astype(float).values) for var in gd_vars},
        coords={'time': gd_unix.astype(float)}
    )

    # Load cw
    cw = pd.read_csv(cw_file)    
    # Build unix time
    cw_unix = build_unix_time(cw, time_format="%Y-%m-%d %H:%M:%S.%f")
    # Build DataArray
    cw_xr = xr.DataArray(data=cw['CO2(mmHg)'].astype(float).values,
                         coords={'time': cw_unix.astype(float)})

    # Load pt
    pt = pd.read_csv(pt_file)
    # Build unix time
    pt_unix = build_unix_time(pt, time_format="%Y-%m-%d %H:%M:%S.%f")
    # Build DataArray
    pt_xr = xr.DataArray(data=pt['Pulse_Time'].astype(float).values,
                        coords={'time': pt_unix.astype(float)})

    return gd_ds, cw_xr, pt_xr


def load_data(lsl_dir, capnograph_dir, fnirs_landmarks_pos_dir=None):
    """
    Read LSL and Capnograph files and return the synchronized data and metadata.
    """

    # Load the LSL file
    streams, header = pyxdf.load_xdf(lsl_dir)

    # Extract the data, timestamps, and metadata from each stream
    for stream in streams:
        name = stream['info']['name'][0]
        if name == 'EspArgos_ButtonPress':
            buttons_data, buttons_ts, buttons_metadata = read_stream(stream)
        if name == 'Aurora':
            fnirs_data, fnirs_ts, fnirs_metadata = read_stream(stream)
        if name == 'LiveAmpSN-101410-1017':
            liveamp_data, liveamp_ts, liveamp_metadata = read_stream(stream)
        if name == 'PsychoPyMarker':
            markers_data, markers_ts, markers_metadata = read_stream(stream)
        if name == 'UnixTime_s':
            unix_time_data, unix_time_ts, unix_time_metadata = read_stream(stream)
        if name =='Aurora_accelerometer':
            aurora_acc_data, aurora_acc_ts, aurora_acc_metadata = read_stream(stream)

    # Load de Canograph data
    gd_ds, cw_xr, pt_xr = load_lifesense_csvs(capnograph_dir)

    # Bring all data to the same time base
    t0 = np.min([d[0] for d in [fnirs_ts, liveamp_ts, markers_ts, buttons_ts, unix_time_ts, aurora_acc_ts]])
    fnirs_ts -= t0
    liveamp_ts -= t0
    markers_ts -= t0
    buttons_ts -= t0
    unix_time_ts -= t0
    aurora_acc_ts -= t0
    # TODO: Do the same for the capnograph data (needs previous mapped to lsl timestamp)

    # Read only the OD data
    fnirs_data = read_od(fnirs_data)
    # Read fnirs metadata to extract: channels, wavelengths, and montage
    fnirs_channels, wl1, wl2, fnirs_geo3d = read_fnirs_metadata(fnirs_metadata, fnirs_landmarks_pos_dir)
    # Convert to xr
    fnirs_xr = build_fnirs_xr(fnirs_data, fnirs_ts, fnirs_channels, wl1, wl2)

    # Read EEG metadata to extract: channels, and montage
    liveamp_channels = read_eeg_metadata(liveamp_metadata)
    # Read electrode data and convert to mne
    electrodes_mne = build_eeg_mne(liveamp_data.T, liveamp_ts, liveamp_channels)
    # Read liveamp auxiliary data
    liveamp_aux_xr = build_liveamp_aux(liveamp_data.T, liveamp_ts, liveamp_channels)

    # Read Aurora Accelerometer LSL data
    aurora_acc_channels = read_eeg_metadata(aurora_acc_metadata)
    aurora_acc_xr = build_liveamp_aux(aurora_acc_data.T, aurora_acc_ts, aurora_acc_channels)


    # Convert markers to pandas
    markers_data = markers_to_pandas(markers_data, markers_ts)

    # Convert buttons to pandas
    buttons_data = buttons_to_pandas(buttons_data, buttons_ts)

    return fnirs_xr, fnirs_geo3d, electrodes_mne, liveamp_aux_xr, gd_ds, cw_xr, pt_xr, markers_data, buttons_data, aurora_acc_xr










import numpy as np
import xarray as xr

import cedalion
import cedalion.sigproc.quality as quality

def prune_fnirs_channels(amp, 
                         snr_thresh=5, 
                         amp_thresh_min=0.001, 
                         amp_thresh_max=0.8,
                         sci_thresh=0.75,
                         psp_thresh=0.1,
                         sci_psp_percentage_thresh=0.75,
                         window_length=5):
    """
    Prune fnirs channels according to some quality assessment measures and predefined thresholds.
    """

    masks = []
    # Calculate masks for each quality metric if parameters were given
    if snr_thresh is not None:
        _, snr_mask = quality.snr(amp, snr_thresh)
        masks.append(snr_mask)
    
    if amp_thresh_min is not None and amp_thresh_max is not None:
        amp_threshs = [amp_thresh_min * cedalion.units.V, 
                    amp_thresh_max * cedalion.units.V]
        _, amp_mask = quality.mean_amp(amp, amp_threshs)
        masks.append(amp_mask)
    
    if sci_thresh is not None and psp_thresh is not None and sci_psp_percentage_thresh is not None:
        window_length = window_length * cedalion.units.s
        sci, sci_mask = quality.sci(amp, window_length, sci_thresh)
        psp, psp_mask = quality.psp(amp, window_length, psp_thresh)
        sci_x_psp_mask = sci_mask & psp_mask
        perc_time_clean = sci_x_psp_mask.sum(dim="time") / len(sci.time)
        sci_psp_mask = perc_time_clean >= sci_psp_percentage_thresh
        masks.append(sci_psp_mask)
    else:
        perc_time_clean = None
        sci_x_psp_mask = None       

    # Prune channels that do not pass at least one of the quality test
    amp_pruned, drop_list = quality.prune_ch(amp, masks, "all")

    # Display pruned channels and resulting clean signal
    print(f"List of pruned channels: {drop_list}  ({len(drop_list)})")

    return amp_pruned, drop_list, perc_time_clean, sci_x_psp_mask
import os
import cedalion
import cedalion.datasets
import cedalion.imagereco.forward_model as fw

def build_headmodel():

    # load pathes to segmentation data for the icbm-152 atlas
    seg_datadir, mask_files, landmarks_file = cedalion.datasets.get_colin27_segmentation()

    # create forward model class for icbm152 atlas
    head = fw.TwoSurfaceHeadModel.from_surfaces(
        segmentation_dir=seg_datadir,
        mask_files=mask_files,
        brain_surface_file=os.path.join(seg_datadir, "mask_brain.obj"),
        landmarks_ras_file=landmarks_file,
        brain_face_count=None,
        scalp_face_count=None,
    )

    # Set correct units
    head.brain.units = cedalion.units.mm
    head.scalp.units = cedalion.units.mm
    head.landmarks = head.landmarks.pint.dequantify()
    head.landmarks.pint.units = cedalion.units.mm

    return head



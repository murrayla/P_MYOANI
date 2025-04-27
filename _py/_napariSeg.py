"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _napariSeg.py
       fix segmentation dataset to form a single cell.
"""

# ∆ Raw
import napari
import numpy as np

# ∆ Main
if __name__ == "__main__":

    # ∆ Load raw and segmentation data
    # raw_data = np.load("/Users/murrayla/Documents/main_PhD/BIG_SEG/big_xyz_100_3700_100_3700_40_339.npy").astype(np.uint16)
    # seg_data = np.load("/Users/murrayla/Documents/main_PhD/BIG_SEG/filtered.npy").astype(np.uint16)
    seg_data = np.load("_npy/red_seg.npy").astype(np.uint16)

    # ∆ Transform orientations to view
    affine = np.array([[-1, 0, 0], [0, 5, 0], [0, 0, 1],])

    # ∆ Create view
    viewer = napari.Viewer(title='Annotator')
    # raw_layer = viewer.add_image(raw_data, name = "M: Raw Disc Data", affine=affine)
    seg_layer = viewer.add_labels(seg_data, name = "L: Disc Annotations", affine=affine)

    # ∆ Binding
    # µ Down 
    @viewer.bind_key('w')
    def update_func(_):
        vz_plane = viewer.dims.order[0]
        current_vz = viewer.dims.current_step[vz_plane]
        viewer.dims.set_current_step(vz_plane, current_vz-1)
    # µ Up 
    @viewer.bind_key('e')
    def update_func2(_):
        vz_plane = viewer.dims.order[0]
        current_vz = viewer.dims.current_step[vz_plane]
        viewer.dims.set_current_step(vz_plane, current_vz+1)
    # µ Save 
    @viewer.bind_key('s')
    def save_func(_):
        with open("SC_.npy", 'wb') as outfile:
            np.save(outfile, seg_layer.data, allow_pickle=False)
        print('saved')

    # += Run
    viewer.dims.order = (2,1,0)
    napari.run()    
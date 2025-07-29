"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _toVTK.py
        Returns .vtk from numpy conversion
"""

# ∆ Raw
import os
import vtk
import numpy as np
import pandas as pd
from scipy import ndimage
from vtk.util import numpy_support

# ∆ Constants
PIXX, PIXY, PIXZ = 11, 11, 50

# ∆ Save points in VTK format
def save_vtk(mu_idx, vals, file):
    
    # ∆ Store points 
    pnts = vtk.vtkPoints()
    for coord in mu_idx:
        pnts.InsertNextPoint(coord)

    # ∆ Create vertices
    vert = vtk.vtkCellArray()
    for i in range(len(mu_idx)):
        vert.InsertNextCell(1)
        vert.InsertCellPoint(i)

    # ∆ Convert data
    vtk_vals = numpy_support.numpy_to_vtk(vals, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_vals.SetName("Values")

    # ∆ Create polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(pnts)
    polydata.SetVerts(vert)
    polydata.GetPointData().AddArray(vtk_vals)

    # ∆ Write to file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(os.path.abspath(file))
    writer.SetInputData(polydata)
    writer.Write()

# ∆ Main
def main():

    # ∆ Load region data
    reg_df = pd.read_csv("_csv/reg_.csv")

    # ∆ Load segmentation data
    seg_np = np.load("_npy/filtered.npy").astype(np.uint16)

    # ∆ Big region
    Mx, mx = reg_df["x"].max() + 1000, reg_df["x"].min()
    My, my = reg_df["y"].max() + 1000, reg_df["y"].min()
    Mz, mz = reg_df["z"].max() + 100, reg_df["z"].min()

    # ∆ Iterate stacks
    stack = []
    for j in range(mz, Mz + 1, 1):

        print(j, mz, Mz)

        # ∆ Break if outside range
        if j >= seg_np.shape[2]:
            break  

        # ∆ Slice
        slice = seg_np[:, :, j]

        # ∆ Rotate
        rotated = ndimage.rotate(slice, -45, reshape=True, order=1)
        cropped = rotated[mx:Mx+1, my:My+1]
        stack.append(cropped)

    # ∆ Cube and save
    data = np.stack(stack, axis=2)
    np.save(f"_npy/seg_big.npy", data)

    # ∆ Extract coordinate data
    idx = np.where(data > 0)
    coords = np.stack(idx, axis=1)

    # ∆ Scale data 
    mu_idx = coords * np.array([PIXX, PIXY, PIXZ])
    mu_idx[:, [0, 1]] = mu_idx[:, [1, 0]]

    # ∆ Value data for storage
    vals = np.ones(mu_idx.shape[0], dtype=np.uint8)

    # ∆ Save points 
    save_vtk(mu_idx, vals, f"_bp/big.vtp")

# ∆ Initialise
if __name__ == "__main__":
    
    # ∆ Main data
    main()
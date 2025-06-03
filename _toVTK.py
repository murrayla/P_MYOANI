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
def main(file, name):

    # ∆ Load data
    data = np.load(file).astype(np.int16)
    data[data > 0] = 1

    # ∆ Extract coordinate data 
    idx = np.where(data == 1)
    coords = np.stack(idx, axis=1)

    # ∆ Scale data 
    mu_idx = coords * np.array([PIXX, PIXY, PIXZ])
    mu_idx[:, [0, 1]] = mu_idx[:, [1, 0]]

    # ∆ Value data for storage
    vals = np.ones(mu_idx.shape[0], dtype=np.uint8)

    # ∆ Save points 
    save_vtk(mu_idx, vals, f"_bp/{name}.vtp")

# ∆ Initialise
if __name__ == "__main__":
    
    # ∆ Main data
    # name = "seg_0"
    names = [f"seg_{x}" for x in range(0, 18, 1)]
    for name in names:
        file = f"_npy/{name}.npy"
        main(file, name)
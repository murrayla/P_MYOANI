The folder compiles all the relevant code for:

To run simulation

1) have conda, miniconda, or Anaconda installed.
2) run "conda env create --name envname --file=environment.yaml" with .yml file provided
3) run "conda activate envname"
4) open _Sim.py
    4.1) scan to bottom of script to enter "tests = ["test"]" and "r = 800" as required for
            test case and refinement level required. 
    4.2) "test" will provide the unaxial control, "0" or "4" regions have been provided for
            simulation as well
5) visualisations will be saved in the provided "_bp/" folder for open in Paraview
6) for plotting and other data please see relevant script below.

Content:

    File Name: 
        environment.yml
    Description:
        conda environment dependencies required for running of code

    File Name: 
        _angHist.py
    Description:
        creates display data for orientation values and other key visualisations from article.

    File Name: 
        _matOpti.py
    Description:
        material parameter optimisation script used for determining Guccione constants 
        significant overlap with _Sim.py

    File Name: 
        _plotStress.py
    Description:
        functions requires for plotting stress output and simulation values

    File Name: 
        _rawMorphologicalStats.py
    Description: 
        calculates and outputs the key morpholigcal data requires for simulations

    File Name: 
        _refTest.py
    Description:
        mesh refinement testing 

    File Name: 
        _Sim.py
    Description:
        main simulation script running FEniCSx compiled into single runnable .py program

    File Name: 
        _tileMesh.py
    Description:
        simplified mesh generation using Gmsh's api

    Folder Name: 
        _msh
    Description:
        .msh, .ele, .nodes storage

            File Name: 
                em_*.ele
            Description:
                element file for mesh data at refinement level "*"

            File Name: 
                em_*.nodes
            Description:
                nodes file for mesh data at refinement level "*"

            File Name: 
                em_*.msh
            Description:
                readable mesh file for GMSH at refinement level "*"

    Folder Name: 
        _bp
    Description:
        .bp paraview file storage

    Folder Name: 
        _csv
    Description:
        .csv storage

            File Name: 
                tile_*_w.csv
            Description:
                tile specific data for simulation, including centroids and orientation values

            File Name: 
                row_norm_whole.csv
            Description:
                all data for cell region requires for simulation

    Folder Name: 
        _npy
    Description:
        .npy storage

            File Name: 
                seg_*.npy
            Description:
                numpy files for specific segmentation data

    Folder Name: 
        _vtp
    Description:
        .vtp storage

            File Name: 
                seg_*.vtp
            Description:
                vtp files for specific segmentation data and visualisation in Paraview





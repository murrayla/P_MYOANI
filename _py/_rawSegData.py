"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _rawseg.py
        Handle raw segmentation data and output 
"""

# ∆ Raw
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# ∆ Constants
RAW_SEG = "filtered.npy"

# ∆ Main function
def main(labs, data):

    # ∆ Create shell
    red_seg = np.zeros_like(data)

    # ∆ Assign new labels
    for l in labs:
        red_seg[data == l] = l
    
    # ∆ Save data
    with open("_npy/red_seg.npy", 'wb') as outfile:
        np.save(outfile, red_seg, allow_pickle=False)


# ∆ Inititate
if __name__ == "__main__":

    # ∆ Load data
    labs = []
    with open('_txt/labels.txt') as f:
        for line in f:
            labs.append(int(line))

    print(len(labs))
    path = os.path.join(os.path.dirname(__file__), RAW_SEG)
    seg_np = np.load(path).astype(np.uint16)

    main(labs, seg_np)



    
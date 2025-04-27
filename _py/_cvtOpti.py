import numpy as np
import pandas as pd

# Read the two text files
opti_c = np.loadtxt('_txt/opti_c.txt')
opti_s = np.loadtxt('_txt/opti_s.txt')

# Check their shapes to make sure they match row-wise
if opti_c.shape[0] != opti_s.shape[0]:
    raise ValueError("Files have different number of rows!")

# Concatenate them horizontally
opti_combined = np.hstack((opti_c, opti_s))

# Create a DataFrame
df = pd.DataFrame(opti_combined)

# Save to CSV
df.to_csv('_csv/opti_.csv', index=False, header=False)


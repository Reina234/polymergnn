import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"  # Use serif font
datasets = ["A", "B", "C", "D"]
methods = ["Rg mean", "Rg SD", "D mean", "SASA mean", "SASA SD", "Re mean", "Average"]

errors = np.array(
    [
        [11, 26, 71, 10, 36, 40, 33],  # Shared layer
        [7.4, 24, 49, 8.6, 32, 43, 27],  # One less shared layer
        [7.8, 28, 38, 9.3, 30, 43, 26],  # Morgan FP, (-) Shared Layer
        [14, 27, 69, 15, 32, 42, 33],  # fine tuned
    ]
)

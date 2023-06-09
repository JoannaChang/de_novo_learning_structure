# _De novo_ motor learning creates structure in neural activity space that shapes adaptation

This repository includes code to reproduce the simulations and figures in [Chang et. al (bioRxiv, 2023)](https://www.biorxiv.org/content/10.1101/2023.05.23.541925v2).

## Getting Started

1. Create a conda environment with ```conda env create -f env.yml``` and activate the environment with ```conda activate structure```
2. Install the package and dependencies with 
  ```pip install -e .```
  from the root of the repository. 
3. Change the project directory to the root of the repository under ```PROJ_DIR``` in ```constants.py```

## Reproducing figures
Each figure in the paper has an associated Jupyter notebook under ```paper/```. Running the cells reproduces all of the subfigures, and the first cell runs the simulations associated with the figure. Note that the experimental data used for Supplementary Figure 6 is currently not included but will be available upon publication. The code and figures are still provided.

## System Requirements
The code has been tested on Linux (Ubuntu 18.04). Around 45GB is required for the data used for all main and supplementary figures, and an additional 7GB is needed for the results.

## License
[MIT](https://opensource.org/license/mit/)

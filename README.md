# _De novo_ motor learning creates structure in neural activity space that shapes adaptation

This repository includes code to reproduce the simulations and figures in [Chang et. al (Nature Communications, 2024)](https://www.nature.com/articles/s41467-024-48008-7).

## Getting Started

1. Create a conda environment with ```conda env create -f env.yml``` and activate the environment with ```conda activate structure```
2. Install the package and dependencies with 
  ```pip install -e .```
  from the root of the repository. 
3. Change the project directory to the root of the repository under ```PROJ_DIR``` in ```constants.py```

## Reproducing figures
Each figure in the paper has an associated Jupyter notebook under ```paper/```. Running the cells reproduces all of the subfigures, and the first cell runs the simulations associated with the figure. Note that the experimental data used for Supplementary Figure 8 is not explicitly included, but most of the data is [publicly available on Dryad](https://doi.org/10.5061/dryad.xd2547dkt). The remaining datasets will be made available on request. The code and figures are still provided.

## System Requirements
The code has been tested on Linux (Ubuntu 18.04). Around 45GB is required for the data used for all main and supplementary figures, and an additional 7GB is needed for the results.

## License
[MIT](https://opensource.org/license/mit/)

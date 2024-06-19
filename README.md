# CrAM

Welcome to the CrAM repository. This repository provides our code and data.

## Overview

### Results
- All of our results can be found in the following folders:
  - `results`: Contains experimental results without misinformation.
  - `results_real`: Includes results under the GPT setting.
  - `results1`: Contains results under the ideal setting.

### Example Code
- You can find some example code in the `run.sh` script.

### Core Code

- The core code for modifying attention weights is located in the `utils/re_weighting.py` file, specifically in the [`Re_Weighting_Strategy` class](./utils/re_weighting.py#L22).
- The core code for calculating the impact of each head on the final result is in the same file, in the [`Find_Best_Heads(Re_Weighting_Strategy)` class](./utils/re_weighting.py#L50).

### Future Release
- A more complete code repository will be released after the review process is completed. **Please note that our paper is still under review and the code has not been fully organized.**

# Open Source On-Chip Inductor Design
**NOTE: THIS IS A WORK IN PROCESS, NOT INTENDED FOR REAL DESIGN (YET)** 

This repository contains code for an Open Source tool capable of inductor geometry
synthesis. It is heavily based on the work by Hershenson et al. ([doi.org/10.1109/DAC.1999.782241](https://doi.org/10.1109/DAC.1999.782241)) with some modifications
as described in [doi.org/10.1109/DAC.1999.782241](https://doi.org/10.1109/DAC.1999.782241).

The code in this repository is in process of being cleaned up and refactored, so
API changes are expected. Please open an issue if you have any questions.

## Installation
You can install this repository as a python package using the following command:
```bash
pip install -e ./projects/inductor_lab
```
Usage examples can be found in the `projects/inductor_lab/tests` directory. Clearer
and more thorough examples should be avaialable in the future.

## Adding new PDKs
You can find the PDK definitions inside the 
`projects/inductor_lab/src/inductor_lab/pdk` directory. If you need a to use
a different process, simply create a new class following the same structure as in
`sg13g2.py` and `gf180mcu.py`. In the future an XML parser should be developed.
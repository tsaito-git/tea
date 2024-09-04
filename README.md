# Tools for Measuring Turbulent Gas Flux using True Eddy Accumulation (TEA)

This repository contains code for measuring turbulent gas fluxes using the true eddy accumulation (TEA) technique. 
The code accompanies a submitted paper and provides tools for data processing and flux calculation.

## Code Structure

- `TEA_KEW_20220422.CR1X`: The main script for the Campbell Scientific CR1000X Datalogger. This script controls the mass flow controllers.
- `211130_EA_down.CR1`: Containes functions for processing accumulated sample.
- 'tea_mean4.py': Containes tools for calculating fluxes.
- 'tob.py': Provides functions for reading TOB files.

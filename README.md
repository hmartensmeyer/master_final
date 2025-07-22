# Reflection-Based Detection of Surface-Attached Vortices

This repository contains the code used for the master's thesis Recognising sub-surface turbulence via light reflected from a free surface at NTNU.

Vortices are detected in both surface optical reflections and in surface elevation data via profilometry, and compared to sub-surface vortices identified using Stereoscopic Particle Image Velocimetry (SPIV). For processing the optical reflections, a GoPro camera is used to record the ceiling reflection, and the data is processed using Master_final/Video and processing/processing.m. The wavelet analysis is performed in Master_final/Wavelet analysis/createWaveletResults.m. For the profilometry data, the same wavelet-based strategy is applied in Master_final/Wavelet analysis/createWaveletResultsProfilometry.m.

Sub-surface vortices are detected in SPIV data using the λ₂-criterion implemented in Master_final/Lambda2 and divergence/readPIV_multipleFrames_import.m. The surface and sub-surface planes are spatially aligned through a calibration process described in Master_final/Calibration/visualizeBothPlanes.m.

There are some other files in there as well. Master_final/Analysis and create figs is for creating the results presented in the thesis.

The full methodology is described in detail in the master's thesis PDF.

## Requirements

- Matlab (pretty new version I would guess)
- Image processing toolbox
- Wavelet toolbox
- Computer vision toolbox
- Signal processing toolbox
- Statistics and machine learning toolbox
(Might not need all these, but I had these installed when I submitted)

## Questions?

hermanmartensmeyer@gmail.com

# CUDA Face Dataset Augmentation Project

## Overview

This project uses NVIDIA's CUDA and NPP (NVIDIA Performance Primitives) library to generate synthetic training data from the CMU Face Images dataset. The application performs GPU-accelerated image transformations including random rotations (0-360°) and horizontal flips to augment face image datasets for training machine learning models.

**Key Features:**
- Batch processing of all PGM images in the CMU Face dataset
- GPU-accelerated image rotation using NPP's `nppiRotate_8u_C1R`
- GPU-accelerated horizontal flipping using NPP's `nppiMirror_8u_C1R`
- Random transformations to generate diverse training data
- Generates 3 augmented variants per original image
- Cross-platform support (Linux/Windows)

**Use Case:** Data augmentation for face detection and recognition model training, significantly expanding training datasets without requiring additional data collection.

## Code Organization

```bin/```
This folder should hold all binary/executable code that is built automatically or manually. Executable code should have use the .exe extension or programming language-specific extension.

```data/```
This folder should hold all example data in any format. If the original data is rather large or can be brought in via scripts, this can be left blank in the respository, so that it doesn't require major downloads when all that is desired is the code/structure.

```lib/```
Any libraries that are not installed via the Operating System-specific package manager should be placed here, so that it is easier for inclusion/linking.

```src/```
The source code should be placed here in a hierarchical fashion, as appropriate.

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```INSTALL```
This file should hold the human-readable set of instructions for installing the code so that it can be executed. If possible it should be organized around different operating systems, so that it can be done by as many people as possible with different constraints.

```Makefile```
There should be some rudimentary scripts for building your project's code in an automatic fashion.

```run.sh```
An optional script used to run your executable code, either with or without command-line arguments.
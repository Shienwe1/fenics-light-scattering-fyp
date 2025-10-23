# Development of a Finite Element Solver for Light Scattering Simulation

This repository contains the code for a Final Year Project: **"Development of a finite element solver for light scattering simulation."**

This project uses the Finite Element Method (FEM) to solve time-harmonic electromagnetic scattering problems, built entirely on the [FEniCSx](https://fenicsproject.org/) platform (DOLFINx).

## Core Problem
Simulating light scattering involves solving Maxwell's equations in an open, unbounded domain.

A key challenge in open-domain problems is preventing non-physical reflections from the computational boundaries. 
This solver implements **Perfectly Matched Layers (PMLs)** to absorb outgoing scattered waves, simulating an infinite domain.

## Project Goals


## Technology Stack


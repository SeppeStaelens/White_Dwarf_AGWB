# AGWB from Extragalactic BWDs

This code can be used to simulate the astrophysical gravitational wave background sourced by a population of extragalactic white dwarf binaries. 

This code was originally created for research in context of a Master's Thesis [MT1](/references/master_thesis_AnA_Seppe_Staelens.pdf) under supervision of prof. Gijs Nelemans. It has been improved prior to submission of [Staelens, Nelemans 2024](https://www.aanda.org/articles/aa/full_html/2024/03/aa48429-23/aa48429-23.html), and subsequently improved for THESIS and [2407.10642](https://arxiv.org/abs/2407.10642).

## Overview of repository:

- `src`: contains the main scripts.
- `output`: contains Figures and GWBs folders, where the latter stores the GWBs and the contributions to the different bins.
- `post_proc`: contains Jupyter Notebooks to process the data.
- `data`: contains the initial values and the interpolated z_at_value function.
- `doc`: contains documentation.
- `references`: contain the Master's Theses as PDF files.

## Installation

Conda environment, yaml file.

## Running the code

Run code from Src directory.

## Clarification on some of the formulas

References to equations are as given in [MT1](/references/master_thesis_AnA_Seppe_Staelens.pdf), unless otherwise mentioned. The clarifications here are to correct numerical factors in [MT1](/references/master_thesis_AnA_Seppe_Staelens.pdf), that have been adapted for [Staelens, Nelemans 2024](https://www.aanda.org/articles/aa/full_html/2024/03/aa48429-23/aa48429-23.html).

### Expressions for Omega

For the bulk, $\Omega$ is calculated as in (2.45), with a different prefactor 2.0E-15 instead of 5.4E-15, due to the normalisation 4E6 solar masses from the paper (as opposed to 1.5E6 solar masses in [MT1](/references/master_thesis_AnA_Seppe_Staelens.pdf)) for the population synthesis.

Similarly, the constant prefactor in (2.48) is changed to 3.2E-15, instead of 8.5E-15, for the birth and merger contributions.

### Expressions for z contribution

In the bulk part of the code, the $z$ contributions are saved as $\Omega$ / (2E-15 * frequency factor), in order not to work with small numbers. This is the reason the $z$ contributions in the merger and birth part are saved as $\Omega$ / (this normalization) as well, to keep the relative contributions the same (as that is all we are interested in).

### Expressions for number of binaries

This part is not in [MT1](/references/master_thesis_AnA_Seppe_Staelens.pdf). I determined the number of binaries in each $z$-$f$ bin as follows:

$$ N(z, f) = (4 \pi \cdot \chi(z)^2 \cdot \Delta \chi(z)) \cdot n (z, f) , $$

where $n(z, f)$ is the number density of systems in the bin. The latter is given by

$$ n(z, f) = \sum_k \frac{\psi(z; k)}{4\cdot 10^6 M_\odot} \cdot \tau(z, f; k) .$$

In this expression, $\psi$ is again the SFH, determined at the birth time of the system and normalized by 4E6 solar masses; and $\tau$ is the time it takes the system to traverse the bin. The reasoning is that all systems produced in the past, during a time corresponding to $\tau$, will have moved to the bin under consideration.

In the code, $\tau$ is calculated in Myr, and therefore multiplied by 1E6 as $\psi$ has units of 1/yr.

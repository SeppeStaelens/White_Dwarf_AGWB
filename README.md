# White_Dwarf_AGWB
Research in context of my Master's Thesis under supervision of prof. Gijs Nelemans.

Overview of folders:

- Src: contains the main scripts.
- Output: contains Figures and GWBs folders, where the latter stores the GWBs and the contributions to the different bins.
- PostProc: contains Jupyter Notebooks to process the data.
- Data: Contains the initial values and the interpolated z_at_value function.

Run code from Src directory.

## Clarification on some of the formulas

References to equations are as given in my thesis, unless otherwise mentioned. The clarifications here are to correct numerical factors in my thesis, that have been adapted for the paper.

### Expressions for Omega

For the bulk, Omega is calculated as in (2.45), with a different prefactor 2.0E-15 instead of 5.4E-15, due to the normalisation 4E6 solar masses from the paper (as opposed to 1.5E6 solar masses in the thesis) for the population synthesis.

Similarly, the constant prefactor in (2.48) is changed to 3.2E-15, instead of 8.5E-15, for the birth and merger contributions.

### Expressions for z contribution

In the bulk part of the code, the z contributions are saved as Omega / (2E-15 * frequency factor), in order not to work with small numbers. This is the reason the z contributions in the merger and birth part are saved as Omega / (this normalization) as well, to keep the relative contributions the same (as that is all we are interested in).

### Expressions for number of binaries

This part is not in my thesis. I determined the number of binaries in each z-f bin as follows:

$$ N(z, f) = (4 \pi \chi(z)^2 * \Delta \chi(z)) \cdot n (z, f)\,, $$

where $n(z, f)$ is the number density of systems in the bin. The latter is given by

$$ n(z, f) = \sum_k \frac{\psi(z; k)}{4\cdot 10^6 M_\odot} \cdot \tau(z, f; k)\,.$$

In this expression, $\psi$ is again the SFH, determined at the birth time of the system and normalized by 4E6 solar masses; and $\tau$ is the time it takes the system to traverse the bin. The reasoning is that all systems produced in the past, during a time corresponding to $\tau$, will have moved to the bin under consideration.

In the code, $\tau$ is calculated in Myr, and therefore multiplied by 1E6 as $\psi$ has units of yr${}^{-1}$.

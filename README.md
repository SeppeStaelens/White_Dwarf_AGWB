# AGWB from Extragalactic BWDs

This code can be used to simulate the astrophysical gravitational wave background sourced by a population of extragalactic white dwarf binaries.

This code was originally created for research in context of a Master's Thesis [MT1](/references/master_thesis_Seppe_Staelens.pdf) under supervision of prof. Gijs Nelemans. It has been improved prior to submission of [Staelens, Nelemans 2024](https://www.aanda.org/articles/aa/full_html/2024/03/aa48429-23/aa48429-23.html), and subsequently improved for [MT2](/references/master_thesis_Sophie_Hofman.pdf) and [2407.10642](https://arxiv.org/abs/2407.10642).

## Overview of repository:

- `data`: contains the results of the population synthesis, the star formation histories and the z_at_age.txt file.
- `doc`: contains documentation in html and pdf format.
- `output`: contains Figures and GWBs folders, where the latter stores the GWBs and the contributions to the different bins.
- `post_proc`: contains a Jupyter Notebook with some examples of how to process the data.
- `references`: contain the two Master's Theses as PDF files.
- `src`: contains the main scripts.

### data

This folder contains subfolders relating to different BWD populations, as produced by the [SeBa](https://github.com/amusecode/SeBa) code. Data is structured along the different models as described in [2407.10642](https://arxiv.org/abs/2407.10642). The population used in [Staelens, Nelemans 2024](https://www.aanda.org/articles/aa/full_html/2024/03/aa48429-23/aa48429-23.html) is stored in `AlphaAlpha/Alpha4/z02`, alongside a similar population used in the other article.

The folder also contains the different star formation rate histories (in SFRD) used in [2407.10642](https://arxiv.org/abs/2407.10642).

Finally, this folder also contains a file `z_at_age.txt`, which is just a data file relating the age of the Universe to the redshift in a Planck 18 cosmology. It is used in the `RedshiftInterpolator` class to quickly determine $z$ at a given age of the Universe. This is in order to circumvent calling the `astropy.cosmology.z_at_value` function too often, as it is computationally very expensive.

### output

`GWBs/SN24` contains 2 examples, both with SFH 1 - 50 frequency bins - 20 integration bins, but integrated over redshift and cosmic time respectively. Some figures based on post-processing this data is stored in `Figures` as an example. This is the GWB as presented in [Staelens, Nelemans 2024](https://www.aanda.org/articles/aa/full_html/2024/03/aa48429-23/aa48429-23.html).
`GWBs/HN24` contains the main GWB as presented in Figure 8 of [2407.10642](https://arxiv.org/abs/2407.10642).

### src

This folder contains the code. `GWB.py` is the main script to calculate the GWB. It relies on many of the functions defined in the modules subfolder. The latter contains the three main parts of the code, auxiliary functions, physical functions, star formation histories and classes `SimModel`, `SFRInterpolator` and `RedshiftInterpolator`.

`Create_z_at_age.py` is used to create a file `z_at_age.txt` stored in `data`, which is used in the main script to interpolate $z$ at a given age of the Universe. `SeBa_pre_process.py` is used to add more columns to the output data from SeBa, which is then used in the main script.

## Installation

The code can be installed simply by cloning the repository. The repository contains a `WD_GWB.yml` file that can be used to create a `conda` environment that contains all the required packages to run the code in `src` and `post_proc`. This is done by running

```
$ conda env create -f WD_GWB.yml
```

## Running the code

The code can be run from the main directory or `src`.
One should start by activating the `WD_GWB` environment as follows:

```
$ conda activate WD_GWB
```

Afterwards, all three scripts can simply be run by doing

```
$ python Create_z_at_age.py
$ python SeBa_pre_process.py
$ python GWB.py param.ini
```

For `Create_z_at_age.py` one only needs to specify a maximum redshift and the number of interpolation points desired.

For `SeBa_pre_process.py`, one only needs to specify the data paths and whether to save the file. Additional datafolders can be made here, and they should be adapted in the main code.

For `GWB.py`, the settings are specified in a parameter file, where `param.ini` is shown as an example in the repository. As an example, the code takes 7-8 minutes to create the examples `SFH1_50_20_*_example.txt`.

## Clarification on some of the formulas

References to equations are as given in [MT1](/references/master_thesis_Seppe_Staelens.pdf), unless otherwise mentioned. The clarifications here are to correct numerical factors in [MT1](/references/master_thesis_Seppe_Staelens.pdf), that have been adapted for [Staelens, Nelemans 2024](https://www.aanda.org/articles/aa/full_html/2024/03/aa48429-23/aa48429-23.html).

### Expressions for Omega

For the bulk, $\Omega$ is calculated as in (2.45), with a different prefactor 8.10E-9 / $S$ where $S$ is a normalization factor introduced through the SeBA code: SeBa takes as input a certain amount of mass $S$ available for star formation. This factor $S$ is 1.5E6 in [MT1](/references/master_thesis_Seppe_Staelens.pdf), 4E6 in [Staelens, Nelemans 2024](https://www.aanda.org/articles/aa/full_html/2024/03/aa48429-23/aa48429-23.html) and 3.4E6 for [2407.10642](https://arxiv.org/abs/2407.10642).

Similarly, the constant prefactor in (2.48) is changed to 1.28E-8 / $S$, instead of 8.5E-15, for the birth and merger contributions.

### Expressions for z contribution

In the bulk part of the code, the $z$ contributions are saved as $\Omega$ / (some factor), in order not to work with small numbers. This is the reason the $z$ contributions in the merger and birth part are saved as $\Omega$ / (this normalization) as well, to keep the relative contributions the same (as that is all we are interested in). This is admittedly still a bit messy in the code, and should be reworked.

### Expressions for number of binaries

This part is not in [MT1](/references/master_thesis_Seppe_Staelens.pdf). I determined the number of binaries in each $z$-$f$ bin as follows:

$$ N(z, f) = (4 \pi \cdot \chi(z)^2 \cdot \Delta \chi(z)) \cdot n (z, f) , $$

where $n(z, f)$ is the number density of systems in the bin. The latter is given by

$$ n(z, f) = \sum\*k \frac{\psi(z; k)}{S} \cdot \tau(z, f; k) .$$

In this expression, $\psi$ is again the SFH, determined at the birth time of the system and normalized by $S$ as explained above; and $\tau$ is the time it takes the system to traverse the bin. The reasoning is that all systems produced in the past, during a time corresponding to $\tau$, will have moved to the bin under consideration.

In the code, $\tau$ is calculated in Myr, and therefore multiplied by 1E6 as $\psi$ has units of 1/yr.

## Citation

If you want to cite this code, please cite the accompanying papers

```
@article{staelens2024likelihood,
	author = {{Staelens, Seppe} and {Nelemans, Gijs}},
	title = {Likelihood of white dwarf binaries to dominate the astrophysical gravitational wave background in the mHz band},
	DOI= "10.1051/0004-6361/202348429",
	url= "https://doi.org/10.1051/0004-6361/202348429",
	journal = {A&A},
	year = 2024,
	volume = 683,
	pages = "A139",
}
```

and

```
@misc{hofman2024uncertaintywhitedwarfastrophysical,
      title={On the uncertainty of the White Dwarf Astrophysical Gravitational Wave Background},
      author={Sophie Hofman and Gijs Nelemans},
      year={2024},
      eprint={2407.10642},
      archivePrefix={arXiv},
      primaryClass={astro-ph.HE},
      url={https://arxiv.org/abs/2407.10642},
}
```

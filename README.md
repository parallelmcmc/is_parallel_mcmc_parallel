
# "How Parallel is Parallel Markov-Chain Monte Carlo Really?"

## Codebase Overview
The experimental settings can be accessed in the following paths.

* Main scripts: `scripts/pmcmc/*`
* Data and datasets: `data/pmcmc/*`, `data/dataset/*`
* Benchmark problems: `scripts/pmcmc/tasks/*`
* Sampler settings  : `scripts/pmcmc/samplers/*`

The sampler implementations are in `src/AdvancedHMC.jl/src`.
Mainly, all of the major changes related to NUTS-INCA and GMH-HMC were made to the file `src/AdvancedHMC.jl/sampler.jl`.
Implementations of *gradient-adaptive metropolis-adjusted langevin* algorithm (gadMALA) and the *delayed rejection adaptive Metropolis* algorithm (DRAM) can also be found.
However, results with these algorithms were not included in our paper due to convergence issues.

## Installation
This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> parallel_smc

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.activate("is_parallel_mcmc_parallel")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box.

## Reproduction
All of our code can be executed from `scripts/pmcmc*`.

### Ground Truth Samples
The ground truth samples (really long MCMC chains) can be obtained by executing the code below.
```julia
include("run_experiments.jl"); main(:reference)
```
The reference chains will be added to `data/pmcmc/reference`.

### Experiments
All of experiments can be reproduced by executing the following.
```julia
include("run_experiments.jl"); main(:experiment)
```
The resulting raw data will be added to `data/pmcmc/exp_raw`.

### Post-Processing Data
The raw data mostly contain the chain-wise average (averaged over the parallel chains) for memory reasons.
To process the raw data execute the following code.
```julia
include("process_data.jl"); main(:all)
```
The resulting raw data will be added to `data/pmcmc/exp_pro`.
All of our figures in our paper utilize the generated files.


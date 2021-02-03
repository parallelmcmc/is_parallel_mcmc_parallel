
# emcee  = PyCall.pyimport("emcee")
# numpy  = PyCall.pyimport("numpy")
# random = PyCall.pyimport("random")

# function run_emcee(prng, ℓπ, dims, init_dist, n_samples, n_adapts, n_chains;
#                    show_progress=true, inca=false)
#     random.seed(Int(prng.seed))
#     numpy.random.seed(Int(prng.seed))

#     n_samples = n_samples*16
#     n_adapts  = n_adapts*16

#     sampler = emcee.EnsembleSampler(n_chains, dims, ℓπ,
#                                     moves=emcee.moves.StretchMove())
#     θ0      = rand(prng, init_dist, n_chains)
#     state   = sampler.run_mcmc(Array(θ0'), ceil(Int, n_adapts / n_chains),
#                                skip_initial_state_check=true,
#                                progress=true)
#     sampler.reset()
#     sampler.run_mcmc(state, ceil(Int, n_samples / n_chains),
#                      skip_initial_state_check=true,
#                      progress=true);
#     samples = sampler.get_chain(flat=false)
#     samples = permutedims(samples, [1, 3, 2])
#     samples
# end

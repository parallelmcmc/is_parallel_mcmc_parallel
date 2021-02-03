
function run_nuts(prng, ℓπ, dims, init_dist, n_samples, n_adapts, n_chains;
                  show_progress=true, inca=false)
    metric      = AHMC.DiagEuclideanMetric(dims)
    hamiltonian = AHMC.Hamiltonian(metric, ℓπ, Zygote)
    initial_θ   = rand(prng, init_dist)

    # Define a leapfrog solver, with initial step size chosen heuristically
    initial_ϵ  = AHMC.find_good_stepsize(prng, hamiltonian, initial_θ)
    integrator = AHMC.Leapfrog(initial_ϵ)
    proposal   = AHMC.NUTS{AHMC.MultinomialTS,
                           AHMC.GeneralisedNoUTurn}(integrator)
    adaptor    =  begin
        if(n_adapts > 150)
            AHMC.StanHMCAdaptor(AHMC.MassMatrixAdaptor(metric),
                                AHMC.StepSizeAdaptor(0.85, integrator))
        else
            AHMC.NaiveHMCAdaptor(AHMC.MassMatrixAdaptor(metric),
                                 AHMC.StepSizeAdaptor(0.85, integrator))
        end
    end

    # Run the sampler to draw samples from the specified Gaussian, where
    #   - `samples` will store the samples
    #   - `stats` will store diagnostic statistics for each sample
    samples, stats, adaptor = AHMC.sample(prng,
                                          hamiltonian,
                                          proposal,
                                          initial_θ,
                                          n_samples + n_adapts,
                                          adaptor,
                                          n_adapts;
                                          generalized=false,
                                          n_chains=n_chains,
                                          progress=show_progress,
                                          inca=inca,
                                          verbose=false,
                                          drop_warmup=true)
    summ = summarize(samples)
    summ[:acceptance_rate] = hcat([i.acceptance_rate for i in stats]...)
    if(inca)
        summ[:preconditioner]  = AHMC.getM⁻¹(adaptor)
    else
        summ[:preconditioner]  = AHMC.getM⁻¹.(adaptor)
    end
    summ
end

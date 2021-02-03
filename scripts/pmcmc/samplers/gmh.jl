
function run_gmh_hmcda(prng, ℓπ, dims, init_dist, n_samples, n_adapts, n_chains;
                       show_progress=true)
    metric      = AHMC.DiagEuclideanMetric(dims)
    hamiltonian = AHMC.Hamiltonian(metric, ℓπ, Zygote)
    initial_θ   = rand(prng, init_dist)

    # Define a leapfrog solver, with initial step size chosen heuristically
    initial_ϵ  = AHMC.find_good_stepsize(prng, hamiltonian, initial_θ)
    integrator = AHMC.Leapfrog(initial_ϵ)
    proposal   = AHMC.StaticTrajectory(integrator, 32)

    adaptor    =  begin
        if(n_adapts > 150)
            AHMC.StanHMCAdaptor(AHMC.MassMatrixAdaptor(metric),
                                AHMC.StepSizeAdaptor(0.65, integrator))
        else
            AHMC.NaiveHMCAdaptor(AHMC.MassMatrixAdaptor(metric),
                                 AHMC.StepSizeAdaptor(0.65, integrator))
        end
    end

    samples, guide, weights = AHMC.sample(prng,
                                          hamiltonian,
                                          proposal,
                                          initial_θ,
                                          n_samples + n_adapts,
                                          adaptor,
                                          n_adapts;
                                          verbose=false,
                                          generalized=true,
                                          n_chains=n_chains,
                                          progress=show_progress,
                                          drop_warmup=true)
    summ = gmh_summarize(samples, weights)
    summ[:guide] = guide
    summ
end

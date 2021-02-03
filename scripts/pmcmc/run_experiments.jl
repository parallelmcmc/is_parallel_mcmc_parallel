
@everywhere using DrWatson
@everywhere @quickactivate "parallel_smc"
@everywhere include(joinpath(srcdir(), "parallel_smc.jl"))

@everywhere using Base.Iterators
@everywhere using CSV
@everywhere using Distributed
@everywhere using Distributions
@everywhere using ForwardDiff
@everywhere using JLD
@everywhere using KahanSummation
@everywhere using MCMCChains
@everywhere using ProgressMeter
@everywhere using Random
@everywhere using StatsFuns
@everywhere using Zygote

@everywhere import .AdvancedHMC
@everywhere AHMC = AdvancedHMC

@everywhere include("utils.jl")
@everywhere include("tasks/tasks.jl")
@everywhere include("samplers/samplers.jl")

function run_experiment(fn,
                        global_settings::Dict,
                        n_samples_seq::Int,
                        n_adapts_seq::Int,
                        n_reps::Int,
                        ℓπ,
                        dims,
                        init_dist;
                        kwargs...)
    for n_chains in [1, 8, 16, 32, 64, 128]
        n_samples = begin
            if(n_chains == 1)
                n_samples_seq
            else
                floor(Int64, n_samples_seq / n_chains * 2)
            end
        end
        n_adapts  = floor(Int64, n_adapts_seq / n_chains)
        settings = Dict(:n_samples  => n_samples,
                        :n_adapts   => n_adapts,
                        :n_chain    => n_chains)
        settings = merge(settings, global_settings)
        fname    = savename(settings, "jld")

        if(isfile(datadir("pmcmc", "exp_raw", fname)))
            @info "Skipping $n_chains chains with " settings=settings 
            continue
        end

        @info "Running $n_chains chains with " settings=settings 
        results = @showprogress pmap(1:n_reps) do i
            _kwargs = kwargs
            prng    = MersenneTwister(i)
            results = fn(prng, ℓπ, dims, init_dist,
                         n_samples, n_adapts, n_chains;
                         _kwargs...)
        end
        results = Dict([key=>[res[key] for res in results]
                        for key in keys(results[1])])
        results = merge(results, settings)
        wsave(datadir("pmcmc", "exp_raw", fname), "data", results,
              compatible=true, compress=true)
        GC.gc()
    end
end

function run_all_experiments(name, ℓπ, dims, init_dist)
    n_reps        = 1024
    n_samples_seq = 4096
    n_adapts_seq  = 4096

    settings = Dict(:method  => "gmh_hmcda",
                    :problem => name)
    run_experiment(run_gmh_hmcda,
                   settings,
                   n_samples_seq,
                   n_adapts_seq,
                   n_reps,
                   ℓπ,
                   dims,
                   init_dist;
                   show_progress=false)

    settings      = Dict(:method  => "nuts",
                         :problem => name)
    run_experiment(run_nuts,
                   settings,
                   n_samples_seq,
                   n_adapts_seq,
                   n_reps,
                   ℓπ,
                   dims,
                   init_dist;
                   show_progress=false,
                   inca=false)

    settings = Dict(:method  => "nuts_inca",
                    :problem => name)
    run_experiment(run_nuts,
                   settings,
                   n_samples_seq,
                   n_adapts_seq,
                   n_reps,
                   ℓπ,
                   dims,
                   init_dist;
                   show_progress=false,
                   inca=true)

    if(name ∉ ["stochastic_volatility",
               "gaussian_dim=250_init=4.0_corr=true",
               "logistic"])
        settings = Dict(:method  => "gadmala_inca",
                        :problem => name)
        run_experiment(run_gadmala,
                       settings,
                       n_samples_seq,
                       n_adapts_seq,
                       n_reps,
                       ℓπ,
                       dims,
                       init_dist;
                       ρ_L=1e-4,
                       show_progress=false,
                       inca=true)
    end

    # settings = Dict(:method  => "emcee",
    #                 :problem => name)
    # run_experiment(run_emcee,
    #                settings,
    #                n_samples_seq,
    #                n_adapts_seq,
    #                n_reps,
    #                ℓπ,
    #                dims,
    #                init_dist;
    #                show_progress=true,
    #                inca=true)
end

function run_reference(task)
    n_samples = 2^20
    n_adapts  = 2^12
    n_chains  = 1
    prng      = MersenneTwister(1)
    run_ref_impl = (name, ℓπ, dims, init_dist)->begin
        metric      = AHMC.DiagEuclideanMetric(dims)
        hamiltonian = AHMC.Hamiltonian(metric, ℓπ, Zygote)
        initial_θ   = rand(prng, init_dist)

        initial_ϵ  = AHMC.find_good_stepsize(prng, hamiltonian, initial_θ)
        integrator = AHMC.Leapfrog(initial_ϵ)
        proposal   = AHMC.NUTS{AHMC.MultinomialTS,
                               AHMC.GeneralisedNoUTurn}(integrator)
        adaptor = AHMC.StanHMCAdaptor(AHMC.MassMatrixAdaptor(metric),
                                      AHMC.StepSizeAdaptor(0.85, integrator))
        samples, stats = AHMC.sample(
            prng,
            hamiltonian,
            proposal,
            initial_θ,
            n_samples + n_adapts,
            adaptor,
            n_adapts;
            generalized=false,
            n_chains=n_chains,
            progress=true,
            inca=false,
            verbose=false,
            drop_warmup=true)
        samples
    end
    task(run_ref_impl)
end


function main(mode=:experiment)
    if(mode == :experiment)
        gaussian(run_all_experiments, 25,  1e-5, false)
        gaussian(run_all_experiments, 25,  2.0,  false)
        gaussian(run_all_experiments, 25,  4.0,  false)
        gaussian(run_all_experiments, 25,  8.0,  false)

        gaussian(run_all_experiments, 50,  4.0,  false)
        gaussian(run_all_experiments, 100, 4.0,  false)

        #gaussian(run_all_experiments, 250, 4.0,  true)
        logistic(run_all_experiments)
        eight_schools(run_all_experiments)
        stochastic_volatility(run_all_experiments)
    else
        samples = run_reference(stochastic_volatility)
        JLD.save(datadir("pmcmc", "reference", "stochastic_volatility.jld"),
                 "samples", samples,
                 compress=true)

        samples = run_reference(logistic)
        JLD.save(datadir("pmcmc", "reference", "logistic.jld"),
                 "samples", samples,
                 compress=true)

        samples = run_reference(eight_schools)
        JLD.save(datadir("pmcmc", "reference", "eight_schools.jld"),
                 "samples", samples,
                 compress=true)
    end
end


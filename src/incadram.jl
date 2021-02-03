
function mh_transition(prng, q, logπ, prev_θ, prev_p)
    @label mh_propose
    prop_θ = rand(prng, q)
    prop_p = logπ(prop_θ)

    if(isinf(prop_p))
        @goto mh_propose
    end

    α = prop_p - prev_p
    u = rand(prng)

    if(min(0.0, α) > log(u))
        prop_θ, prop_p, prop_p, true
    else
        prev_θ, prev_p, prop_p, false
    end
end

function delayed_rejection(prng, Σ, logπ,
                           prev_θ::AbstractVector,
                           prev_p::Real,
                           delays::Int,
                           online_stat)
#=
    Implementation strategy described in
    A. Mira. On Metropolis-Hastings algorithms with delayed rejection. 
    Metron, 2001, Vol. LIX, n. 3-4, pp. 231-241.
=##
    prop_q = MvNormal(prev_θ, Σ)
    next_θ, next_p, prop_p, acc = mh_transition(prng, prop_q, logπ,
                                                prev_θ, prev_p) 
    if(!acc && delays > 0)
        σ      = 1.0
        best_p = prop_p
        
        for i = 1:delays
            OnlineStats.fit!(online_stat, 0)
            σ /= 100.0

            @label dr_propose
            prop_θ = rand(prng, MvNormal(next_θ, σ*Σ))
            prop_p = logπ(prop_θ)

            if(isinf(prop_p))
                @goto dr_propose
            end

            best_expp = exp(best_p)
            P = (exp(prop_p) - best_expp) / (prev_p - best_expp)
            u = rand(prng)

            next_θ = prop_θ
            next_p = prop_p
            if(prop_p >= prev_p)
                acc = true
                break
            elseif(prop_p < best_p)
                continue
            elseif(P > u)
                acc = true
                break
            else
                best_p = max(best_p, prop_p)
            end
        end

        if(!acc)
            next_θ = prev_θ
            next_p = prev_p
        end
    end

    OnlineStats.fit!(online_stat, Int(acc))
    next_θ, next_p, Int64(acc)
end

function inca_dram(logπ,
                   initial_q,
                   n_samples::Int,
                   n_burn::Int,
                   n_chains::Int;
                   n_adaptfreq::Int=100,
                   prng=MersenneTwister(1),
                   rejection_delays::Int=0,
                   show_progress::Bool=true)
#=
    Solonen, Antti, et al. 
    "Efficient MCMC for climate model parameter estimation: 
     Parallel adaptive chains and early rejection." 
    Bayesian Analysis 7.3 (2012): 715-736.

    Haario, Heikki, Eero Saksman, and Johanna Tamminen. 
    "An adaptive Metropolis algorithm." Bernoulli 7.2 (2001): 223-242.
    
    Haario, Heikki, et al. 
    "DRAM: efficient adaptive MCMC."  Statistics and computing 16.4 (2006): 339-354.

    Roberts, Gareth O., and Jeffrey S. Rosenthal. 
    "Optimal scaling for various Metropolis-Hastings algorithms." 
    Statistical science 16.4 (2001): 351-367.
=##
    state_θ = hcat([rand(prng, initial_q) for i = 1:n_chains]...)
    state_p = [logπ(state_θ[:,c]) for c = 1:n_chains]
    n_vars  = size(state_θ, 1)
    samples = zeros(Float64, n_samples+n_burn, n_vars, n_chains)

    σ       = 2.38^2 / n_vars
    Σ_am    = σ / 1e+2 * I
    Σ       = deepcopy(Σ_am)
    n_am    = 0

    total_acc = 0
    buf_acc   = Vector{Float64}(undef, n_chains)
    prog      = ProgressMeter.Progress(n_samples+n_burn)
    inca_μ    = zeros(n_vars)
    acc_stat  = OnlineStats.Mean()
    for t = 1:n_burn+n_samples
        for c = 1:n_chains
            prev_θ = state_θ[:,c]
            prev_p = state_p[c]

            next_θ, next_p, acc = delayed_rejection(
                prng, Σ, logπ, prev_θ,
                prev_p, rejection_delays,
                acc_stat)

            state_θ[:,c]   = next_θ
            state_p[c]     = next_p

            #if(t > n_burn)
            #    samples[t-n_burn,:,c] = next_θ
                samples[t,:,c] = next_θ
            #else
            if(t <= n_burn)
                buf_acc[c]     = acc 
            end
        end

        if(t <= n_burn)
            # Simple trick used by mcmcstat
            total_acc += sum(buf_acc)
            if(t % 32 == 0)
                avg_acc = total_acc / 32 / n_chains
                if(avg_acc > 0.95)
                    Σ *= 2
                elseif(avg_acc < 0.5)
                    Σ /= 2
                end
                total_acc = 0
            end
        end

        if(t  == n_burn)
            burn_samples = samples[1:n_burn,:,:]
            burn_samples = permutedims(samples, (1,3,2))
            burn_samples = reshape(samples, (:, n_vars))
            inca_μ       = mean(burn_samples, dims=1)[1,:]
            Σ_am         = cov(burn_samples, corrected=false)
            n_am         = n_burn*n_chains
            Σ = PDMats.PDMat(σ * Σ_am)
        end

        if(t > n_burn && t % n_adaptfreq == 0)
            for c = 1:n_chains
                for i = 1:n_adaptfreq
                    θ_am    = samples[t-n_adaptfreq+i, :, c]
                    n_am   += 1 

                    Δθ      = θ_am - inca_μ
                    inca_μ += (θ_am - inca_μ) / n_am
                    Σ_am    = ((n_am - 1) / n_am * Σ_am) + ((Δθ * Δθ') / n_am)
                end
            end
            Σ = PDMats.PDMat(σ * Σ_am)
        end

        if(show_progress)
            ProgressMeter.next!(prog; showvalues=[
                (:iteration, t),
                (:average_acceptance_rate, acc_stat.μ),
                (:average_probability,     mean(state_p)),
            ])
        end
    end
    samples[n_burn+1:end,:,:]
end

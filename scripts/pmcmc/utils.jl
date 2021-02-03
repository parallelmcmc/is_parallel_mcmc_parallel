
function summarize(samples)
    n_iters  = size(samples,1)
    n_vars   = size(samples,2)
    n_chains = size(samples,3)

    divisors    = n_chains * (1:n_iters)
    running_avg = cumsum_kbn(samples, 1) ./ divisors
    running_avg = sum(running_avg, dims=3)[:,:,1]

    running_std = Array{Float64}(undef, n_iters, n_vars)
    @inbounds for i = 1:n_iters
        @simd for j = 1:n_vars
            running_std[i,j] = std(samples[1:i,j,:];
                                   mean=running_avg[i,j],
                                   corrected=true)
        end
    end

    res = Dict()
    res[:est_mean] = running_avg
    res[:est_std]  = running_std
    res
end

function gmh_summarize(samples, weights)
    n_iters  = size(samples,1)
    n_vars   = size(samples,2)
    n_chains = size(samples,3)

    Ewh  = Array{Float64}(undef, n_iters, n_vars)
    Ewh² = Array{Float64}(undef, n_iters, n_vars)
    for t = 1:n_iters
        Ewh[t,:]  = samples[t,:,:]    * weights[t,:] 
        Ewh²[t,:] = samples[t,:,:].^2 * weights[t,:]
    end

    divisors     = 1:n_iters
    run_avg_Ewh  = cumsum_kbn(Ewh,  1) ./ divisors
    run_avg_Ewh² = cumsum_kbn(Ewh², 1) ./ divisors

    res = Dict()
    res[:est_mean] = run_avg_Ewh
    res[:est_std]  = sqrt.(max.(run_avg_Ewh² - run_avg_Ewh.^2, eps(Float64)))
    res
end

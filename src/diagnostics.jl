
function precompute_fft(samples; fft_plan=nothing, ifft_plan=nothing)
    niter = length(samples)

    # create cache for FFT
    T = complex(eltype(samples))
    n = nextprod([2, 3], 2 * niter - 1)
    samples_cache = Vector{T}(undef, n)

    # create plans of FFTs
    fft_plan  = isnothing(fft_plan)  ? plan_fft!(samples_cache, 1)  : fft_plan
    ifft_plan = isnothing(ifft_plan) ? plan_ifft!(samples_cache, 1) : ifft_plan

    samples = samples .- mean(samples)
    samples_cache[1:niter]          = samples
    samples_cache[(niter + 1):end] .= zero(T)

    fft_plan * samples_cache
    @. samples_cache = abs2(samples_cache)
    ifft_plan * samples_cache
    return samples_cache
end

function lagged_autocor(k::Int, samples_cache)
    ϵ = eps(Float64)
    real(samples_cache[k+1]) / (real(samples_cache[1]) + ϵ)
end

function init_fft(samples)
    niter = length(samples)

    # create cache for FFT
    T = complex(eltype(samples))
    n = nextprod([2, 3], 2 * niter - 1)
    samples_cache = Vector{T}(undef, n)

    # create plans of FFTs
    fft_plan  = plan_fft!(samples_cache, 1)
    ifft_plan = plan_ifft!(samples_cache, 1)
    fft_plan, ifft_plan
end

function fft_autocor(samples::AbstractArray, max_lag::Int=250;
                     fft_plan=nothing, ifft_plan=nothing)
    max_lag       = min(floor(Int, length(samples) / 2), max_lag)
    samples_cache = precompute_fft(samples; fft_plan=fft_plan, ifft_plan=ifft_plan)

    ρ_even = 1
    ρ_odd  = lagged_autocor(1, samples_cache)

    pₜ     = ρ_odd + ρ_even
    sum_pₜ = pₜ
    for k = 2:2:max_lag
        ρ_even = lagged_autocor(k,   samples_cache)
        ρ_odd  = lagged_autocor(k+1, samples_cache)
        Δ      = ρ_odd + ρ_even
        if(Δ <= 0)
            break
        end
        pₜ      = min(pₜ, Δ)
        sum_pₜ += pₜ
    end
    τ = 2 * sum_pₜ - 1
    τ
end

function fft_autocov(samples::AbstractArray, max_lag::Int=250;
                     fft_plan=nothing, ifft_plan=nothing)
    τ  =  fft_autocor(samples, max_lag; fft_plan=fft_plan, ifft_plan=ifft_plan)
    σ² = var(samples, corrected=true)
    τ*σ²
end

function autocov_elemwise(samples::AbstractArray)
    n_param             = size(samples, 2)
    fft_plan, ifft_plan = init_fft(samples[:,1])
    max_lag             = min(250, round(Int64,size(samples,1)/2))
    map(1:n_param) do i
        fft_autocov(samples[:,i], max_lag;
                    fft_plan=fft_plan, ifft_plan=ifft_plan)
    end
end

function autocor_elemwise(samples::AbstractArray;
                          fft_plan=nothing,
                          ifft_plan=nothing)
    n_param             = size(samples, 2)
    fft_plan, ifft_plan = begin
        if(isnothing(fft_plan))
            init_fft(samples[:,1,1])
        else
            fft_plan, ifft_plan
        end
    end
    max_lag = min(250, round(Int64,size(samples,1)/2))
    map(1:n_param) do i
        fft_autocov(samples[:,i], max_lag;
                    fft_plan=fft_plan, ifft_plan=ifft_plan)
    end
end

function lwp_ess(samples, weights)
    # iteration × dimension × chain
    Ewh  = zeros(size(samples)[1:2]...)
    Ewh2 = zeros(size(samples)[1:2]...)
    for t = 1:size(samples, 1)
        Ewh[t,:]  = samples[t,:,:] * weights[t,:]
        Ewh2[t,:] = (samples[t,:,:].^2) * weights[t,:]
    end

    Eh   = mean(Ewh,  dims=1)[1,:]
    Eh²  = mean(Ewh2, dims=1)[1,:]

    n  = size(samples, 1)
    N  = size(samples, 2)

    ρ  = autocov_elemwise(Ewh)
    σ² = Eh² - Eh.^2

    ESS_lwp = n * σ² ./ ρ
    ESS_lwp
end

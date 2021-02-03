# Update of hamiltonian and proposal

reconstruct(h::Hamiltonian, ::AbstractAdaptor) = h
function reconstruct(
    h::Hamiltonian, adaptor::Union{MassMatrixAdaptor, NaiveHMCAdaptor, StanHMCAdaptor}
)
    metric = renew(h.metric, getM⁻¹(adaptor))
    return reconstruct(h, metric=metric)
end

reconstruct(τ::AbstractProposal, ::AbstractAdaptor) = τ
function reconstruct(
    τ::AbstractProposal, adaptor::Union{StepSizeAdaptor, NaiveHMCAdaptor, StanHMCAdaptor}
)
    # FIXME: this does not support change type of `ϵ` (e.g. Float to Vector)
    # FIXME: this is buggy for `JitteredLeapfrog`
    integrator = reconstruct(τ.integrator, ϵ=getϵ(adaptor))
    return reconstruct(τ, integrator=integrator)
end

function resize(h::Hamiltonian, θ::AbstractVecOrMat{T}) where {T<:AbstractFloat}
    metric = h.metric
    if size(metric) != size(θ)
        metric = typeof(metric)(size(θ))
        h = reconstruct(h, metric=metric)
    end
    return h
end

##
## Interface functions
##

function sample_init(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}}, 
    h::Hamiltonian, 
    θ::AbstractVecOrMat{<:AbstractFloat}
)
    # Ensure h.metric has the same dim as θ.
    h = resize(h, θ)
    # Initial transition
    t = Transition(phasepoint(rng, θ, h), NamedTuple())
    return h, t
end

# A step is a momentum refreshment plus a transition
function step(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}}, 
    h::Hamiltonian, 
    τ::AbstractProposal, 
    z::PhasePoint
)
    # Refresh momentum
    z = refresh(rng, z, h)
    # Make transition
    return transition(rng, τ, h, z)
end

Adaptation.adapt!(
    h::Hamiltonian,
    τ::AbstractProposal,
    adaptor::Adaptation.NoAdaptation,
    i::Int,
    n_adapts::Int,
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat}
) = h, τ, false

function Adaptation.adapt!(
    h::Hamiltonian,
    τ::AbstractProposal,
    adaptor::AbstractAdaptor,
    i::Int,
    n_adapts::Int,
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat}
)
    isadapted = false
    if i <= n_adapts
        i == 1 && Adaptation.initialize!(adaptor, n_adapts)
        adapt!(adaptor, θ, α; adapt_stepsize=true)
        i == n_adapts && finalize!(adaptor)
        h = reconstruct(h, adaptor)
        τ = reconstruct(τ, adaptor)
        isadapted = true
    end
    return h, τ, isadapted
end

function Adaptation.adapt!(
    h::Hamiltonian,
    τ::AbstractProposal,
    adaptor::AbstractAdaptor,
    i::Int,
    n_adapts::Int,
    θs::Vector{<:AbstractVector},
    αs::AbstractVector{<:AbstractFloat}
)
    isadapted = false
    if i <= n_adapts
        i == 1 && Adaptation.initialize!(adaptor, n_adapts)
        adapt!(adaptor, θs[1], mean(αs); adapt_stepsize=true)
        i == n_adapts && finalize!(adaptor)

        for θ in θs
            adapt!(adaptor, θ, αs[1]; adapt_covariance=true)
        end

        h = reconstruct(h, adaptor)
        τ = reconstruct(τ, adaptor)
        isadapted = true
    end
    return h, τ, isadapted
end

function Adaptation.adapt!(
    h::AbstractArray,
    τ::AbstractArray,
    adaptors::AbstractVector{<:AbstractAdaptor},
    i::Int,
    n_adapts::Int,
    θs::Vector{<:AbstractVector},
    αs::AbstractVector{<:AbstractFloat}
)
    isadapted = false
    if i <= n_adapts
        for c = 1:length(adaptors)
            i == 1 && Adaptation.initialize!(adaptors[c], n_adapts)
            adapt!(adaptors[c], θs[c], αs[c];
                   adapt_stepsize=true, adapt_covariance=true)
            i == n_adapts && finalize!(adaptors[c])
            h[c] = reconstruct(h[c], adaptors[c])
            τ[c] = reconstruct(τ[c], adaptors[c])
        end
        isadapted = true
    end
    return h, τ, isadapted
end

"""
Progress meter update with all trajectory stats, iteration number and metric shown.
"""
function pm_next!(pm, stat::NamedTuple)
    ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stat)])
end

"""
Simple progress meter update without any show values.
"""
simple_pm_next!(pm, stat::NamedTuple) = ProgressMeter.next!(pm)

##
## Sampling functions
##

sample(
    h::Hamiltonian,
    τ::AbstractProposal,
    θ::AbstractVecOrMat{<:AbstractFloat},
    n_samples::Int,
    adaptor::AbstractAdaptor=NoAdaptation(),
    n_adapts::Int=min(div(n_samples, 10), 1_000);
    generalized::Bool=false,
    inca::Bool=false,
    n_chains::Int=1,
    drop_warmup=false,
    interchain::Bool=false,
    verbose::Bool=true,
    progress::Bool=false,
    (pm_next!)::Function=pm_next!
) = sample(
    GLOBAL_RNG,
    h,
    τ,
    θ,
    n_samples,
    adaptor,
    n_adapts;
    generalized=generalized,
    inca=inca,
    n_chains=n_chains,
    drop_warmup=drop_warmup,
    verbose=verbose,
    progress=progress,
    (pm_next!)=pm_next!,
)


"""
    sample(
        rng::AbstractRNG,
        h::Hamiltonian,
        τ::AbstractProposal,
        θ::AbstractVecOrMat{T},
        n_samples::Int,
        adaptor::AbstractAdaptor=NoAdaptation(),
        n_adapts::Int=min(div(n_samples, 10), 1_000);
        drop_warmup::Bool=false,
        verbose::Bool=true,
        progress::Bool=false
    )

Sample `n_samples` samples using the proposal `τ` under Hamiltonian `h`.
- The randomness is controlled by `rng`. 
    - If `rng` is not provided, `GLOBAL_RNG` will be used.
- The initial point is given by `θ`.
- The adaptor is set by `adaptor`, for which the default is no adaptation.
    - It will perform `n_adapts` steps of adaptation, for which the default is the minimum of `1_000` and 10% of `n_samples`
- `drop_warmup` controls to drop the samples during adaptation phase or not
- `verbose` controls the verbosity
- `progress` controls whether to show the progress meter or not
"""

function sample(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    h::Hamiltonian,
    τ::AbstractProposal,
    θ::T,
    n_samples::Int,
    adaptor::AbstractAdaptor=NoAdaptation(),
    n_adapts::Int=min(div(n_samples, 10), 1_000);
    generalized::Bool=false,
    inca::Bool=false,
    n_chains::Int=1,
    drop_warmup=false,
    verbose::Bool=true,
    progress::Bool=false,
    (pm_next!)::Function=pm_next!
) where {T<:AbstractVecOrMat{<:AbstractFloat}}
    if(generalized && (typeof(τ) <: HMCDA || typeof(τ) <: StaticTrajectory))
        gmh_impl(
            rng,
            h,
            τ,
            θ,
            n_samples,
            adaptor,
            n_adapts;
            n_chains=n_chains,
            drop_warmup=drop_warmup,
            verbose=verbose,
            progress=progress,
            (pm_next!)=pm_next!)
    else
        mh_impl(
            rng,
            h,
            τ,
            θ,
            n_samples,
            adaptor,
            n_adapts;
            inca=inca,
            n_chains=n_chains,
            drop_warmup=drop_warmup,
            verbose=verbose,
            progress=progress,
            (pm_next!)=pm_next!)
    end
end

function mh_impl(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    h::Hamiltonian,
    τ::AbstractProposal,
    θ::T,
    n_samples::Int,
    adaptor::AbstractAdaptor=NoAdaptation(),
    n_adapts::Int=min(div(n_samples, 10), 1_000);
    inca::Bool=false,
    n_chains::Int=1,
    drop_warmup=false,
    verbose::Bool=true,
    progress::Bool=false,
    (pm_next!)::Function=pm_next!
) where {T<:AbstractVecOrMat{<:AbstractFloat}}
    @assert !(drop_warmup && (adaptor isa Adaptation.NoAdaptation)) "Cannot drop warmup samples if there is no adaptation phase."
    # Prepare containers to store sampling results
    n_keep = n_samples - (drop_warmup ? n_adapts : 0)
    θs    = [Vector{T}(undef, n_keep)          for i = 1:n_chains]
    stats = Vector{NamedTuple}(undef, n_keep)
    # Initial sampling
    init_vars = [sample_init(rng, h, θ) for i = 1:n_chains]
    h         = Any[init_var[1] for init_var in init_vars]
    t         = Any[init_var[2] for init_var in init_vars]
    τ         = Any[τ           for i = 1:n_chains]
    adaptor   = inca ? adaptor : [deepcopy(adaptor) for i = 1:n_chains]
    αs        = Vector{Float64}(undef, n_chains)

    # Progress meter
    pm = progress ? ProgressMeter.Progress(n_samples, desc="Sampling", barlen=31) : nothing

    time = @elapsed for i = 1:n_samples
        for c = 1:n_chains
            # Make a step
            if(inca)
                t[c] = step(rng, h[1], τ[1], t[c].z)
            else
                t[c] = step(rng, h[c], τ[c], t[c].z)
            end
            # Adapt h and τ; what mutable is the adaptor
        end
        tstats   = [stat(t_i)             for t_i in t]
        αs       = [tstat.acceptance_rate for tstat in tstats]

        θ_current = [t_i.z.θ for t_i in t]
        if(inca)
            ĥ, τ̂, isadapted = adapt!(h[1], τ[1], adaptor, i, n_adapts, θ_current, αs)
            h[1] = ĥ
            τ[1] = τ̂
        else
            h, τ, isadapted = adapt!(h, τ, adaptor, i, n_adapts, θ_current, αs)
        end

        # Update progress meter
        if progress
            ϵ = stepsize=nom_step_size(τ[1].integrator)
            L = stepsize=tstats[1][:n_steps]
            # Do include current iteration and mass matrix
            pm_next!(pm, (iterations=i,
                          acceptance=mean(αs),
                          mass_matrix=h[1].metric,
                          stepsize=ϵ,
                          numsteps=L))
            # Report finish of adapation
        elseif verbose && i == n_adapts
            @info "Finished $n_adapts adapation steps" adaptor τ[1].integrator h[1].metric acceptance=mean(αs)
        end
        # Store sample
        if !drop_warmup || i > n_adapts
            j = i - drop_warmup * n_adapts

            stats[j] = (acceptance_rate=αs,)
            for c = 1:n_chains
                θs[c][j] = θ_current[c]
            end
        end
    end
    θs = cat([Array(hcat(chain...)') for chain in θs]..., dims=3)
    return θs, stats, adaptor
end

struct GMHProposal{P<:PhasePoint}
    "Phase-point for the transition."
    z       ::  P
    "MH ratio in logarithmic scale"
    logα    ::  Float64
end

function gmh_compute_trajectory(rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
                                τ::StaticTrajectory,
                                h::Hamiltonian,
                                z::PhasePoint)
    H0 = energy(z)
    integrator = jitter(rng, τ.integrator)
    z′   = step(τ.integrator, h, z, τ.n_steps)
    H1 = energy(z′)
    logα = min(0, H0 - H1)
    return GMHProposal(z′, logα)
end

function gmh_compute_trajectory(rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
                                τ::HMCDA{S},
                                h::Hamiltonian,
                                z::PhasePoint) where {S}
    n_steps = max(1, floor(Int, τ.λ / nom_step_size(τ.integrator)))
    static_τ = StaticTrajectory{S}(τ.integrator, n_steps)
    gmh_compute_trajectory(rng, static_τ, h, z)
end

function ghm_propose_transition(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}}, 
    h::Hamiltonian, 
    τ::AbstractProposal, 
    z::PhasePoint
)
    # Refresh momentum
    z = refresh(rng, z, h)
    prop = gmh_compute_trajectory(rng, τ, h, z)
    prop
end

function gmh_impl(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    h::Hamiltonian,
    τ::AbstractProposal,
    θ::T,
    n_samples::Int,
    adaptor::AbstractAdaptor=NoAdaptation(),
    n_adapts::Int=min(div(n_samples, 10), 1_000);
    n_chains::Int=1,
    n_resample::Int=n_chains,
    drop_warmup=false,
    verbose::Bool=true,
    progress::Bool=false,
    (pm_next!)::Function=pm_next!
) where {T<:AbstractVecOrMat{<:AbstractFloat}}
    @assert !(drop_warmup && (adaptor isa Adaptation.NoAdaptation)) "Cannot drop warmup samples if there is no adaptation phase."
    @assert n_resample <= n_chains
    # Prepare containers to store sampling results
    n_keep = n_samples - (drop_warmup ? n_adapts : 0)
    θs     = Matrix{T}(undef, n_chains+1, n_keep)
    θ0s    = Array{T}(undef, n_keep)
    ws     = zeros(Float64, n_keep, n_chains+1)
    # Initial sampling
    h, t   = sample_init(rng, h, θ) 

    w = Vector{Float64}(undef, n_chains+1)
    proposals = Array{GMHProposal}(undef, n_chains+1)

    # Progress meter
    pm = progress ? ProgressMeter.Progress(n_samples, desc="Sampling", barlen=31) : nothing

    time = @elapsed for i = 1:n_samples
        for c in 2:n_chains+1
            proposals[c] = ghm_propose_transition(rng, h, τ, t.z)
        end
        αs       = [proposals[c].logα for c = 2:n_chains+1]
        αs[isnan.(αs)] .= -Inf 
        logw     = αs .- log(n_chains)
        w[2:end] = exp.(logw)
        w[1]     = 1 - exp(logsumexp(logw))

        proposals[1] = GMHProposal(t.z, 0.0)
        resample_prop = Categorical(w)

        chain_idx = rand(rng, resample_prop)
        prop = proposals[chain_idx]
        z    = prop.z
        α    = exp(prop.logα)

        prop_θs     = [proposal_t.z.θ for proposal_t in proposals]
        resample_θs = prop_θs[rand(rng, resample_prop, n_chains)]

        αs = exp.(αs)
        h, τ, isadapted = adapt!(h, τ, adaptor, i, n_adapts, resample_θs, αs)
        avg_α = mean(αs)

        # Reverse momentum variable to preserve reversibility
        t = Transition(PhasePoint(z.θ, -z.r, z.ℓπ, z.ℓκ), t.stat)

        if progress
            ϵ = stepsize=nom_step_size(τ.integrator)
            # Do include current iteration and mass matrix
            pm_next!(pm, (iterations=i,
                          acceptance=avg_α,
                          mass_matrix=h.metric,
                          stepsize=ϵ))
            # Report finish of adapation
        elseif verbose && i == n_adapts
            @info "Finished $n_adapts adapation steps" adaptor τ.integrator h.metric
        end
        # Store sample
        if !drop_warmup || i > n_adapts
            j = i - drop_warmup * n_adapts

            θs[:,j] = prop_θs
            θ0s[j]  = t.z.θ
            ws[j,:] = w
        end
    end
    θ0s = Array(hcat(θ0s...)')
    θ0s = reshape(θ0s, (size(θ0s)..., 1))

    sz = size(θs) 
    θs = reshape(θs, :)
    θs = Array(hcat(θs...)')
    θs = reshape(θs, (sz..., length(θ)))
    θs = permutedims(θs, (2, 3, 1))
    return θs, θ0s, ws
end

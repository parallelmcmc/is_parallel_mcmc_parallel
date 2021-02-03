

function vi_grad_L(α, L, β, Lᵀ∇logpx_ϵ_half, Lᵀ∇logpy_half, ∇logpx, ∇logpy)
    if(α < 0)
        t1 = Lᵀ∇logpx_ϵ_half - Lᵀ∇logpy_half
        t2 = (∇logpx - ∇logpy) / 2 

        tril(-t1*t2' + diagm(β./diag(L)))
    else
        diagm(β./diag(L))
    end
end

function grad_adapt(it, L, β, ρ_β, ρ_L, accept, α, opt,
                    β1, β2, m0, v0, Lᵀ∇logpx_ϵ_half, Lᵀ∇logpy_half,
                    ∇logpx, ∇logpy)
    ∇L = begin
        grads = [vi_grad_L(α[c], L, β,
                           Lᵀ∇logpx_ϵ_half[c], Lᵀ∇logpy_half[c],
                           ∇logpx[c], ∇logpy[c]) for c = 1:length(α)]

        # Average the subgradients to reduce variance
        mean(cat(grads..., dims=3), dims=3)[:,:,1]
    end

    # Adam
    m0 = β1*m0 + (1 - β1)*∇L
    v0 = β2*v0 + (1 - β2)*(∇L.^2)
    αt = ρ_L*sqrt(1 - β2^it) / (1 - β1^it)
    L  = L + αt .* (m0 ./ (sqrt.(v0) .+ 1e-8))

    if(minimum(diag(L)) < 1e-3)
        L[diagind(L)] = max.(diag(L), 1e-3)
    end

    # update also the hyperparameter beta to match the desirable accept rate
    β = β + ρ_β*(mean(Int.(accept)) - opt)*β
    β = max.(β, 1e-4)

    L, β, m0, v0
end

function gadmala(logπ,
                 ∇logπ,
                 initial_q,
                 n_samples::Int,
                 n_burn::Int,
                 n_chains::Int;
                 inca::Bool=false,
                 prng=MersenneTwister(1),
                 β::Float64=1.0,
                 ρ_L::Float64=1.5e-4,
                 show_progress::Bool=true)
    if(inca)
        gadmala_inca(logπ,
                     ∇logπ,
                     initial_q,
                     n_samples,
                     n_burn,
                     n_chains;
                     prng=prng,
                     β=β,
                     ρ_L=ρ_L,
                     show_progress=show_progress)
    else
        prog   = ProgressMeter.Progress((n_burn + n_samples)*n_chains)
        chains = map(1:n_chains) do io
            gadmala_inca(logπ,
                         ∇logπ,
                         initial_q,
                         n_samples,
                         n_burn,
                         1;
                         prng=MersenneTwister(rand(prng, UInt64)),
                         β=β,
                         ρ_L=ρ_L,
                         prog=prog,
                         show_progress=show_progress)
        end
        cat(chains..., dims=3)
    end
end

function gadmala_inca(logπ,
                      ∇logπ,
                      initial_q,
                      n_samples::Int,
                      n_burn::Int,
                      n_chains::Int;
                      prng=MersenneTwister(1),
                      β::Float64=1.0,
                      ρ_L::Float64=1.5e-4,
                      show_progress::Bool=true,
                      prog=ProgressMeter.Progress(n_burn+n_samples))
    x0    = [rand(prng, initial_q) for i = 1:n_chains]
    dims  = length(x0[1])
    adapt = true
    x     = deepcopy(x0)
    y     = deepcopy(x0)
    n     = length(x0[1])
    L     = diagm(fill(0.1/sqrt(dims), dims))
    α     = zeros(n_chains)
    β     = fill(β,n)

    # collected samples 
    samples = zeros(n_samples,n,n_chains)

    opt = 0.55
    ρ_β = 0.02

    β1  = 0.9
    β2  = 0.999
    m0  = zeros(size(L))
    v0  = zeros(size(L))

    logpx           = logπ.(x)
    ∇logpx          = ∇logπ.(x)
    logpy           = zeros(n_chains)
    ∇logpy          = [Array{Float64}(undef, n) for c = 1:n_chains]
    Lᵀ∇logpx        = [L' * ∇logpx[c]           for c = 1:n_chains]
    Lᵀ∇logpy        = [Array{Float64}(undef, n) for c = 1:n_chains]
    Lᵀ∇logpy_half   = [Array{Float64}(undef, n) for c = 1:n_chains]
    Lᵀ∇logpx_ϵ_half = [Array{Float64}(undef, n) for c = 1:n_chains]

    accept    = falses(n_chains)
    total_acc = 0
    for it=1:n_samples+n_burn
        for c = 1:n_chains
            ϵ                  = randn(prng, n)
            Lᵀ∇logpx_ϵ_half[c] = Lᵀ∇logpx[c] / 2 + ϵ
            y[c]               = x[c] + L*Lᵀ∇logpx_ϵ_half[c];

            logpy[c]  = logπ(y[c])
            ∇logpy[c] = ∇logπ(y[c])

            # evaluate forward & backgward proposals q[y|x], q[x|y] 
            ϵ²               = ϵ.^2
            Lᵀ∇logpy[c]      = L'* ∇logpy[c]
            Lᵀ∇logpy_half[c] = Lᵀ∇logpy[c] / 2
            Δ∇logp           = Lᵀ∇logpx_ϵ_half[c] + Lᵀ∇logpy_half[c]
            logqyx           = sum(ϵ²) / -2
            logqxy           = dot(Δ∇logp, Δ∇logp) / -2

            α[c] = logpy[c] - logpx[c] + logqxy - logqyx
            
            # Decide to accept | reject  
            u         = rand(prng)
            accept[c] = log(u) < α[c]
        end

        # Adapt the proposal during burning
        if(it <= n_burn)
            L, β, m0, v0 = grad_adapt(it, L, β, ρ_β, ρ_L, accept, α, opt,
                                      β1, β2, m0, v0,
                                      Lᵀ∇logpx_ϵ_half, Lᵀ∇logpy_half,
                                      ∇logpx, ∇logpy)
        end

        if(show_progress)
            ProgressMeter.next!(
                prog; showvalues=[(:iteration,  it),
                                  (:acceptance, mean(accept)),
                                  (:avg_accept, total_acc / it / n_chains),
                                  (:likelihood, mean(logpy)),
                                  (:α,          mean(α))])
        end

        for c = 1:n_chains
            # Update Markov-chain state
            if(accept[c])
                x[c]        = y[c]
                logpx[c]    = logpy[c]
                ∇logpx[c]   = ∇logpy[c]
                Lᵀ∇logpx[c] = L'*∇logpx[c]
                total_acc  += 1
            else
                Lᵀ∇logpx[c] = L'*∇logpx[c];
            end

            if(it > n_burn)
                samples[it-n_burn,:,c] = x[c]
            end
        end
    end
    samples
end

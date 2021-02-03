
function run_gadmala(prng, ℓπ, dims, init_dist, n_samples, n_adapts, n_chains;
                     ρ_L=1e-4,
                     show_progress=true, inca=false)
    n_samples = n_samples*4
    n_adapts  = n_adapts*4
    
    ∇ℓπ(θ)  = Zygote.gradient(ℓπ, θ)[1]
    samples = gadmala(ℓπ,
                      ∇ℓπ,
                      init_dist,
                      n_samples,
                      n_adapts,
                      n_chains;
                      inca=inca,
                      prng=prng,
                      β=1.0,
                      ρ_L=ρ_L,
                      show_progress=show_progress)
    summarize(samples)
end

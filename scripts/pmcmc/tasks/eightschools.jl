
struct EightSchools end

Base.rand(prng::Random.AbstractRNG, ::EightSchools) = begin
    μ = rand(prng, Normal(0, 10))
    τ = rand(prng, truncated(Cauchy(0, 5), 1e-10, Inf))
    θ = rand(prng, MvNormal(fill(μ, 8), fill(τ, 8)))
    vcat(θ, τ, μ)
end

function eight_schools(sample)
#=
    Model from:
    Stan Development Team, Stan User’s Guide, Version 2.19
    
    Gelman, Andrew. 
    "Prior distributions for variance parameters in hierarchical models 
    (comment on article by Browne and Draper)." Bayesian analysis 1.3 (2006): 515-534.
    
    Data From:
    Gelman, Andrew, et al. Bayesian data analysis. CRC press, 2013.

    Origin:
    Rubin, Donald B.
    "Estimation in parallel randomized experiments." 
    Journal of Educational Statistics 6.4 (1981): 377-401.

=##
    y = Float64[28,  8, -3,  7, -1,  1, 18, 12]
    σ = Float64[15, 10, 16, 11,  9, 11, 10, 18]
    d_τ = truncated(Cauchy(0, 5), 1e-10, Inf)
    d_μ = Normal(0, 10)
    ℓπ = x->begin
        θ    = x[1:8] 
        τ    = x[9] 
        μ    = x[10] 

        d_y = MvNormal(θ, σ)
        d_θ = MvNormal(fill(μ, 8), fill(τ, 8))

        p_y = logpdf(d_y, y)
        p_θ = logpdf(d_θ, θ)
        p_μ = logpdf(d_μ, μ)
        p_τ = logpdf(d_τ, τ)
        
        p_y + p_θ + p_μ + p_τ
    end
    dims = 10
    sample("eightschools", ℓπ, dims, EightSchools())
end

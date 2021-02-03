
struct Gaussian
    dims::Int
    init_width::Real
end

Base.rand(prng::Random.AbstractRNG, gauss::Gaussian) = begin
    dims = gauss.dims
    σ    = gauss.init_width
    init_dist = MvNormal(dims, σ)
    rand(prng, init_dist)
end

function gaussian(sample, dims, init_width, correlated::Bool)
    # RNG for problem setup
    prng  = MersenneTwister(1)

    Σ = begin
        if(correlated)
            Σdist = Wishart(dims, diagm(range(1/dims, 1.0, length=dims)))
            rand(prng, Σdist)
        else
            diagm(range(1/dims, 1.0, length=dims))
        end
    end
    ℓπ(θ) = logpdf(MvNormal(ones(dims), Σ), θ)

    sample("gaussian_dim=$(dims)_init=$(init_width)_corr=$(correlated)",
           ℓπ, dims, Gaussian(dims, init_width))
end

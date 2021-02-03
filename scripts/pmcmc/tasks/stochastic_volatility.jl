
gammalogpdf_local(k::Real, θ::Real, x::Number) = -StatsFuns.loggamma(k) - k * log(θ) + (k - 1) * log(x) - x / θ

struct StochasticVolatility
    n_y::Int
end

Base.rand(prng::Random.AbstractRNG, stock::StochasticVolatility) = begin
    d_s1 = Exponential(0.1)
    d_τ  = Gamma(1/2, 2)
    n_y  = stock.n_y

    logs = Array{Float64}(undef, n_y)
    τ    = rand(prng, d_τ)

    d_s_vol = MvNormal(n_y - 1, 1/τ)

    logs[1] = exp(rand(prng, d_s1))
    s_vol   = rand(prng, d_s_vol)
    for t = 2:741
        logs[t] = (s_vol[t-1] - logs[t-1]) / 100
    end
    vcat(log(τ), logs)
end

function stochastic_volatility(sample)
    data   = CSV.File(datadir("dataset", "snp500_2000-2020.csv")) |> DataFrame
    data   = filter(r->r.Date .> DateTime(2017,1,1), data)
    data   = filter(r->r.Date .< DateTime(2019,12,12), data)
    data_y = data.Close
    n_y    = length(data_y)

    # Precomputed
    logyratio = log.(data_y[2:end] ./ data_y[1:end-1])

    d_s1 = Exponential(0.1)
    d_τ  = Gamma(1/2, 2)
    ℓπ = x->begin
        τ    = exp(x[1])
        logs = x[2:end]
        s    = exp.(logs)
        
        d_s = MvNormal(n_y - 1, 1/τ)
        d_y = MvNormal(zeros(n_y - 1), s[2:end])

        p_s1 = logpdf(d_s1, s[1])
        p_s  = logpdf(d_s,  100*(logs[2:end] - logs[1:end-1]))
        p_y  = logpdf(d_y,  logyratio)
        p_τ  = gammalogpdf_local(1/2, 2, τ)

        p_τ + p_s1 + p_s + p_y
    end
    sample("stochastic_volatility", ℓπ, n_y+1, StochasticVolatility(n_y))
end

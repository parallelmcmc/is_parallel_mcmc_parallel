
struct Logistic end

Base.rand(prng::Random.AbstractRNG, ::Logistic) = begin
    n_β    = div((24*24 - 24), 2) + 24

    d_logσ = Normal(0.0, 1.0)
    logσ   = rand(prng, d_logσ)
    σ      = exp(logσ)
    d_β    = MvNormal(n_β, σ)
    d_α    = Normal(0, σ)

    β = rand(prng, d_β)
    α = rand(prng, d_α)
    vcat(σ, α, β)
end

function logistic(sample)
    data   = readdlm(datadir("dataset", "german.data-numeric"))
    data_x = Array{Float64}(data[:, 1:end-1])
    data_y = (Array{Int64}(data[:, end]) .- 1.5)*2

    n_β    = div((24*24 - 24), 2) + 24
    n_vars = size(data_x, 2)

    X = Array{Float64}(undef, size(data_x, 1), n_β)
    X[:,1:n_vars] = data_x

    # two-way interactions
    idx = 1
    for i = 1:n_vars
        for j = i+1:n_vars
            @views X[:,n_vars+idx] = data_x[:,i] .* data_x[:,j] 
            idx += 1
        end
    end

    pos_y = data_y
    neg_y = 1 .- data_y

    d_logσ = Normal(0.0, 1.0)
    ℓπ = x->begin
        logσ = x[1]
        σ = exp(logσ)
        α = x[2]
        @views β = x[3:end]

        d_β = MvNormal(n_β, σ)
        d_α = Normal(0, σ)

        s    = X * β .+ α
        p_y  = sum(-log1pexp.(-data_y.*s))
        p_β  = logpdf(d_β, β)
        p_α  = logpdf(d_α, α)
        p_σ  = logpdf(d_logσ, logσ)
        p_y + p_β + p_α + p_σ
    end

    sample("logistic", ℓπ, n_β + 2, Logistic())
end

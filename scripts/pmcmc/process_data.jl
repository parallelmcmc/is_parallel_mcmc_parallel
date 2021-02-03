
using DataFrames
using DrWatson
@quickactivate "parallel_smc"

using JLD
using Distributions
using LinearAlgebra
using Statistics
using ProgressMeter
using DataFramesMeta
using KahanSummation
using Random123

function parse_method(name)
    if(name[8:14] == "gadmala")
        :gadmala
    elseif(name[8:16] == "nuts_inca")
        :nuts_inca
    elseif(name[8:11] == "nuts")
        :nuts
    elseif(name[8:16] == "gmh_hmcda")
        :gmh_hmcda
    end
end

function find_n_pos(entry, name)
    first_digit    = (findfirst(entry, name)                  |> last) + 2
    last_digit_off = (findfirst('_',   name[first_digit:end]) |> last) - 2
    name[first_digit:first_digit+last_digit_off]
end

function parse_problem(name)
    first_digit    = (findfirst("problem", name)                  |> last) + 2
    last_digit_off = (findfirst('.',       name[first_digit:end]) |> last) - 2
    problem_name   = name[first_digit:first_digit+last_digit_off]

    if(occursin("gaussian", problem_name))
        parsed = DrWatson.parse_savename(problem_name)
        dict   = parsed[2]
        res    = NamedTuple{Tuple(Symbol.(keys(dict)))}(values(dict))
        merge((problem="gaussian",), res)
    else
        (problem=problem_name,)
    end
end

function parse_data(name)
    method = parse_method(name)
    adapts = find_n_pos("n_adapts",  name)
    sample = find_n_pos("n_samples", name)
    chains = find_n_pos("n_chain", name)
    n_adapts  = parse(Int, adapts)
    n_samples = parse(Int, sample)
    n_chains  = parse(Int, chains)
    problem   = parse_problem(name)
    parsed    = (method=method,
                 n_adapts=n_adapts,
                 n_samples=n_samples,
                 n_chains=n_chains,
                 path=name)
    merge(problem, parsed)
end

function compute_mse(data, μ)
    ϵ² = map(data[:est_mean]) do batch
        Σx  = cumsum_kbn(batch, 1)
        Ex  = mapslices(x-> x ./ collect(1:length(x)), Σx, dims=1)
        ϵ²  = mapslices(x-> (x - μ).^2, Ex, dims=2)
    end
    mean(ϵ²)
end

function estimate_τ(data, μ, N, n_chains, σ²)
    n_boot  = 1024
    n_data  = length(data[:est_mean])
    n_dims  = size(data[:est_mean][1], 2)
    prng    = Random123.Philox4x()
    dist    = DiscreteUniform(1, n_data)

    τs = @showprogress map(1:n_boot) do var_idx
        sampling_idx = rand(prng, dist, n_data)
        batches      = view(data[:est_mean], sampling_idx)
        n_iter       = div(N, n_chains)
        ϵ²s = map(batches) do batch
            batch_clipped = view(batch, 1:n_iter, 1:size(batch, 2))
            Ex  = mapslices(sum_kbn, batch_clipped; dims=1)[1,:] / n_iter
            ϵ²  = (Ex - μ).^2
        end
        mse     = mean(ϵ²s)
        Neffinv = mse ./ σ²
        τ       = Neffinv .* N
    end
    τs = hcat(τs...)
    τs
end

function estimate_S(P, N, τ_seq, τ_par)
    S_samples = P * τ_seq ./ τ_par
    S_stats   = mapslices(S_samples, dims=2) do Si_samples
        μ    = mean(Si_samples)
        q_80 = quantile(Si_samples, [0.1,   0.9])
        q_95 = quantile(Si_samples, [0.025, 0.975])
        [μ,  2*μ - q_80[1], 2*μ - q_80[2], 2*μ - q_95[1], 2*μ - q_95[2],]
    end

    iter_samples = N ./ S_samples
    iter_stats   = mapslices(iter_samples, dims=2) do iteri_samples
        μ    = mean(iteri_samples)
        q_80 = quantile(iteri_samples, [0.1,   0.9])
        q_95 = quantile(iteri_samples, [0.025, 0.975])
        [μ,  2*μ - q_80[1], 2*μ - q_80[2], 2*μ - q_95[1], 2*μ - q_95[2],]
    end
    iter_avg = mean(iter_samples, dims=2)[:,1]
    res = DataFrame(idx=1:size(S_stats,1),
                    S_mean=S_stats[:,1],
                    S_p80_l=S_stats[:,2],
                    S_p80_h=S_stats[:,3],
                    S_p95_l=S_stats[:,4],
                    S_p95_h=S_stats[:,5],

                    iter_mean=iter_stats[:,1],
                    iter_p80_l=iter_stats[:,2],
                    iter_p80_h=iter_stats[:,3],
                    iter_p95_l=iter_stats[:,4],
                    iter_p95_h=iter_stats[:,5],
                    )
    println(res)
    res
end

function fetch_data_gaussian(df, method, init, dims, n_adapts)
    df   = @linq df |>
        where(:init     .== init) |>
        where(:method   .== method) |>
        where(:n_adapts .== n_adapts)  |>
        where(:dim      .== dims) |>
        select(:n_chains, :path)
    n_chains = df[1,:n_chains]
    data     = JLD.load(datadir("pmcmc", "exp_raw", df[1,:path]), "data")
    data, n_chains
end

function fetch_data_realistic(df, method, n_adapts)
    df   = @linq df |>
        where(:method   .== method) |>
        where(:n_adapts .== n_adapts)  |>
        select(:n_chains, :path)
    n_chains = df[1,:n_chains]
    data     = JLD.load(datadir("pmcmc", "exp_raw", df[1,:path]), "data")
    data, n_chains
end

function exp_gaussian(path)
    files  = readdir(path)
    files  = filter(x->occursin("gaussian", x), files)
    parsed = parse_data.(files)   
    df     = DataFrame(parsed)

    path    = datadir("pmcmc", "exp_pro", "gaussian")
    dims    = 25
    var_idx = 25

    println(df)

    for setup ∈ [(dims=25, init=2, method=:nuts),
                 (dims=25, init=4, method=:nuts),
                 (dims=25, init=8, method=:nuts),
                 (dims=25, init=2, method=:nuts_inca),
                 (dims=25, init=4, method=:nuts_inca),
                 (dims=25, init=8, method=:nuts_inca),]
        dims   = setup.dims
        init   = setup.init
        method = setup.method

        σ²   = range(1/dims, 1.0, length=dims)
        μ    = fill(1.0, dims)

        n_adapts = 4096
        N        = 4096

        data, n_chains = fetch_data_gaussian(df, method, init, dims, N)
        mse_seq = compute_mse(data, μ)
        τ_seq   = estimate_τ(data, μ, N, n_chains, σ²)

        settings = Dict(:nadapts=>n_adapts,
                        :dims=>dims,
                        :init=>init,
                        :varidx=>var_idx,
                        :method=>method,
                        )

        wsave(joinpath(path, "mse", savename(settings) * ".csv"),
              DataFrame(iter=1:size(mse_seq,1), mse=mse_seq[:,var_idx]))

        @showprogress for n_adapts ∈ [32, 64, 128, 512]
            data, n_chains = fetch_data_gaussian(df, method, init, dims, n_adapts)
            mse_par = compute_mse(data, μ)
            τ_par   = estimate_τ(data, μ, N, n_chains, σ²)
            P       = n_chains
            S_est   = estimate_S(P, N, τ_seq, τ_par)

            settings = Dict(:nadapts=>n_adapts,
                            :dims=>dims,
                            :init=>init,
                            :varidx=>var_idx,
                            :method=>method
                            )
            fname = savename(settings) * ".csv"
            wsave(joinpath(path, "mse", fname),
                  DataFrame(iter=1:size(mse_par,1), mse=mse_par[:,var_idx]))
            wsave(joinpath(path, "S",   fname), S_est)
        end
    end
end

function exp_gaussian_precond(path)
    files  = readdir(path)
    files  = filter(x->occursin("gaussian", x), files)
    parsed = parse_data.(files)   
    df     = DataFrame(parsed)
    path   = datadir("pmcmc", "exp_pro", "gaussian")

    dims = 25
    init = 2
    for setup ∈ [(dims=25, init=2, method=:nuts),
                 (dims=25, init=4, method=:nuts),
                 (dims=25, init=8, method=:nuts),
                 (dims=25, init=2, method=:nuts_inca),
                 (dims=25, init=4, method=:nuts_inca),
                 (dims=25, init=8, method=:nuts_inca)]
        dims   = setup.dims
        init   = setup.init
        method = setup.method

        n_adapts = [32, 64, 128, 512, 4096]
        θ_stats  = @showprogress map(n_adapts) do n_adapt
            σ² = range(1/dims, 1.0, length=dims)
            data, n_chains = fetch_data_gaussian(df, method, init, dims, n_adapt)

            precond_data = if(method == :nuts)
                vcat(data[:preconditioner]...)
            else
                data[:preconditioner]
            end
            θs = map(precond_data) do precond
                cosθ = dot(precond, σ²) / norm(precond) / norm(σ²)
                θ    = acos(cosθ) * 180 / π
            end
            θ_stats = quantile(θs, [0.05, 0.25, 0.5, 0.75, 0.95])
        end
        θ_stats = hcat(θ_stats...)

        settings = Dict(:dims=>dims,
                        :init=>init,
                        :method=>String(method))
        fname = savename(settings) * ".csv"
        precond_df = DataFrame(n_adapts=n_adapts,
                               θ_med=θ_stats[3,:],
                               θ_p90_l=θ_stats[1,  :],
                               θ_p90_h=θ_stats[end,:],
                               θ_p50_l=θ_stats[2,    :],
                               θ_p50_h=θ_stats[end-1,:])
        wsave(joinpath(path, "precond", fname), precond_df)
    end
end

function exp_gaussian_acceptance(path)
    files  = readdir(path)
    files  = filter(x->occursin("gaussian", x), files)
    parsed = parse_data.(files)   
    df     = DataFrame(parsed)
    path   = datadir("pmcmc", "exp_pro", "gaussian")

    dims = 25
    init = 2
    for setup ∈ [(dims=25, init=2, method=:nuts),
                 (dims=25, init=4, method=:nuts),
                 (dims=25, init=8, method=:nuts),
                 (dims=25, init=2, method=:nuts_inca),
                 (dims=25, init=4, method=:nuts_inca),
                 (dims=25, init=8, method=:nuts_inca)]
        dims   = setup.dims
        init   = setup.init
        method = setup.method

        n_adapts = [32, 64, 128, 512, 4096]
        stats  = @showprogress map(n_adapts) do n_adapt
            data, n_chains = fetch_data_gaussian(df, method, init, dims, n_adapt)
            αs = reshape(vcat(data[:acceptance_rate]...), :)
            α_stats = quantile(αs, [0.05, 0.25, 0.75, 0.95])
            α_stats[3] = mean(αs)
            α_stats
        end
        stats = hcat(stats...)

        settings = Dict(:dims=>dims,
                        :init=>init,
                        :method=>String(method))
        fname = savename(settings) * ".csv"
        precond_df = DataFrame(n_adapts=n_adapts,
                               alpha_avg=stats[3,:],
                               alpha_p90_l=stats[1,  :],
                               alpha_p90_h=stats[end,:],
                               alpha_p50_l=stats[2,    :],
                               alpha_p50_h=stats[end-1,:])
        wsave(joinpath(path, "acceptance", fname), precond_df)
    end
end

function compute_error_stats()
end

function exp_realistic(path, raw_name, ref_name)
    files  = readdir(path)
    files  = filter(x->occursin(raw_name, x), files)
    parsed = parse_data.(files)   
    df     = DataFrame(parsed)
    path   = datadir("pmcmc", "exp_pro", ref_name)

    println(df)

    reference = JLD.load(datadir("pmcmc", "reference", ref_name*".jld"), "samples")
    μ  = mean(reference, dims=1)[1,:,1]
    σ² = var(reference, dims=1)[1,:,1]

    method         = :nuts_inca

    for method ∈ [:nuts, :nuts_inca, :gmh_hmcda]
        n_adapts       = 2048
        data, n_chains = fetch_data_realistic(df, method, n_adapts)
        mse_seq        = compute_mse(data, μ)[n_adapts,:]

        adapt_list = [2048, 256, 128, 64, 32, 16]
        Ss = @showprogress map(adapt_list) do n_adapts
            data, n_chains = fetch_data_realistic(df, method, n_adapts)
            mse_par        = compute_mse(data, μ)[n_adapts,:]
            S              = n_chains * mse_seq ./ mse_par
			GC.gc()
	        S
        end
        Ss = Array(hcat(Ss...)')
        settings = Dict(:method=>method,)
        fname = savename(settings) * ".csv"
        wsave(joinpath(path, "S", fname),
              DataFrame(hcat(div.(2048, adapt_list), Ss)))
    end
end

function main(task::Symbol)
    exp_raw_path = datadir("pmcmc", "exp_raw")
    if(task == :exp_gaussian || task == :all)
        exp_gaussian(exp_raw_path)
    end
    if(task == :exp_gaussian_precond || task == :all)
        exp_gaussian_precond(exp_raw_path)
    end
    if(task == :exp_gaussian_acceptance || task == :all)
        exp_gaussian_acceptance(exp_raw_path)
    end
    if(task == :exp_logistic || task == :all)
        exp_realistic(exp_raw_path, "logistic", "logistic")
    end
    if(task == :exp_eightschools || task == :all)
        exp_realistic(exp_raw_path, "eightschools", "eight_schools")
    end

    # display(plot(log.(rmse_short[:,1])))
    # display(hline!([log(rmse_truth[1])]))
    # display(vline!([64]))
end

include("AGESS.jl")
using LinearAlgebra, LogExpFunctions, Distributions, LinearAlgebra, JLD2, Random, StatsBase, RCall, StatsPlots, LaTeXStrings
dir = "Users\\ndm34\\Projects\\AGESS_Simulation\\Volcano"

function sci_format(v::Real)
    v == 0 && return "0"
    s = @sprintf("%.1e", v)
    mantissa, expstr = split(s, 'e')
    e = parse(Int, expstr)
    mantissa = replace(mantissa, r"\.0$" => "")
    return string(mantissa, "e", e)
end


function log_posterior(X::AbstractVector{<:AbstractFloat})
    lpdf = norm(X)
    ## Prior distributions
    lpdf -= 0.5 * norm(X)^2

    return lpdf
end

function log_likelihood(X::AbstractVector{<:AbstractFloat})
    lpdf = norm(X)
    return lpdf
end

function log_likelihood_alpha(X::AbstractVector{<:AbstractFloat})
    lpdf = norm(X)
    lpdf -= 0.5 * norm(X)^2
    lpdf += (0.5 / 2) * norm(X)^2
    return lpdf
end

function log_likelihood_opt(X::AbstractVector{<:AbstractFloat})
    lpdf = norm(X)
    lpdf -= 0.5 * norm(X)^2
    lpdf += (0.5 / (1 + 1/length(X))) * norm(X)^2
    return lpdf
end

function log_prior(X::AbstractVector{<:AbstractFloat})
    ## Prior distributions
    lpdf = -0.5 * norm(X)^2

    return lpdf
end

function ESS_fx(MCMC::AbstractMatrix{<:AbstractFloat},
                n_0::Integer, n_1::Integer, n_2::Integer)
    n_calcs = size(MCMC)[1] - n_1
    ESS_est = ones(length(1:n_2:n_calcs))
    f_x = zeros(n_1)
    for j in 1:(n_1)
        @views f_x[j] = norm(MCMC[j,:])
    end

    index = 1
    for i in 1:n_2:n_calcs
        ESS_est[index] += 2 * sum(autocor(f_x, 1:n_0))
        f_x[1:(n_1 - n_2)] .= f_x[(n_2 + 1):(n_1)]
        if (n_1 + i + n_2) < size(MCMC)[1]
            for j in 1:n_2
                @views f_x[n_1 - n_2 + j] = norm(MCMC[n_1 + i + j,:])
            end
        end
        index += 1
    end

    return 1 ./ ESS_est
end

function ESS_fx(MCMC::AbstractMatrix{<:AbstractFloat},
                n_0::Integer)
    f_x = zeros(size(MCMC)[1])
    for j in 1:size(MCMC)[1]
        @views f_x[j] = norm(MCMC[j,:])
    end

    ESS_est = 1 + 2 * sum(autocor(f_x, 1:n_0))

    return 1 / ESS_est
end


function AGESS_volcano(x::AbstractMatrix{Y}, log_posterior::Function,
                       μ::AbstractVector{Y}, Σ::AbstractMatrix{Y}; ν::Y = 6.0, burnin::Y = 0.5, ϵ::Y = 0.1,
                       single_step_prop::Y = 0.05, β::Y = 0.5) where {Y<:AbstractFloat}
    t_dist = false
    P = size(x)[2]
    n_MCMC = size(x)[1]
    z = zeros(P)
    burnin_num = floor(Int64, burnin * n_MCMC)
    t1 = time()

    μ_adapt = copy(μ)
    μ_adapt_ph = copy(μ)
    ph = similar(μ_adapt)

    Σ_chol = cholesky(Σ)
    Σ_chol_adapt = deepcopy(Σ_chol.L)
    Σ_chol_adapt_ph = deepcopy(Σ_chol.L)

    μ_0 = zeros(P)
    ph_cholesky_update = ones(P)
    w_const = max(2/3, ((cbrt(P) - 1) / cbrt(P)))
    N_J = 2
    n_j = 2

    @views current_posterior = log_posterior(x[1,:])
    total_num_likevals = 1
    num_lik_iter = 0

    for i in 2:n_MCMC
        if i == burnin_num
            t1 = time()
            total_num_likevals = 0
        end

        if P >= 10
            if i < burnin_num * single_step_prop
                current_posterior, num_lik_iter = AGESS_SingleStep_1d(x, log_posterior, t_dist, ν, μ_adapt,
                                                                      Σ_chol_adapt, current_posterior, i)
                total_num_likevals += num_lik_iter
            else
                if rand() > ϵ
                    current_posterior, num_lik_iter = AGESS_SingleStep(x, z, log_posterior, ph, t_dist, ν, μ_adapt,
                                                                       Σ_chol_adapt, current_posterior, i)
                    total_num_likevals += num_lik_iter
                elseif rand() > 0.5
                    current_posterior, num_lik_iter = AGESS_SingleStep_1d(x, log_posterior, t_dist, ν, μ_adapt,
                                                                          Σ_chol_adapt, current_posterior, i)
                    total_num_likevals += num_lik_iter
                else
                    current_posterior, num_lik_iter = AGESS_SingleStep(x, z, log_posterior, ph, t_dist, ν, μ_0,
                                                                       Σ_chol.L, current_posterior, i)
                    total_num_likevals += num_lik_iter
                end
            end
        else
            if rand() > ϵ
                current_posterior, num_lik_iter = AGESS_SingleStep(x, z, log_posterior, ph, t_dist, ν, μ_adapt,
                                                                   Σ_chol_adapt, current_posterior, i)
                total_num_likevals += num_lik_iter
            else
                current_posterior, num_lik_iter = AGESS_SingleStep(x, z, log_posterior, ph, t_dist, ν, μ_0,
                                                                   Σ_chol.L, current_posterior, i)
                total_num_likevals += num_lik_iter
            end

        end

        w_i = i^(-w_const)
        @views Σ_chol_adapt_ph[diagind(Σ_chol_adapt_ph)] .= sqrt.((1 - w_i) *  Σ_chol_adapt_ph[diagind(Σ_chol_adapt_ph)].^2 .+ ( w_i * (norm(x[i,:] .- μ_adapt_ph)^2/ P)))
        @views μ_adapt_ph .= (1 - w_i) * μ_adapt_ph +  w_i * x[i,:]

        ## Adapt mean and covariance
        if i == N_J
            Σ_chol_adapt .= Σ_chol_adapt_ph
            μ_adapt .= μ_adapt_ph
            n_j += 1
            N_J += floor(n_j^β)
        end

        ## Populate next value in Markov Chain
        if i < n_MCMC
            @views x[i+1,:] .= x[i,:]
        end

        # Update User
        if P >= 10
            if i < burnin_num * single_step_prop
                if (i % 25) == 0
                    println("MCMC iter: ", i)
                    log_lik = @sprintf("%.2f", current_posterior)
                    println("Log Posterior: ", log_lik)
                end
            else
                if (i % 1000) == 0
                    println("MCMC iter: ", i)
                    log_lik = @sprintf("%.2f", current_posterior)
                    println("Log Posterior: ", log_lik)
                end
            end
        else
            if (i % 1000) == 0
                println("MCMC iter: ", i)
                log_lik = @sprintf("%.2f", current_posterior)
                println("Log Posterior: ", log_lik)
            end
        end

    end

    return time() - t1, Σ_chol_adapt * Σ_chol_adapt', μ_adapt, total_num_likevals
end

# Runs a single chain and evaluate ESS metrics
function run_and_ess(x_size::NTuple{2,Int}, sampler!::Function, analyses::Vector{<:Function})
    x_samp = zeros(x_size)
    sampler!(x_samp)
    results = [f(x_samp) for f in analyses]
    x_samp = nothing
    GC.gc()
    return results
end


Random.seed!(1234)
ess_ESS_alpha_total = zeros(3)
ess_AGESS_total = zeros(3)
ess_AGESS_norm_total = zeros(3)
ess_ESS_total = zeros(3)
ess_ESS_opt_total = zeros(3)
ess_ARW_total = zeros(3)

### D = 10
D = 10
n_0 = 100*D
total_range = 6000*D:10000*D

Σ = diagm(ones(D)* 2)
μ_j = zeros(D)
ess_ESS_alpha_total[1] = run_and_ess((10000*D, D), x -> ESS(x, log_likelihood_alpha, μ_j, Σ, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]
ess_AGESS_total[1] = run_and_ess((10000*D, D), x -> AGESS(x, log_posterior, μ_j, Σ, true, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]
ess_AGESS_norm_total[1] = run_and_ess((10000*D, D), x -> AGESS_volcano(x, log_posterior, μ_j, Σ, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]

Σ = diagm(ones(D))
ess_ESS_total[1] = run_and_ess((10000*D, D), x -> ESS(x, log_likelihood, μ_j, Σ, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]

Σ = diagm(ones(D) * (1 + 1/sqrt(D)))
ess_ESS_opt_total[1] = run_and_ess((10000*D, D), x -> ESS(x, log_likelihood_opt, μ_j, Σ, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]
ess_ARW_total[1] = run_and_ess((10000*D, D), x -> ARW(x, log_likelihood, log_prior, 10000, 0.01, μ_j, Σ, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]


## D = 100
D = 100
n_0 = 100*D
total_range = 6000*D:10000*D

Σ = diagm(ones(D)* 2)
μ_j = zeros(D)
ess_ESS_alpha_total[2] = run_and_ess((10000*D, D), x -> ESS(x, log_likelihood_alpha, μ_j, Σ, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]
ess_AGESS_total[2] = run_and_ess((10000*D, D), x -> AGESS(x, log_posterior, μ_j, Σ, true, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]
ess_AGESS_norm_total[2] = run_and_ess((10000*D, D), x -> AGESS_volcano(x, log_posterior, μ_j, Σ, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]

Σ = diagm(ones(D))
ess_ESS_total[2] = run_and_ess((10000*D, D), x -> ESS(x, log_likelihood, μ_j, Σ, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]

Σ = diagm(ones(D) * (1 + 1/sqrt(D)))
ess_ESS_opt_total[2] = run_and_ess((10000*D, D), x -> ESS(x, log_likelihood_opt, μ_j, Σ, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]
ess_ARW_total[2] = run_and_ess((10000*D, D), x -> ARW(x, log_likelihood, log_prior, 10000, 0.01, μ_j, Σ, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]


## D = 500
D = 500
n_0_total = 100*D
total_range = 3000*D:5000*D
n_0_iter, n_1_iter, n_2_iter = 5000, 500000, 500
iter_range = 20*D:5000*D
analyses() = [x -> ESS_fx(@view(x[total_range,:]), n_0_total),
              x -> ESS_fx(@view(x[iter_range,:]), n_0_iter, n_1_iter, n_2_iter)]

Σ = diagm(ones(D)* 2)
μ_j = zeros(D)
ess_ESS_alpha_total[3], ess_ESS_alpha = run_and_ess((5000*D, D), x -> ESS(x, log_likelihood_alpha, μ_j, Σ, burnin = 0.25), analyses())
ess_AGESS_total[3], ess_AGESS = run_and_ess((5000*D, D), x -> AGESS(x, log_posterior, μ_j, Σ, true, burnin = 0.25), analyses())
ess_AGESS_norm_total[3], ess_AGESS_norm = run_and_ess((5000*D, D), x -> AGESS_volcano(x, log_posterior, μ_j, Σ, burnin = 0.25), analyses())

Σ = diagm(ones(D))
ess_ESS_total[3], ess_ESS = run_and_ess((5000*D, D), x -> ESS(x, log_likelihood, μ_j, Σ, burnin = 0.25), analyses())

Σ = diagm(ones(D) * (1 + 1/sqrt(D)))
ess_ESS_opt_total[3], ess_ESS_opt = run_and_ess((5000*D, D), x -> ESS(x, log_likelihood_opt, μ_j, Σ, burnin = 0.25), analyses())
ess_ARW_total[3], ess_ARW = run_and_ess((5000*D, D), x -> ARW(x, log_likelihood, log_prior, 10000, 0.01, μ_j, Σ, burnin = 0.25), analyses())

labels = ["ESS (Σ = 2I)" "AGESS (t)" "AGESS (Normal)" "ESS (Σ = I)" L"ESS (Σ = (1 + $P^{-1/2}$)I)" "ARW"]
colors = [:red :green :purple :blue :orange :black]
shapes = [:circle :rect :utriangle :diamond :star5 :xcross]

iters = collect(1:500:1_990_500)
ess_iter = [ess_ESS_alpha ess_AGESS ess_AGESS_norm ess_ESS ess_ESS_opt ess_ARW]

D_vec = [10, 100, 500]
ess_total = [ess_ESS_alpha_total ess_AGESS_total ess_AGESS_norm_total ess_ESS_total ess_ESS_opt_total ess_ARW_total]

p1 = plot(D_vec, ess_total, color = colors, shape = shapes, markersize = 8, markerstrokewidth = 0,
     linewidth = 2, yscale = :log10, yticks = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0], ylim = [0.00001, 1.1],
     legend = false, fontfamily = "Computer Modern", titlefontsize = 26, guidefontsize = 20,
     tickfontsize = 16, legendfontsize = 16, framestyle = :axes, grid = false, title = "Volcano Target")
ylabel!(L"Effective Sample Size per Iteration $\left(‖x‖^2\right)$")
xlabel!(p1, "Dimension of Target Distribution (P)")

p2 = plot(iters, ess_iter, color = colors, linewidth = 2,
     yscale = :log10, yticks = [0.0001, 0.001, 0.01, 0.1, 1.0], ylim = [0.00001, 1.1], xlim = [1, 2_000_000],
     legend = false, fontfamily = "Computer Modern", titlefontsize = 26, guidefontsize = 20,
     tickfontsize = 16, legendfontsize = 16, framestyle = :axes, grid = false, title = "P = 500", xformatter = sci_format)
ylabel!(L"Effective Sample Size per Iteration $\left(‖x‖^2\right)$")
xlabel!(p2, "MCMC Iteration")

p_legend = plot(fill(NaN, 1, 6), fill(NaN, 1, 6), label = labels, color = colors, shape = shapes,
     linewidth = 2, markersize = 8, markerstrokewidth = 0, grid = false,
     showaxis = false, ticks = false, legend = :bottom, legend_column = 3, legendfontsize = 16,
     fontfamily = "Computer Modern", framestyle = :none)

plot(p1, p2, p_legend, layout =  @layout([a b; c{0.12h}]), dpi = 300, margin = 12Plots.mm, bottom_margin = 4Plots.mm)
plot!(size = (1600, 750))
savefig(string(dir ,"\\ESS_combined.pdf"))

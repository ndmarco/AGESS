include("AGESS.jl")
using LinearAlgebra, LogExpFunctions, Distributions, LinearAlgebra, JLD2, Random, StatsBase, RCall, StatsPlots, LaTeXStrings
dir = ".\\Normal"


function sci_format(v::Real)
    v == 0 && return "0"
    s = @sprintf("%.1e", v)
    mantissa, expstr = split(s, 'e')
    e = parse(Int, expstr)
    mantissa = replace(mantissa, r"\.0$" => "")
    return string(mantissa, "e", e)
end

function log_posterior(X::AbstractVector{<:AbstractFloat})
    lpdf =  -0.5 * norm(X)^2
    return lpdf
end

function log_likelihood_opt(X::AbstractVector{<:AbstractFloat})
    lpdf = 0.0
    return lpdf
end

function log_likelihood_alpha(X::AbstractVector{<:AbstractFloat}, α::AbstractFloat)
    lpdf = -0.5  * norm(X)^2
    lpdf += (0.5 / (α + 1.0)) * norm(X)^2
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
                @views f_x[n_1 - n_2 + j] =  norm(MCMC[n_1 + i + j,:])
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
        @views f_x[j] =  norm(MCMC[j,:])
    end

    ESS_est = 1 + 2 * sum(autocor(f_x, 1:n_0))

    return 1 / ESS_est
end

# Runs a single chain, evaluates each of `analyses` against it, then drops the
# chain (which at D=500 is ~10GB) so it's GC-eligible before the next chain is
# allocated -- only one big MCMC matrix is ever resident at a time.
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
ess_ESS_alpha10_total = zeros(3)
ess_ARW_total = zeros(3)

### D = 10
D = 10
n_0 = 100*D
total_range = 6000*D:10000*D
μ_j = zeros(D)

Σ = diagm(ones(D)* 2)
ess_ESS_alpha_total[1] = run_and_ess((10000*D, D), x -> ESS(x, y -> log_likelihood_alpha(y, 1.0), μ_j, Σ, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]

Σ = diagm(ones(D)* 10)
ess_ESS_alpha10_total[1] = run_and_ess((10000*D, D), x -> ESS(x, y -> log_likelihood_alpha(y, 9.0), μ_j, Σ, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]
ess_AGESS_total[1] = run_and_ess((10000*D, D), x -> AGESS(x, log_posterior, μ_j, Σ, true, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]
ess_AGESS_norm_total[1] = run_and_ess((10000*D, D), x -> AGESS(x, log_posterior, μ_j, Σ, false, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]

Σ = diagm(ones(D))
ess_ESS_total[1] = run_and_ess((10000*D, D), x -> ESS(x, log_likelihood_opt, μ_j, Σ, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]
ess_ARW_total[1] = run_and_ess((10000*D, D), x -> ARW(x, log_likelihood_opt, log_prior, 10000, 0.01, μ_j, Σ, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]


## D = 100
D = 100
n_0 = 100*D
total_range = 6000*D:10000*D
μ_j = zeros(D)

Σ = diagm(ones(D)* 2)
ess_ESS_alpha_total[2] = run_and_ess((10000*D, D), x -> ESS(x, y -> log_likelihood_alpha(y, 1.0), μ_j, Σ, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]

Σ = diagm(ones(D)* 10)
ess_ESS_alpha10_total[2] = run_and_ess((10000*D, D), x -> ESS(x, y -> log_likelihood_alpha(y, 9.0), μ_j, Σ, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]
ess_AGESS_total[2] = run_and_ess((10000*D, D), x -> AGESS(x, log_posterior, μ_j, Σ, true, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]
ess_AGESS_norm_total[2] = run_and_ess((10000*D, D), x -> AGESS(x, log_posterior, μ_j, Σ, false, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]

Σ = diagm(ones(D))
ess_ESS_total[2] = run_and_ess((10000*D, D), x -> ESS(x, log_likelihood_opt, μ_j, Σ, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]
ess_ARW_total[2] = run_and_ess((10000*D, D), x -> ARW(x, log_likelihood_opt, log_prior, 10000, 0.01, μ_j, Σ, burnin = 0.25),
    [x -> ESS_fx(@view(x[total_range,:]), n_0)])[1]


## D = 500
D = 500
n_0_total = 100*D
total_range = 3000*D:5000*D
n_0_iter, n_1_iter, n_2_iter = 5000, 500000, 500
iter_range = 20*D:5000*D
μ_j = zeros(D)
analyses() = [x -> ESS_fx(@view(x[total_range,:]), n_0_total),
              x -> ESS_fx(@view(x[iter_range,:]), n_0_iter, n_1_iter, n_2_iter)]

Σ = diagm(ones(D)* 2)
ess_ESS_alpha_total[3], ess_ESS_alpha = run_and_ess((5000*D, D), x -> ESS(x, y -> log_likelihood_alpha(y, 1.0), μ_j, Σ, burnin = 0.25), analyses())

Σ = diagm(ones(D)* 10)
ess_ESS_alpha10_total[3], ess_ESS_alpha10 = run_and_ess((5000*D, D), x -> ESS(x, y -> log_likelihood_alpha(y, 9.0), μ_j, Σ, burnin = 0.25), analyses())
ess_AGESS_total[3], ess_AGESS = run_and_ess((5000*D, D), x -> AGESS(x, log_posterior, μ_j, Σ, true, burnin = 0.25), analyses())
ess_AGESS_norm_total[3], ess_AGESS_norm = run_and_ess((5000*D, D), x -> AGESS(x, log_posterior, μ_j, Σ, false, burnin = 0.25), analyses())

Σ = diagm(ones(D))
ess_ESS_total[3], ess_ESS = run_and_ess((5000*D, D), x -> ESS(x, log_likelihood_opt, μ_j, Σ, burnin = 0.25), analyses())
ess_ARW_total[3], ess_ARW = run_and_ess((5000*D, D), x -> ARW(x, log_likelihood_opt, log_prior, 10000, 0.01, μ_j, Σ, burnin = 0.25), analyses())

labels = ["ESS (α = 1)" "ESS (α = 9)" "AGESS (t)" "AGESS (Normal)" "ESS (α = 0)" "ARW"]
colors = [:red :blue :green :purple :orange :black]
shapes = [:circle :rect :utriangle :diamond :star5 :xcross]

iters = collect(1:500:1_990_500)
ess_iter = [ess_ESS_alpha ess_ESS_alpha10 ess_AGESS ess_AGESS_norm ess_ESS ess_ARW]

D_vec = [10, 100, 500]
ess_total = [ess_ESS_alpha_total ess_ESS_alpha10_total ess_AGESS_total ess_AGESS_norm_total ess_ESS_total ess_ARW_total]

p1 = plot(D_vec, ess_total, color = colors, shape = shapes, markersize = 8, markerstrokewidth = 0,
     linewidth = 2, yscale = :log10, yticks = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0], ylim = [0.00001, 1.1],
     legend = false, fontfamily = "Computer Modern", titlefontsize = 26, guidefontsize = 20,
     tickfontsize = 16, legendfontsize = 16, framestyle = :axes, grid = false, title = "Normal Target")
ylabel!(L"Effective Sample Size per Iteration $\left(‖x‖^2\right)$")
xlabel!(p1, "Dimension of Target Distribution (P)")

p2 = plot(iters, ess_iter, color = colors, linewidth = 2,
     yscale = :log10, yticks = [0.0001, 0.001, 0.01, 0.1, 1.0], ylim = [0.00001, 1.1], xlim = [1, 2_000_000],
     legend = false, fontfamily = "Computer Modern", titlefontsize = 26, guidefontsize = 20,
     tickfontsize = 16, legendfontsize = 16, framestyle = :axes, grid = false, title = "P = 500", xformatter = sci_format)
ylabel!(L"Effective Sample Size per Iteration $\left(‖x‖^2\right)$")
xlabel!(p2, "MCMC Iteration")

p_legend = scatter(fill(NaN, 1, 6), fill(NaN, 1, 6), label = labels, color = colors, shape = shapes,
     markersize = 8, markerstrokewidth = 0, grid = false, showaxis = false, ticks = false,
     legend = :bottom, legend_column = 3, legendfontsize = 16, fontfamily = "Computer Modern",
     framestyle = :none)

plot(p1, p2, p_legend, layout =  @layout([a b; c{0.12h}]), dpi = 300, margin = 12Plots.mm, bottom_margin = 4Plots.mm)
plot!(size = (1600, 750))
savefig(string(dir ,"//ESS_combined.pdf"))

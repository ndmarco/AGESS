include("AGESS.jl")
using LinearAlgebra, LogExpFunctions, Distributions, LinearAlgebra, JLD2, Random, StatsBase, RCall, StatsPlots, LaTeXStrings
dir = ".\\Normal"


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



Random.seed!(1234)
ess_ESS_alpha_total = zeros(3)
ess_AGESS_total = zeros(3)
ess_AGESS_norm_total = zeros(3)
ess_ESS_total = zeros(3)
ess_ESS_alpha10_total = zeros(3)
ess_ARW_total = zeros(3)
### D = 10
D = 10
Σ = diagm(ones(D)* 2)
μ_j = zeros(D)


x_samp_ESS_alpha = zeros(10000 * D, D)
t2 = time()
time_non_burnin_ESS = ESS(x_samp_ESS_alpha, x -> log_likelihood_alpha(x, 1.0), μ_j, Σ, burnin = 0.25)
total_time_ESS_alpha = time() - t2

Σ = diagm(ones(D)* 10)
μ_j = zeros(D)
x_samp_ESS_alpha10 = zeros(10000 * D, D)
t2 = time()
time_non_burnin_ESS = ESS(x_samp_ESS_alpha10, x -> log_likelihood_alpha(x, 9.0), μ_j, Σ, burnin = 0.25)
total_time_ESS_alpha2 = time() - t2


x_samp_AGESS = zeros(10000 * D, D)
t2 = time()
time_non_burnin_AGESS, Σ_adapt = AGESS(x_samp_AGESS, log_posterior, μ_j, Σ, true, burnin = 0.25)
total_time_AGESS = time() - t2

x_samp_AGESS_norm = zeros(10000 * D, D)
t2 = time()
time_non_burnin_AGESS_norm, Σ_adapt_norm = AGESS(x_samp_AGESS_norm, log_posterior, μ_j, Σ, false, burnin = 0.25)
total_time_AGESS_norm = time() - t2

Σ = diagm(ones(D))
μ_j = zeros(D)
x_samp_ESS = zeros(10000 * D, D)
t2 = time()
time_non_burnin_ESS = ESS(x_samp_ESS, log_likelihood_opt, μ_j, Σ, burnin = 0.25)
total_time_ESS = time() - t2


Σ = diagm(ones(D))

x_samp_ARW = zeros(10000 * D, D)
t2 = time()
time_non_burnin_ARW = ARW(x_samp_ARW,log_likelihood_opt, log_prior, 10000, 0.01, μ_j, Σ, burnin = 0.25)
total_time_ARW = time() - t2

n_0 = 100*D

ess_ESS_alpha_total[1] = ESS_fx(x_samp_ESS_alpha[6000*D:10000*D,:], n_0)
ess_ESS_alpha10_total[1] = ESS_fx(x_samp_ESS_alpha10[6000*D:10000*D,:], n_0)
ess_AGESS_total[1] = ESS_fx(x_samp_AGESS[6000*D:10000*D,:], n_0)
ess_AGESS_norm_total[1] = ESS_fx(x_samp_AGESS_norm[6000*D:10000*D,:], n_0)
ess_ESS_total[1] = ESS_fx(x_samp_ESS[6000*D:10000*D,:], n_0)
ess_ARW_total[1] = ESS_fx(x_samp_ARW[6000*D:10000*D,:], n_0)

## D= 100
D = 100
Σ = diagm(ones(D)* 2)
μ_j = zeros(D)


x_samp_ESS_alpha = zeros(10000 * D, D)
t2 = time()
time_non_burnin_ESS = ESS(x_samp_ESS_alpha,  x -> log_likelihood_alpha(x, 1.0), μ_j, Σ, burnin = 0.25)
total_time_ESS_alpha = time() - t2

Σ = diagm(ones(D)* 10)
μ_j = zeros(D)
x_samp_ESS_alpha10 = zeros(10000 * D, D)
t2 = time()
time_non_burnin_ESS = ESS(x_samp_ESS_alpha10, x -> log_likelihood_alpha(x, 9.0), μ_j, Σ, burnin = 0.25)
total_time_ESS_alpha2 = time() - t2

x_samp_AGESS = zeros(10000 * D, D)
t2 = time()
time_non_burnin_AGESS, Σ_adapt = AGESS(x_samp_AGESS, log_posterior, μ_j, Σ, true, burnin = 0.25)
total_time_AGESS = time() - t2

x_samp_AGESS_norm = zeros(10000 * D, D)
t2 = time()
time_non_burnin_AGESS_norm, Σ_adapt_norm = AGESS(x_samp_AGESS_norm, log_posterior, μ_j, Σ, false, burnin = 0.25)
total_time_AGESS_norm = time() - t2

Σ = diagm(ones(D))
μ_j = zeros(D)
x_samp_ESS = zeros(10000 * D, D)
t2 = time()
time_non_burnin_ESS = ESS(x_samp_ESS, log_likelihood_opt, μ_j, Σ, burnin = 0.25)
total_time_ESS = time() - t2

x_samp_ARW = zeros(10000 * D, D)
t2 = time()
time_non_burnin_ARW = ARW(x_samp_ARW,log_likelihood_opt, log_prior, 10000, 0.01, μ_j, Σ, burnin = 0.25)
total_time_ARW = time() - t2


n_0 = 100*D
ess_ESS_alpha_total[2] = ESS_fx(x_samp_ESS_alpha[6000*D:10000*D,:], n_0)
ess_ESS_alpha10_total[2] = ESS_fx(x_samp_ESS_alpha10[6000*D:10000*D,:], n_0)
ess_AGESS_total[2] = ESS_fx(x_samp_AGESS[6000*D:10000*D,:], n_0)
ess_AGESS_norm_total[2] = ESS_fx(x_samp_AGESS_norm[6000*D:10000*D,:], n_0)
ess_ESS_total[2] = ESS_fx(x_samp_ESS[6000*D:10000*D,:], n_0)
ess_ARW_total[2] = ESS_fx(x_samp_ARW[6000*D:10000*D,:], n_0)


###
D = 500
Σ = diagm(ones(D)* 2)
μ_j = zeros(D)


x_samp_ESS_alpha = zeros(5000 * D, D)
t2 = time()
time_non_burnin_ESS = ESS(x_samp_ESS_alpha, x -> log_likelihood_alpha(x, 1.0), μ_j, Σ, burnin = 0.25)
total_time_ESS_alpha = time() - t2


Σ = diagm(ones(D)* 10)
μ_j = zeros(D)
x_samp_ESS_alpha10 = zeros(5000 * D, D)
t2 = time()
time_non_burnin_ESS = ESS(x_samp_ESS_alpha10, x -> log_likelihood_alpha(x, 9.0), μ_j, Σ, burnin = 0.25)
total_time_ESS_alpha2 = time() - t2

x_samp_AGESS = zeros(5000 * D, D)
t2 = time()
time_non_burnin_AGESS, Σ_adapt = AGESS(x_samp_AGESS, log_posterior, μ_j, Σ, true, burnin = 0.25)
total_time_AGESS = time() - t2

x_samp_AGESS_norm = zeros(5000 * D, D)
t2 = time()
time_non_burnin_AGESS_norm, Σ_adapt_norm = AGESS(x_samp_AGESS_norm, log_posterior, μ_j, Σ, false, burnin = 0.25)
total_time_AGESS_norm = time() - t2

Σ = diagm(ones(D))
μ_j = zeros(D)
x_samp_ESS = zeros(5000 * D, D)
t2 = time()
time_non_burnin_ESS = ESS(x_samp_ESS, log_likelihood_opt, μ_j, Σ, burnin = 0.25)
total_time_ESS = time() - t2



x_samp_ARW = zeros(5000 * D, D)
t2 = time()
time_non_burnin_ARW = ARW(x_samp_ARW,log_likelihood_opt, log_prior, 10000, 0.01, μ_j, Σ, burnin = 0.25)
total_time_ARW = time() - t2

n_0 = 5000
n_1 = 500000
n_2 = 500
ess_ESS_alpha = ESS_fx(x_samp_ESS_alpha[20*D:5000*D,:], n_0, n_1, n_2)
ess_ESS_alpha10 = ESS_fx(x_samp_ESS_alpha10[20*D:5000*D,:], n_0, n_1, n_2)
ess_AGESS = ESS_fx(x_samp_AGESS[20*D:5000*D,:],n_0, n_1, n_2)
ess_AGESS_norm = ESS_fx(x_samp_AGESS_norm[20*D:5000*D,:], n_0, n_1, n_2)
ess_ESS = ESS_fx(x_samp_ESS[20*D:5000*D,:], n_0, n_1, n_2)
ess_ARW = ESS_fx(x_samp_ARW[20*D:5000*D,:], n_0, n_1, n_2)

labels = ["ESS (α = 1)" "ESS (α = 9)" "AGESS (t)" "AGESS (Normal)" "ESS (α = 0)" "ARW"]
colors = [:red :blue :green :purple :orange :black]
linestyles = [:solid :dash :dot :dashdot :dashdotdot :solid]
shapes = [:circle :rect :utriangle :diamond :star5 :xcross]

iters = collect(1:500:1_990_500)
ess_iter = [ess_ESS_alpha ess_ESS_alpha10 ess_AGESS ess_AGESS_norm ess_ESS ess_ARW]
plot(iters, ess_iter, label = labels, color = colors, linestyle = linestyles, linewidth = 2,
     yscale = :log10, yticks = [0.0001, 0.001, 0.01, 0.1, 1.0], ylim = [0.00001, 1.1], xlim = [1, 2_000_000],
     legend = :outerright, fontfamily = "Computer Modern", titlefontsize = 16, guidefontsize = 14,
     tickfontsize = 12, legendfontsize = 12, framestyle = :axes, grid = false)
ylabel!(L"Effective Sample Size per Iteration ($\|x\|^2$)")
xlabel!("MCMC Iteration")
plot!(size = (1600, 700), dpi = 300)
savefig(string(dir ,"//ESS_iteration_500.pdf"))


n_0 = 100*D
ess_ESS_alpha_total[3] = ESS_fx(x_samp_ESS_alpha[3000*D:5000*D,:], n_0)
ess_ESS_alpha10_total[3] = ESS_fx(x_samp_ESS_alpha10[3000*D:5000*D,:], n_0)
ess_AGESS_total[3] = ESS_fx(x_samp_AGESS[3000*D:5000*D,:], n_0)
ess_AGESS_norm_total[3] = ESS_fx(x_samp_AGESS_norm[3000*D:5000*D,:], n_0)
ess_ESS_total[3] = ESS_fx(x_samp_ESS[3000*D:5000*D,:], n_0)
ess_ARW_total[3] = ESS_fx(x_samp_ARW[3000*D:5000*D,:], n_0)

D_vec = [10, 100, 500]
ess_total = [ess_ESS_alpha_total ess_ESS_alpha10_total ess_AGESS_total ess_AGESS_norm_total ess_ESS_total ess_ARW_total]
plot(D_vec, ess_total, label = labels, color = colors, shape = shapes, markersize = 8, markerstrokewidth = 0,
     linewidth = 2, yscale = :log10, yticks = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0], ylim = [0.00001, 1.1],
     legend = :outerright, fontfamily = "Computer Modern", titlefontsize = 16, guidefontsize = 14,
     tickfontsize = 12, legendfontsize = 12, framestyle = :axes, grid = false)
ylabel!(L"Effective Sample Size per Iteration ($\|x\|^2$)")
xlabel!("Dimension of Target Distribution (P)")
plot!(size = (1600, 700), dpi = 300)

savefig(string(dir ,"//ESS.pdf"))

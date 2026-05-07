include("AGESS.jl")
using LinearAlgebra, LogExpFunctions, Distributions, LinearAlgebra, JLD2, Random, StatsBase, RCall, StatsPlots
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

ess_ESS_alpha_total[1] = ESS_fx(x_samp_ESS_alpha[5000*D:10000*D,:], n_0)
ess_ESS_alpha10_total[1] = ESS_fx(x_samp_ESS_alpha10[5000*D:10000*D,:], n_0)
ess_AGESS_total[1] = ESS_fx(x_samp_AGESS[5000*D:10000*D,:], n_0)
ess_AGESS_norm_total[1] = ESS_fx(x_samp_AGESS_norm[5000*D:10000*D,:], n_0)
ess_ESS_total[1] = ESS_fx(x_samp_ESS[5000*D:10000*D,:], n_0)
ess_ARW_total[1] = ESS_fx(x_samp_ARW[5000*D:10000*D,:], n_0)

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
ess_ESS_alpha_total[2] = ESS_fx(x_samp_ESS_alpha[2000*D:10000*D,:], n_0)
ess_ESS_alpha10_total[2] = ESS_fx(x_samp_ESS_alpha10[2000*D:10000*D,:], n_0)
ess_AGESS_total[2] = ESS_fx(x_samp_AGESS[2000*D:10000*D,:], n_0)
ess_AGESS_norm_total[2] = ESS_fx(x_samp_AGESS_norm[2000*D:10000*D,:], n_0)
ess_ESS_total[2] = ESS_fx(x_samp_ESS[2000*D:10000*D,:], n_0)
ess_ARW_total[2] = ESS_fx(x_samp_ARW[2000*D:10000*D,:], n_0)


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

iters = collect(1:500:1990500)
plot(iters, ess_ESS_alpha, yscale=:log10, yticks = [0.0001, 0.001, 0.01, 0.1, 1.0], ylim = [0.00001,1.1], label = "ESS (α = 1)", xlim = [1,2000000])
plot!(iters, ess_ESS_alpha10, label = "ESS (α = 9)")
plot!(iters, ess_AGESS, label = "AGESS")
plot!(iters, ess_AGESS_norm, label = "AGESS_norm")
plot!(iters, ess_ESS, label = "ESS (α = 0)")
plot!(iters, ess_ARW, label = "ARW")
ylabel!("Effective Sample Size per Iteration")
xlabel!("MCMC Iteration")
plot!(size = (750, 500))
savefig(string(dir ,"//ESS_iteration_500.pdf"))


n_0 = 100*D
ess_ESS_alpha_total[3] = ESS_fx(x_samp_ESS_alpha[3000*D:5000*D,:], n_0)
ess_ESS_alpha10_total[3] = ESS_fx(x_samp_ESS_alpha10[3000*D:5000*D,:], n_0)
ess_AGESS_total[3] = ESS_fx(x_samp_AGESS[3000*D:5000*D,:], n_0)
ess_AGESS_norm_total[3] = ESS_fx(x_samp_AGESS_norm[3000*D:5000*D,:], n_0)
ess_ESS_total[3] = ESS_fx(x_samp_ESS[3000*D:5000*D,:], n_0)
ess_ARW_total[3] = ESS_fx(x_samp_ARW[3000*D:5000*D,:], n_0)

D_vec = [10, 100, 500]
plot(D_vec, ess_ESS_alpha_total, yscale=:log10, yticks = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0], ylim = [0.00001,1.1], label = "ESS (α= 1)", legend =:top,  markershape=:circle, markersize = 6)
plot!(D_vec, ess_ESS_alpha10_total, label = "ESS (α = 9)", markershape=:circle, markersize = 6)
plot!(D_vec, ess_AGESS_total, label = "AGESS (T distribution)", markershape=:circle, markersize = 6)
plot!(D_vec, ess_AGESS_norm_total, label = "AGESS (Normal)", markershape=:circle, markersize = 6)
plot!(D_vec, ess_ESS_total, label = "ESS (α = 0)", markershape=:circle, markersize = 6)
plot!(D_vec, ess_ARW_total, label = "ARW", markershape=:circle, markersize = 6)
ylabel!("Effective Sample Size per Iteration")
xlabel!("Dimension of Target Distribution")
plot!(size = (750, 500))

savefig(string(dir ,"//ESS.pdf"))

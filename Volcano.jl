include("AGESS.jl")
using LinearAlgebra, LogExpFunctions, Distributions, LinearAlgebra, JLD2, Random, StatsBase, RCall, StatsPlots
dir = ".\\Volcano"

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

    for i in 2:n_MCMC
        if i == burnin_num
            t1 = time()
        end

        if P >= 10
            if i < burnin_num * single_step_prop
                AGESS_SingleStep_1d(x, log_posterior, t_dist, ν, μ_adapt, Σ_chol_adapt, i)
            else
                if rand() > ϵ
                    AGESS_SingleStep(x, z, log_posterior, ph, t_dist, ν, μ_adapt,
                                        Σ_chol_adapt, i)
                elseif rand() > 0.5
                    AGESS_SingleStep_1d(x, log_posterior, t_dist, ν, μ_adapt, Σ_chol_adapt, i)
                else
                    AGESS_SingleStep(x, z, log_posterior, ph, t_dist, ν, μ_0, Σ_chol.L, i)
                end
            end
        else
            if rand() > ϵ
                AGESS_SingleStep(x, z, log_posterior, ph, t_dist, ν, μ_adapt,
                                        Σ_chol_adapt, i)
            else
                AGESS_SingleStep(x, z, log_posterior, ph, t_dist, ν, μ_0, Σ_chol.L, i)
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
                    @views log_lik = @sprintf("%.2f", log_posterior(x[i,:]))
                    println("Log Posterior: ", log_lik)
                end
            else
                if (i % 1000) == 0
                    println("MCMC iter: ", i)
                    @views log_lik = @sprintf("%.2f", log_posterior(x[i,:]))
                    println("Log Posterior: ", log_lik)
                end
            end
        else
            if (i % 1000) == 0
                println("MCMC iter: ", i)
                @views log_lik = @sprintf("%.2f", log_posterior(x[i,:]))
                println("Log Posterior: ", log_lik)
            end
        end
        
    end

    return time() - t1, Σ_chol_adapt * Σ_chol_adapt', μ_adapt
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
Σ = diagm(ones(D)* 2)
μ_j = zeros(D)


x_samp_ESS_alpha = zeros(10000 * D, D)
t2 = time()
time_non_burnin_ESS = ESS(x_samp_ESS_alpha, log_likelihood_alpha, μ_j, Σ, burnin = 0.25)
total_time_ESS_alpha = time() - t2

x_samp_AGESS = zeros(10000 * D, D)
t2 = time()
time_non_burnin_AGESS, Σ_adapt = AGESS(x_samp_AGESS, log_posterior, μ_j, Σ, true, burnin = 0.25)
total_time_AGESS = time() - t2

x_samp_AGESS_norm = zeros(10000 * D, D)
t2 = time()
time_non_burnin_AGESS_norm, Σ_adapt_norm = AGESS_volcano(x_samp_AGESS_norm, log_posterior, μ_j, Σ, burnin = 0.25)
total_time_AGESS_norm = time() - t2

Σ = diagm(ones(D))
μ_j = zeros(D)
x_samp_ESS = zeros(10000 * D, D)
t2 = time()
time_non_burnin_ESS = ESS(x_samp_ESS, log_likelihood, μ_j, Σ, burnin = 0.25)
total_time_ESS = time() - t2

Σ = diagm(ones(D) * (1 + 1/sqrt(D)))
μ_j = zeros(D)
x_samp_ESS_opt = zeros(10000 * D, D)
t2 = time()
time_non_burnin_ESS_opt = ESS(x_samp_ESS_opt, log_likelihood_opt, μ_j, Σ, burnin = 0.25)
total_time_ESS = time() - t2


x_samp_ARW = zeros(10000 * D, D)
t2 = time()
time_non_burnin_ARW = ARW(x_samp_ARW,log_likelihood, log_prior, 10000, 0.01, μ_j, Σ, burnin = 0.25)
total_time_ARW = time() - t2

n_0 = 100*D

ess_ESS_alpha_total[1] = ESS_fx(x_samp_ESS_alpha[6000*D:10000*D,:], n_0)
ess_AGESS_total[1] = ESS_fx(x_samp_AGESS[6000*D:10000*D,:], n_0)
ess_AGESS_norm_total[1] = ESS_fx(x_samp_AGESS_norm[6000*D:10000*D,:], n_0)
ess_ESS_total[1] = ESS_fx(x_samp_ESS[6000*D:10000*D,:], n_0)
ess_ESS_opt_total[1] = ESS_fx(x_samp_ESS_opt[6000*D:10000*D,:], n_0)
ess_ARW_total[1] = ESS_fx(x_samp_ARW[6000*D:10000*D,:], n_0)

## D= 100
D = 100
Σ = diagm(ones(D)* 2)
μ_j = zeros(D)


x_samp_ESS_alpha = zeros(10000 * D, D)
t2 = time()
time_non_burnin_ESS = ESS(x_samp_ESS_alpha, log_likelihood_alpha, μ_j, Σ, burnin = 0.25)
total_time_ESS_alpha = time() - t2

x_samp_AGESS = zeros(10000 * D, D)
t2 = time()
time_non_burnin_AGESS, Σ_adapt = AGESS(x_samp_AGESS, log_posterior, μ_j, Σ, true, burnin = 0.25)
total_time_AGESS = time() - t2

x_samp_AGESS_norm = zeros(10000 * D, D)
t2 = time()
time_non_burnin_AGESS_norm, Σ_adapt_norm = AGESS_volcano(x_samp_AGESS_norm, log_posterior, μ_j, Σ, burnin = 0.25)
total_time_AGESS_norm = time() - t2

Σ = diagm(ones(D))
μ_j = zeros(D)
x_samp_ESS = zeros(10000 * D, D)
t2 = time()
time_non_burnin_ESS = ESS(x_samp_ESS, log_likelihood, μ_j, Σ, burnin = 0.25)
total_time_ESS = time() - t2

Σ = diagm(ones(D) * (1 + 1/sqrt(D)))
μ_j = zeros(D)
x_samp_ESS_opt = zeros(10000 * D, D)
t2 = time()
time_non_burnin_ESS_opt = ESS(x_samp_ESS_opt, log_likelihood_opt, μ_j, Σ, burnin = 0.25)
total_time_ESS = time() - t2


x_samp_ARW = zeros(10000 * D, D)
t2 = time()
time_non_burnin_ARW = ARW(x_samp_ARW,log_likelihood, log_prior, 10000, 0.01, μ_j, Σ, burnin = 0.25)
total_time_ARW = time() - t2

n_0 = 100*D

ess_ESS_alpha_total[2] = ESS_fx(x_samp_ESS_alpha[6000*D:10000*D,:], n_0)
ess_AGESS_total[2] = ESS_fx(x_samp_AGESS[6000*D:10000*D,:], n_0)
ess_AGESS_norm_total[2] = ESS_fx(x_samp_AGESS_norm[6000*D:10000*D,:], n_0)
ess_ESS_total[2] = ESS_fx(x_samp_ESS[6000*D:10000*D,:], n_0)
ess_ESS_opt_total[2] = ESS_fx(x_samp_ESS_opt[6000*D:10000*D,:], n_0)
ess_ARW_total[2] = ESS_fx(x_samp_ARW[6000*D:10000*D,:], n_0)


## D = 500
D = 500
Σ = diagm(ones(D)* 2)
μ_j = zeros(D)


x_samp_ESS_alpha = zeros(5000 * D, D)
t2 = time()
time_non_burnin_ESS = ESS(x_samp_ESS_alpha, log_likelihood_alpha, μ_j, Σ, burnin = 0.25)
total_time_ESS_alpha = time() - t2

x_samp_AGESS = zeros(5000 * D, D)
t2 = time()
time_non_burnin_AGESS, Σ_adapt = AGESS(x_samp_AGESS, log_posterior, μ_j, Σ, true, burnin = 0.25)
total_time_AGESS = time() - t2

x_samp_AGESS_norm = zeros(5000 * D, D)
t2 = time()
time_non_burnin_AGESS_norm, Σ_adapt_norm = AGESS_volcano(x_samp_AGESS_norm, log_posterior, μ_j, Σ, burnin = 0.25)
total_time_AGESS_norm = time() - t2

Σ = diagm(ones(D))
μ_j = zeros(D)
x_samp_ESS = zeros(5000 * D, D)
t2 = time()
time_non_burnin_ESS = ESS(x_samp_ESS, log_likelihood, μ_j, Σ, burnin = 0.25)
total_time_ESS = time() - t2

Σ = diagm(ones(D) * (1 + 1/sqrt(D)))
μ_j = zeros(D)
x_samp_ESS_opt = zeros(5000 * D, D)
t2 = time()
time_non_burnin_ESS_opt = ESS(x_samp_ESS_opt, log_likelihood_opt, μ_j, Σ, burnin = 0.25)
total_time_ESS = time() - t2


x_samp_ARW = zeros(5000 * D, D)
t2 = time()
time_non_burnin_ARW = ARW(x_samp_ARW,log_likelihood, log_prior, 10000, 0.01, μ_j, Σ, burnin = 0.25)
total_time_ARW = time() - t2

n_0 = 5000
n_1 = 500000
n_2 = 500
ess_ESS_alpha = ESS_fx(x_samp_ESS_alpha[20*D:5000*D,:], n_0, n_1, n_2)
ess_AGESS = ESS_fx(x_samp_AGESS[20*D:5000*D,:], n_0, n_1, n_2)
ess_AGESS_norm = ESS_fx(x_samp_AGESS_norm[20*D:5000*D,:], n_0, n_1, n_2)
ess_ESS = ESS_fx(x_samp_ESS[20*D:5000*D,:], n_0, n_1, n_2)
ess_ESS_opt = ESS_fx(x_samp_ESS_opt[20*D:5000*D,:], n_0, n_1, n_2)
ess_ARW = ESS_fx(x_samp_ARW[20*D:5000*D,:], n_0, n_1, n_2)

iters = collect(1:500:1990500)
plot(iters, ess_ESS_alpha, yscale=:log10, yticks = [0.0001, 0.001, 0.01, 0.1, 1.0], ylim = [0.00001,1.1], label = "ESS α", xlim = [1,2000000])
plot!(iters, ess_AGESS, label = "AGESS")
plot!(iters, ess_AGESS_norm, label = "AGESS_norm")
plot!(iters, ess_ESS, label = "ESS")
plot!(iters, ess_ESS_opt, label = "ESS optimal")
plot!(iters, ess_ARW, label = "ARW")
ylabel!("Effective Sample Size per Iteration")
xlabel!("MCMC Iteration")
plot!(size = (750, 500))
savefig(string(dir ,"//ESS_iteration_500.pdf"))

n_0 = 100*D
ess_ESS_alpha_total[3] = ESS_fx(x_samp_ESS_alpha[3000*D:5000*D,:], n_0)
ess_AGESS_total[3] = ESS_fx(x_samp_AGESS[3000*D:5000*D,:], n_0)
ess_AGESS_norm_total[3] = ESS_fx(x_samp_AGESS_norm[3000*D:5000*D,:], n_0)
ess_ESS_total[3] = ESS_fx(x_samp_ESS[3000*D:5000*D,:], n_0)
ess_ESS_opt_total[3] = ESS_fx(x_samp_ESS_opt[3000*D:5000*D,:], n_0)
ess_ARW_total[3] = ESS_fx(x_samp_ARW[3000*D:5000*D,:], n_0)


D_vec = [10, 100, 500]
plot(D_vec, ess_ESS_alpha_total, yscale=:log10, yticks = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0], ylim = [0.00001,1.1], label = "ESS α", legend =:top, markershape=:circle, markersize = 6)
plot!(D_vec, ess_AGESS_total, label = "AGESS", markershape=:circle, markersize = 6)
plot!(D_vec, ess_AGESS_norm_total, label = "AGESS_norm", markershape=:circle, markersize = 6)
plot!(D_vec, ess_ESS_total, label = "ESS", markershape=:circle, markersize = 6)
plot!(D_vec, ess_ESS_opt_total, label = "ESS optimal", markershape=:circle, markersize = 6)
plot!(D_vec, ess_ARW_total, label = "ARW", markershape=:circle, markersize = 6)
ylabel!("Effective Sample Size per Iteration")
xlabel!("Dimension of Target Distribution")
plot!(size = (750, 500))
savefig(string(dir ,"//ESS.pdf"))

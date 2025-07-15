using KernelFunctions, LinearAlgebra, LogExpFunctions, Distributions, LinearAlgebra, JLD2, Random, StatsBase, RCall, StatsPlots
include("AGESS.jl")

function gen_data(N::T, min_eval::Y, max_eval::Y) where {Y<:AbstractFloat, T<:Integer}
    X = collect(LinRange(min_eval, max_eval, N))
    Y_N = sin.(X) .+ 2 * exp.(-30 * X.^2)
    return X, Y_N
end

function gen_data(X::AbstractVector{Y}) where {Y<:AbstractFloat}
    Y_N = sin.(X) .+ 2 * exp.(-30 * X.^2)
    return Y_N
end

function gen_data_Higdon(N::T,min_eval::Y, max_eval::Y) where {Y<:AbstractFloat, T<:Integer}
    X = collect(LinRange(min_eval, max_eval, N))
    Y_N = sin.((π .* X) ./ 5) + 0.2 * cos.(((4 * π) .* X) ./ 5)
    Y_N[X .> 10] .= (X[X .> 10.0]) ./ 10 .- 0.8
    return X, Y_N
end

function construct_Kernel_Mat!(Σ::AbstractMatrix{Y}, X::AbstractVector{Y}, θ::Y) where {Y<:AbstractFloat}
    for j in 1:size(Σ)[1],k in 1:size(Σ)[2]
        Σ[j,k] = exp(-(((X[j] - X[k])^2) / θ))
    end
    Σ .= Symmetric(Σ)
end

function construct_Kernel_Mat_y!(Σ::AbstractMatrix{Y}, X::AbstractVector{Y}, W::AbstractVector{Y}, θ_y_x::Y, θ_y_w::Y) where {Y<:AbstractFloat}
    for j in 1:size(Σ)[1],k in 1:size(Σ)[2]
        Σ[j,k] = -(((X[j] - X[k])^2) / θ_y_x)
        Σ[j,k] -= (((W[j] - W[k])^2) / θ_y_w)
    end
    Σ .= exp.(Σ)
    Σ .= Symmetric(Σ)
end

function likelihood_Y(W::AbstractVector{Y}, X::AbstractVector{Y}, Y_N::AbstractVector{Y}, g::Y, θ_y_x::Y, θ_y_w::Y, ph::AbstractVector{Y}, 
                      Σ::AbstractMatrix{Y}) where {Y<:AbstractFloat}
    construct_Kernel_Mat_y!(Σ, X, W, θ_y_x, θ_y_w)
    Σ[diagind(Σ)] .+=  g
    cholesky!(Σ)
    ph .= LowerTriangular(Σ) \ Y_N
    lpdf =  -sum(log.(diag(Σ)))  - (0.5 * (1 +length(ph)) * log1p(dot(ph, ph)))

    return lpdf
end

function likelihood_W(W::AbstractVector{Y}, X_N::AbstractVector{Y}, θ_w::Y, g::Y, ph::AbstractVector{Y}, Σ::AbstractMatrix{Y}) where {Y<:AbstractFloat}
    construct_Kernel_Mat!(Σ, X_N, θ_w)
    Σ[diagind(Σ)] .+= g
    cholesky!(Σ)

    ph .= LowerTriangular(Σ) \ W
    lpdf = -sum(log.(diag(Σ)))  - 0.5 * dot(ph, ph)

    return lpdf
end


function sample_aux_parameters(θ_y_x::AbstractVector{Y}, θ_y_w::AbstractVector{Y},
                               θ_w::AbstractVector{Y}, ph1::AbstractVector{Y}, 
                               ph2::AbstractVector{Y}, Σ1::AbstractMatrix{Y},
                               Σ2::AbstractMatrix{Y}, X_N::AbstractVector{Y}, 
                               W::AbstractVector{Y}, Y_N::AbstractVector{Y},
                               σ_θ_w::Y, σ_θ_y_x::Y, σ_θ_y_w::Y,
                               prior_θ::Function, g::Y, i::T, accept_vec::AbstractVector{Y}) where {Y<:AbstractFloat, T<:Integer}

    θ_y_x_prop = rand(LogNormal(log(θ_y_x[i]), σ_θ_y_x))
    accept_prob = likelihood_Y(W, X_N, Y_N, g, θ_y_x_prop, θ_y_w[i], ph1, Σ1) + prior_θ(θ_y_x_prop) - (likelihood_Y(W, X_N, Y_N, g, θ_y_x[i], θ_y_w[i], ph2, Σ2) + prior_θ(θ_y_x[i]))
    accept_prob += (logpdf(LogNormal(log(θ_y_x_prop), σ_θ_y_x), θ_y_x[i]) -logpdf(LogNormal(log(θ_y_x[i]), σ_θ_y_x), θ_y_x_prop))
    if !isnan(accept_prob)
        if isfinite(accept_prob)
            if log(rand()) < accept_prob
                θ_y_x[i] = θ_y_x_prop
                accept_vec[1] += 1
            end
        end
    end

    θ_y_w_prop = rand(LogNormal(log(θ_y_w[i]), σ_θ_y_w))
    accept_prob = likelihood_Y(W, X_N, Y_N, g, θ_y_x[i], θ_y_w_prop, ph1, Σ1) + prior_θ(θ_y_w_prop) - (likelihood_Y(W, X_N, Y_N, g, θ_y_x[i], θ_y_w[i], ph2, Σ2) + prior_θ(θ_y_w[i]))
    accept_prob += (logpdf(LogNormal(log(θ_y_w_prop), σ_θ_y_w), θ_y_w[i]) -logpdf(LogNormal(log(θ_y_w[i]), σ_θ_y_w), θ_y_w_prop))
    if !isnan(accept_prob)
        if isfinite(accept_prob)
            if log(rand()) < accept_prob
                θ_y_w[i] = θ_y_w_prop
                accept_vec[2] += 1
            end
        end
    end


    θ_w_prop = rand(LogNormal(log(θ_w[i]), σ_θ_w))
    accept_prob = likelihood_W(W, X_N, θ_w_prop, g, ph1, Σ1) + prior_θ(θ_w_prop) - (likelihood_W(W, X_N, θ_w[i], g, ph2, Σ2) + prior_θ(θ_w[i]))
    accept_prob += (logpdf(LogNormal(log(θ_w_prop), σ_θ_w), θ_w[i]) - logpdf(LogNormal(log(θ_w[i]), σ_θ_w), θ_w_prop))
    if !isnan(accept_prob)
        if isfinite(accept_prob)
            if log(rand()) < accept_prob
                θ_w[i] = θ_w_prop
                accept_vec[3] += 1
            end
        end
    end

    return nothing
end

function predictive_draws(time_points::AbstractVector{Y}, W::AbstractMatrix{Y}, θ_w::AbstractVector{Y},
                          θ_y_x::AbstractVector{Y}, θ_y_w::AbstractVector{Y}, g::Y, Y_N::AbstractVector{Y}, 
                          X_N::AbstractVector{Y}, burnin::Y=0.5) where {Y<:AbstractFloat}
    n_MCMC = size(W)[1]
    burnin_num = floor(Int64, burnin * n_MCMC)
    P_out = length(time_points)
    P = length(X_N)
    Y_out = zeros(n_MCMC - burnin_num, P_out)
    W_out = zeros(P_out)
    k_out_X_const = zeros(P_out, P)
    k_out_X = zeros(P_out, P)
    k_out_Y = zeros(P_out, P)
    Σ = zeros(P,P)
    ph = zeros(P_out, P)
    ph1 = zeros(P)

    μ_out = zeros(P_out)
    Σ_out = zeros(P_out, P_out) 

    for j in 1:P_out
        for k in 1:P
            k_out_X_const[j,k] = -(time_points[j] - X_N[k])^2
        end
    end

    for i in (burnin_num + 1):n_MCMC
        ## get distribution of W_out
        k_out_X .= exp.(k_out_X_const ./ θ_w[i])
        construct_Kernel_Mat!(Σ, X_N, θ_w[i])
        Σ[diagind(Σ)] .+=  g
        ph .= (Σ \ k_out_X')'
        μ_out .= ph * W[i,:]
        construct_Kernel_Mat!(Σ_out, time_points, θ_w[i])
        Σ_out .-= ph * k_out_X'
        Σ_out[diagind(Σ_out)] .+= g
        Σ_out .= Hermitian(Σ_out)
        cholesky!(Σ_out)

        ## Generate sample of W_out
        W_out .= LowerTriangular(Σ_out) * randn(P_out) 
        W_out .+= μ_out

        ## Generate sample of Y_out
        for j in 1:P_out
            for k in 1:P
                k_out_Y[j,k] = exp(-(((W_out[j] - W[i,k])^2 / θ_y_w[i]) + ((time_points[j] - X_N[k])^2 / θ_y_x[i])))
            end
        end


        construct_Kernel_Mat_y!(Σ, X_N, W[i,:], θ_y_x[i], θ_y_w[i])
        Σ[diagind(Σ)] .+=  g
        ph .= (Σ \ k_out_Y')'
        μ_out = ph * Y_N

        cholesky!(Σ)
        ph1 .= LowerTriangular(Σ) \ Y_N
        construct_Kernel_Mat_y!(Σ_out, time_points, W_out, θ_y_x[i], θ_y_w[i])
        Σ_out[diagind(Σ_out)] .+= g
        Σ_out .-= ph * k_out_Y'
        Σ_out .= Hermitian(Σ_out)
        Σ_out .*= ((1 + dot(ph1, ph1)) / (1 + P))
        Σ_out .*= rand(Gamma((P + 1) * 0.5, 2 / (P+1)))
        cholesky!(Σ_out)

        
        Y_out[i - burnin_num,:] .= LowerTriangular(Σ_out) * randn(P_out) 
        Y_out[i - burnin_num,:] .+= μ_out
    end

    return Y_out
end

function plot_CI(Y_N::AbstractVector{Y}, X_N::AbstractVector{Y}, Y_out::AbstractMatrix{Y},
                 time_points::AbstractVector{Y}, truth::AbstractVector{Y}) where {Y<:AbstractFloat}
    p = plot(X_N, Y_N, seriestype=:scatter, color = "red", label = "Observed Data")
    P_out = length(time_points)
    Upper_CI = zeros(P_out)
    Lower_CI = zeros(P_out)
    median_est = zeros(P_out)
    for i in 1:P_out
        median_est[i] = median(Y_out[:,i])
        Lower_CI[i] = quantile(Y_out[:,i], 0.025)
        Upper_CI[i] = quantile(Y_out[:,i], 0.975)
    end
    p = plot!(p, time_points, median_est, color = "blue", label ="Posterior Median")
    p = plot!(p, time_points, truth, color = "red", label ="Truth")
    p = plot!(p, time_points, Lower_CI, fillrange = Upper_CI, fillalpha = 0.3, alpha = 0.3, label = "95% CI")

    return p
end

function plot_Var(Y_out::AbstractMatrix{Y}, time_points::AbstractVector{Y}) where {Y<:AbstractFloat}
    P_out = length(time_points)
    sd = zeros(P_out)
    for i in 1:P_out
        sd[i] = sqrt(var(Y_out[:,i]))
    end
    p = plot(time_points, sd)

    return p
end

function sampler_ESS(Y_N::AbstractVector{Y}, X_N::AbstractVector{Y}, W::AbstractMatrix{Y}, θ_w::AbstractVector{Y}, 
                     θ_y_x::AbstractVector{Y}, θ_y_w::AbstractVector{Y}, g::Y, prior_θ::Function,
                     σ_θ_y_x::Y = 0.1, σ_θ_y_w::Y = 0.1, σ_θ_w::Y = 0.1, tuning_step::T = 25) where {Y<:AbstractFloat, T<:Integer}
    accept_vec = zeros(3)
    n_MCMC = size(W)[1]
    P = length(Y_N)

    Σ1 = diagm(ones(P))
    Σ2 = diagm(ones(P))

    ph = ones(P)
    ph1 = ones(P)
    ph2 = ones(P)

    for i in 2:n_MCMC
        ## Sample auxillary parameters
        @views sample_aux_parameters(θ_y_x, θ_y_w, θ_w, ph1, ph2, Σ1, Σ2, X_N, W[i,:], Y_N,
                                     σ_θ_w, σ_θ_y_x, σ_θ_y_w, prior_θ, g, i, accept_vec)

        
        ## Sample W
        construct_Kernel_Mat!(Σ2, X_N, θ_w[i])
        Σ2[diagind(Σ2)] .+= g
        cholesky!(Σ2) 
        ESS_SingleStep(W, ph, b -> likelihood_Y(b, X_N, Y_N, g, θ_y_x[i], θ_y_w[i], ph1, Σ1), zeros(P), LowerTriangular(Σ2), i)

        if (i % tuning_step) == 0
            println("MCMC iter: ", i)
            println("Acceptance θ_y_x: ", round(accept_vec[1] / tuning_step, digits=3))
            println("Acceptance θ_y_w: ", round(accept_vec[2] / tuning_step, digits=3))
            println("Acceptance θ_w: ", round(accept_vec[3] / tuning_step, digits=3))
            log_lik = @sprintf("%.2f", likelihood_Y(W[i,:], X_N, Y_N, g, θ_y_x[i], θ_y_w[i], ph1, Σ1))
            println("Log Likelihood: ", log_lik)
            σ_θ_w = exp(log(σ_θ_w) + ((accept_vec[3] / tuning_step) - 0.44) / i)
            σ_θ_y_x = exp(log(σ_θ_y_x) + ((accept_vec[1] / tuning_step) - 0.44) / i)
            σ_θ_y_w = exp(log(σ_θ_y_w) + ((accept_vec[2] / tuning_step) - 0.44) / i)
            accept_vec .= 0
        end

        ## Update next state
        if i < n_MCMC
            W[i+1,:] .= W[i,:]
            θ_w[i+1] = θ_w[i]
            θ_y_x[i+1] = θ_y_x[i]
            θ_y_w[i+1] = θ_y_w[i]
        end

    end
end

function sampler_GESS(Y_N::AbstractVector{Y}, X_N::AbstractVector{Y}, W::AbstractMatrix{Y}, θ_w::AbstractVector{Y}, 
                      θ_y_x::AbstractVector{Y}, θ_y_w::AbstractVector{Y}, g::Y, prior_θ::Function,
                      μ::AbstractVector{Y}, Σ::AbstractMatrix{Y}; ν::Y = 3.0) where {Y<:AbstractFloat}
    n_MCMC = size(W)[1]
    P = length(Y_N)

    Σ1 = diagm(ones(P))
    Σ2 = diagm(ones(P))

    ph = ones(P+3)
    ph1 = ones(P)
    ph2 = ones(P)
    ph3 = ones(P + 3)

    W_θ = zeros(n_MCMC, P + 3)
    W_θ[1, 1:P] .= W[1,:]
    W_θ[1, P+1] = θ_y_x[1]
    W_θ[1, P+2] = θ_y_w[1]
    W_θ[1, P+3] = θ_w[1]
    Σ_chol = cholesky(Σ)

    for i in 2:n_MCMC
        W_θ[i, 1:P] .= W[i,:]
        W_θ[i, P+1] = θ_y_x[i]
        W_θ[i, P+2] = θ_y_w[i]
        W_θ[i, P+3] = θ_w[i]
        α = 0.5 * (ν + P + 3)
        ph .= Σ_chol.L \ (W_θ[i,:] - μ)
        β = 0.5 * (ν + dot(ph,ph))
        s = rand(InverseGamma(α, β))


        ESS_SingleStep(W_θ, ph, b -> (likelihood_Y(b[1:P], X_N, Y_N, g, exp(b[P+1]), exp(b[P+2]), ph1, Σ1) + 
                        likelihood_W(b[1:P], X_N, exp(b[P+3]), g, ph2, Σ2) + prior_θ(exp(b[P+1])) + b[P+1] + 
                        prior_θ(exp(b[P+2])) + b[P+2] + prior_θ(exp(b[P+3])) + b[P+3] - dMvT(b, μ, Σ_chol.L, 
                        ph3, ν, P+3)), μ, sqrt(s) * Σ_chol.L, i)

        W[i,:] .= W_θ[i, 1:P]
        θ_y_x[i] = W_θ[i, P+1]
        θ_y_w[i] = W_θ[i, P+2]
        θ_w[i] = W_θ[i, P+3]
        ## Update next state

        if (i % 100) == 0
            println("MCMC iter: ", i)
            log_lik = @sprintf("%.2f", likelihood_Y(W[i,:], X_N, Y_N, g, exp(θ_y_x[i]), exp(θ_y_w[i]), ph1, Σ1))
            println("Log Likelihood: ", log_lik)
        end
        
        if i < n_MCMC
            W[i+1,:] .= W[i,:]
            θ_w[i+1] = θ_w[i]
            θ_y_x[i+1] = θ_y_x[i]
            θ_y_w[i+1] = θ_y_w[i]
        end

    end

    θ_w .= exp.(θ_w)
    θ_y_x .= exp.(θ_y_x)
    θ_y_w .= exp.(θ_y_w)

    return nothing
end

function sampler_AGESS(Y_N::AbstractVector{Y}, X_N::AbstractVector{Y}, W::AbstractMatrix{Y},
                       θ_w::AbstractVector{Y}, θ_y_x::AbstractVector{Y}, θ_y_w::AbstractVector{Y}, 
                       g::Y, prior_θ::Function, t_dist::Bool; ν::Y = 6.0, ϵ::Y = 0.00) where {Y<:AbstractFloat, T<:Integer}

    n_MCMC = size(W)[1]
    P = length(Y_N)

    Σ2 = diagm(ones(P))
    Σ1 = diagm(ones(P))
    Σ_chol_adapt = cholesky(diagm(ones(P+3)))
    ph = ones(P + 3)
    ph1 = ones(P)
    ph2 = ones(P)
    println(ν)

    W_θ = zeros(n_MCMC, P + 3)
    W_θ[1, 1:P] .= W[1,:]
    W_θ[1, P+1] = θ_y_x[1]
    W_θ[1, P+2] = θ_y_w[1]
    W_θ[1, P+3] = θ_w[1]

    μ_adapt = zeros(P + 3)


    for i in 2:n_MCMC
        W_θ[i, 1:P] .= W[i,:]
        W_θ[i, P+1] = θ_y_x[i]
        W_θ[i, P+2] = θ_y_w[i]
        W_θ[i, P+3] = θ_w[i]
        if rand() > ϵ
            ## Sample W, θ_y_x, θ_y_w using AGESS E(μ,Σ)
            AGESS_SingleStep(W_θ, b -> likelihood_Y(b[1:P], X_N, Y_N, g, exp(b[P+1]), exp(b[P+2]), ph1, Σ1) , c -> (likelihood_W(c[1:P], X_N, exp(c[P+3]), g, ph2, Σ2) + prior_θ(exp(c[P+1])) + c[P+1] + prior_θ(exp(c[P+2])) + c[P+2] + prior_θ(exp(c[P+3])) + c[P+3]), 
                             ph, t_dist, ν, μ_adapt, Σ_chol_adapt.L, i)
        else
            ## Sample W, θ_y_x, θ_y_w using AGESS E(0,I)
            AGESS_SingleStep(W_θ, b -> likelihood_Y(b[1:P], X_N, Y_N, g, exp(b[P+1]), exp(b[P+2]), ph1, Σ1) , c -> (likelihood_W(c[1:P], X_N, exp(c[P+3]), g, ph2, Σ2) + prior_θ(exp(c[P+1])) + c[P+1] + prior_θ(exp(c[P+2])) + c[P+2] + prior_θ(exp(c[P+3])) + c[P+3]), 
                             ph, t_dist, ν, zeros(P+3), LowerTriangular(diagm(ones(P+3))), i)
        end
        W[i,:] .= W_θ[i, 1:P]
        θ_y_x[i] = W_θ[i, P+1]
        θ_y_w[i] = W_θ[i, P+2]
        θ_w[i] = W_θ[i, P+3]
        ## Adapt parameters
        w_i = min(1.0/(10 * P), (i^(-2/3)))
        μ_adapt .= (1 - w_i) * μ_adapt +  w_i * W_θ[i,:]
        Σ_chol_adapt.U .= sqrt((1 - w_i)) .*  Σ_chol_adapt.U 
        lowrankupdate!(Σ_chol_adapt, sqrt(w_i) .* (W_θ[i,:] .- μ_adapt))


        if (i % 100) == 0
            println("MCMC iter: ", i)
            log_lik = @sprintf("%.2f", likelihood_Y(W[i,:], X_N, Y_N, g, exp(θ_y_x[i]), exp(θ_y_w[i]), ph1, Σ1))
            println("Log Likelihood: ", log_lik)
        end

        ## Update next state
        if i < n_MCMC
            W[i+1,:] .= W[i,:]
            θ_w[i+1] = θ_w[i]
            θ_y_x[i+1] = θ_y_x[i]
            θ_y_w[i+1] = θ_y_w[i]
        end
    end
    θ_w .= exp.(θ_w)
    θ_y_x .= exp.(θ_y_x)
    θ_y_w .= exp.(θ_y_w)

end

## Generate data

N_obs = 50 
X_N, Y_N = gen_data(N_obs, -5.0, 5.0)

W = ones(100000, N_obs) 
## Initialize with W = X_N
W[1,:] .= X_N
W[2,:] .= X_N

g = 1e-1
θ_w = ones(100000) * 0.5
θ_y_x = ones(100000) * 0.5
θ_y_w = ones(100000) * 0.5

sampler_ESS(Y_N, X_N, W, θ_w, θ_y_x, θ_y_w, g, k -> logpdf(Gamma(2,1), k))

time_points = collect(collect(LinRange(-5.0, 5.0, 500)))
time_points = setdiff(time_points, X_N)

Y_pred = predictive_draws(time_points, W, θ_w, θ_y_x, θ_y_w, g, Y_N, X_N)
truth = gen_data(time_points)
p = plot_CI(Y_N, X_N, Y_pred, time_points, truth)
p

p1 = plot_Var(Y_pred, time_points)
p1

p2 = plot(θ_w, ylabel = "θ_w", label = "θ_w")
p3 = plot(θ_y_x, ylabel = "θ_y_x", label = "θ_y_x")
p4 = plot(θ_y_w, ylabel = "θ_y_w", label = "θ_y_w")
p5 = plot(W[90000:100000,:]', ylabel = "W", legend=false)
plot(p, p1, p2, p3, p4, p5, layout=(3,2))
plot!(size=(4000,2000))

savefig("//Users//ndm34//Downloads//ESS_DGP2.png")

g = 1e-8
W_GESS = ones(100000, N_obs) 
## Initialize with W = X_N
W_GESS[1,:] .= X_N
W_GESS[2,:] .= X_N


θ_w_GESS = log.(ones(100000) * 0.5)
θ_y_x_GESS = log.(ones(100000) * 0.5)
θ_y_w_GESS = log.(ones(100000) * 0.5)


sampler_GESS(Y_N, X_N, W_GESS, θ_w_GESS, θ_y_x_GESS, θ_y_w_GESS, g, k -> logpdf(Gamma(1,2), k),
    zeros(N_obs + 3), diagm(ones(N_obs + 3)))


W_AGESS = ones(100000, N_obs) 
## Initialize with W = X_N
W_AGESS[1,:] .= X_N
W_AGESS[2,:] .= X_N


θ_w_AGESS = log.(ones(100000) * 0.5)
θ_y_x_AGESS = log.(ones(100000) * 0.5)
θ_y_w_AGESS = log.(ones(100000) * 0.5)
sampler_AGESS(Y_N, X_N, W_AGESS, θ_w_AGESS, θ_y_x_AGESS, θ_y_w_AGESS, g, k -> logpdf(Gamma(1,2), k), acceptance, true, ν = 3.0, ϵ = 0.1)


time_points = collect(collect(LinRange(-5.0, 5.0, 500)))
time_points = setdiff(time_points, X_N)
Y_pred = predictive_draws(time_points, W_AGESS, θ_w_AGESS, θ_y_x_AGESS, θ_y_w_AGESS, g, Y_N, X_N)
truth = gen_data(time_points)
p = plot_CI(Y_N, X_N, Y_pred, time_points, truth)
p

p1 = plot_Var(Y_pred, time_points)
p1

p2 = plot(θ_w_AGESS, ylabel = "θ_w", label = "θ_w")
p3 = plot(θ_y_x_AGESS, ylabel = "θ_y_x", label = "θ_y_x")
p4 = plot(θ_y_w_AGESS, ylabel = "θ_y_w", label = "θ_y_w")
p5 = plot(W_AGESS[90000:100000,:]', ylabel = "W", legend=false)

plot(p, p1, p2, p3, p4, p5, layout=(3,2))
plot!(size=(4000,2000))

savefig("//Users//ndm34//Downloads//AGESS_DGP_Mixture.png")


X_N, Y_N = gen_data_Higdon(25, 0.0, 15.0)
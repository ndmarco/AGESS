using KernelFunctions, LinearAlgebra, LogExpFunctions, Distributions, LinearAlgebra, JLD2, Random, StatsBase, RCall, StatsPlots
using StanBase
#set_cmdstan_home!("/Users/ndm34/Projects/cmdstan")
set_cmdstan_home!("C:\\Users\\ndmar\\Projects\\cmdstan")
using StanSample, DataFrames, Stan
include("AGESS.jl")


function posterior(Λ::AbstractMatrix{Y}, η::AbstractMatrix{Y}, Y_obs::AbstractMatrix{Y}, Σ::AbstractMatrix{Y}, 
                   ϕ::AbstractMatrix{Y}, δ::AbstractVector{Y}, τ_ph::AbstractVector{Y}, a_1::Y, a_2::Y, ν::Y,
                   ph::AbstractVector{Y})::Float64 where {Y<:AbstractFloat}
    lpdf::Float64 = 0.0
    ## Likelihood
    for i in 1:size(Y_obs)[1]
        @views mul!(ph, Λ', η[i,:])
        @views ph .-= Y_obs[i,:]
        @views lpdf += (-0.5* sum(log.(Σ[diagind(Σ)])) - 0.5 * dot(ph ./ Σ[diagind(Σ)], ph))
    end

    ##Priors 
    for i in eachindex(δ)
        if i == 1
            lpdf += logpdf(Gamma(a_1, 1), exp(δ[1])) + δ[1]
        else 
            lpdf += logpdf(Gamma(a_2, 1), exp(δ[i])) + δ[i] 
        end
    end

    ig_d = InverseGamma(1,1)
    for i in 1:size(Σ)[1]
        @views lpdf += logpdf(ig_d, Σ[i,i]) + log(Σ[i,i])
    end

    τ_ph .= exp.(δ)
    for i in 2:length(δ)
        τ_ph[i] *= τ_ph[i-1]
    end

    gamma_d = Gamma(0.5 * ν, 0.5 * ν)
    for i in 1:size(Λ)[1], j in 1:size(Λ)[2]
        @views lpdf += logpdf(Normal(0.0, sqrt(1 / (exp(ϕ[i,j]) * τ_ph[i]))), Λ[i,j])
        @views lpdf += logpdf(gamma_d, exp(ϕ[i,j])) + ϕ[i,j]
    end

    for i in 1:size(η)[1]
        @views lpdf += - 0.5 * dot(η[i,:], η[i,:])
    end

    return lpdf
end

function transform_posterior(Λ::AbstractVector{Y}, η::AbstractVector{Y}, Y_obs::AbstractMatrix, σ_sq::AbstractVector{Y},
                             ϕ::AbstractVector{Y}, δ::AbstractVector{Y}, τ_ph::AbstractVector{Y}, a_1::Y, a_2::Y, ν::Y,
                             N::T, P::T, K::T, Σ_ph::AbstractMatrix{Y}, ph::AbstractVector{Y})::Float64 where {Y<:AbstractFloat, T<:Integer}
    Λ_ph = reshape(Λ, (K, P))
    η_ph = reshape(η, (N, K))
    Σ_ph[diagind(Σ_ph)] .= exp.(σ_sq)
    ϕ_ph = reshape(ϕ, (K, P))
    @views lpdf = posterior(Λ_ph, η_ph, Y_obs, Σ_ph, ϕ_ph, δ, τ_ph, a_1, a_2, ν, ph)

    return lpdf
end

K = 6
N = 1000
P = 500
N_MCMC = 100000

a_1 = 2.0
a_2 = 2.0
ν = 2.0

Σ = diagm(ones(P))
ph = zeros(P)
for i in 1:(K - 3)
    ph = randn(P)
    Σ .+= i .* (ph * ph')
end

μ_0 = zeros(P)
y_obs = rand(MvNormal(μ_0, Σ), N)'
Σ_truth = similar(Σ)
Σ_truth .= Σ


MCMC_iters = 100000
x_AGESS1 = ones(MCMC_iters, (N * K) + 2*(K * P) + P + K)
Σ = diagm(ones((N * K) + 2*(K * P) + P + K))
μ_AGESS = zeros((N * K) + 2*(K * P) + P + K)

τ_ph = zeros(K)
Σ_ph = diagm(ones(P))

ph = zeros(P)
Λ_ph = zeros(K, P)
η_ph = zeros(N, K)
ϕ_ph = zeros(K, P)
AGESS_time1 = AGESS(x_AGESS1, b -> transform_posterior(b[1:K*P], b[(K*P + 1):(N*K + K*P)], y_obs, 
                                                      b[(N*K + K*P + 1):(N*K + K*P + P)],
                                                      b[(N*K + K*P + P + 1):(N*K + 2*K*P + P)], 
                                                      b[(N*K + 2*K*P + P + 1):(N*K + 2*K*P + P + K)],
                                                      τ_ph, a_1, a_2, ν, N, P, K, Σ_ph, ph), 
                    μ_AGESS, Σ, true, burnin = 0.5)
b = x_AGESS[1,:]


τ_ph = zeros(K)
Σ_ph = diagm(ones(P))

ph = zeros(P)
μ_0 = zeros(K)
Λ_ph = zeros(K, P)
η_ph = zeros(N, K)
ϕ_ph = zeros(K, P)
@benchmark lpdf = transform_posterior(b[1:K*P], b[(K*P + 1):(N*K + K*P)], y_obs, b[(N*K + K*P + 1):(N*K + K*P + P)],
                           b[(N*K + K*P + P + 1):(N*K + 2*K*P + P)], b[(N*K + 2*K*P + P + 1):(N*K + 2*K*P + P + K)],
                           τ_ph, a_1, a_2, ν, N, P, K, Σ_ph, ph)


function posterior_Σ(x_AGESS::AbstractMatrix{Y}, P::T, K::T, N::T; burnin = 0.5) where {Y<:AbstractFloat, T<:Integer}
    n_MCMC = size(x_AGESS)[1]
    burnin_num = floor(Int64, burnin * n_MCMC)
    posterior_samps = zeros(n_MCMC - burnin_num, P, P)
    for i in (burnin_num +1):n_MCMC
        Λ_ph = reshape(x_AGESS[i, 1:K*P], (K, P))
        @views posterior_samps[i - burnin_num,:,:] .= Λ_ph' * Λ_ph
        posterior_samps[i - burnin_num,:,:] .+= diagm(exp.(x_AGESS[i, (N*K + K*P + 1):(N*K + K*P + P)]))
    end
    return posterior_samps
end

samps1 = posterior_Σ(x_AGESS1, P, K ,N, burnin = 0.5)
plot(samps1[:, 1,1])
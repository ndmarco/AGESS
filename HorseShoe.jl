using StanBase
set_cmdstan_home!("/Users/ndm34/Projects/cmdstan")
#set_cmdstan_home!("C:\\Users\\ndmar\\Projects\\cmdstan")
using StanSample, DataFrames, Stan
include("AGESS.jl")
using LinearAlgebra, LogExpFunctions, Distributions, LinearAlgebra, JLD2, Random, StatsBase, RCall, StatsPlots, KernelDensity

model = "
data {
  int N;
  int P;
  matrix[N,P] X;
  vector[N] y;
}

parameters {
  vector[P] beta;
  vector<lower = 0>[P] lambda;
  real<lower = 0> tau;
  real<lower = 0> sigma;
}

model {
  for (i in 1:P){
    lambda[i] ~ cauchy(0,1);
  }
  tau ~ cauchy(0,1);
  sigma ~ inv_gamma(3,1);
  for (i in 1:P){
    beta[i] ~ normal(0, tau * lambda[i]);
  }

  y ~ normal_id_glm(X, 0, beta, sigma);
}
";

function gen_data(N::T, P::T; sparsity::Y = 0.8, ρ::Y = 0.2, σ_sq::Y = 1.0) where {Y<:AbstractFloat, T<:Integer}
    Σ = ones(P, P) * ρ
    Σ[diagind(Σ)] .= 1
    X = zeros(N, P)
    X .= rand(MultivariateNormal(zeros(P), Σ), N)'
    β = zeros(P)
    for i in 1:P
        if rand(Bernoulli(1 - sparsity)) == 1
            β[i] = rand(TDist(2.0))
        end
    end

    Y_obs = rand(MultivariateNormal(X * β, σ_sq * diagm(ones(N))))

    return X, Y_obs, β
end




N = 100
P = 50

X,Y, β = gen_data(N, P, ρ = 0.0)

data = Dict("N" => N, "P" => P, "X" => X, "y" => Y)


sm_HS = SampleModel("HorseShoe", model);

rc = stan_sample(sm_HS; num_cpp_chains=1, num_chains=1, num_warmups=50000, num_samples=50000, data);

df = read_samples(sm_HS, :array);
df = df[:,:,1]


function prior_β(β::AbstractVector{Y}, τ::Y, λ::AbstractVector{Y}) where {Y<:AbstractFloat}
    lpdf = 0.0
    for i in eachindex(β)
        lpdf += logpdf(Normal(0, exp(τ) * exp(λ[i])), β[i])
    end
    return lpdf
end

function prior_λ(λ::AbstractVector{Y}) where {Y<:AbstractFloat}
    lpdf = 0.0
    for i in eachindex(λ)
        lpdf += logpdf(Cauchy(0,1), exp(λ[i])) + λ[i]
    end
    return lpdf
end

MCMC_iters = 100000
x_AGESS = zeros(MCMC_iters, 2*P+2)
x_AGESS[:,(2*P+1):end] .= 0.5
Σ_I = diagm(ones(N))
Σ = diagm(ones(2*P+2))
μ_AGESS = zeros(2*P+2)
AGESS_time = AGESS(x_AGESS, b -> logpdf(MvNormal(data["X"] * b[1:P], Σ_I .* exp(b[2*P+2])), data["y"]), 
      c -> (prior_β(c[1:P], c[2*P+1], c[(P+1):(2*P)]) + prior_λ(c[(P+1):(2*P)]) + logpdf(Cauchy(0, 1), exp(c[2*P+1])) + logpdf(InverseGamma(3, 1), exp(c[2*P+2])) + c[2*P+1] + c[2*P+2]), 
      μ_AGESS, Σ, true)


x_GESS = zeros(MCMC_iters, 2*P+2)
x_GESS[:,(2*P+1):end] .= 0.5
μ_GESS = zeros(2*P+2)
GESS_time = GESS(x_GESS,  b -> (logpdf(MvNormal(data["X"] * b[1:P], Σ_I .* exp(b[2*P+2])), data["y"]) + prior_β(b[1:P], b[2*P+1], b[(P+1):(2*P)]) + prior_λ(b[(P+1):(2*P)]) + logpdf(Cauchy(0, 1), exp(b[2*P+1])) + logpdf(InverseGamma(3, 1), exp(b[2*P+2])) + b[2*P+1] + b[2*P+2]),
                 μ_GESS, Σ)



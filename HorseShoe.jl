using StanBase
set_cmdstan_home!("/Users/ndm34/Projects/cmdstan")
#set_cmdstan_home!("C:\\Users\\ndmar\\Projects\\cmdstan")
using StanSample, DataFrames, Stan
include("AGESS.jl")
using LinearAlgebra, LogExpFunctions, Distributions, LinearAlgebra, JLD2, Random, StatsBase, RCall, StatsPlots, KernelDensity

dir = "/Users/ndm34/Projects/AGESS_Simulation/Horseshoe"
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
    beta[i] ~ normal(0, sigma * tau^2 * lambda[i]^2);
  }

  y ~ normal_id_glm(X, 0, beta, sigma);
}
";

function prior_β(β::AbstractVector{Y}, τ::Y, λ::AbstractVector{Y}, σ_sq::Y)::Float64 where {Y<:AbstractFloat}
  lpdf::Float64 = 0.0
  for i in eachindex(β)
      lpdf += logpdf(Normal(0, exp(σ_sq) * exp(τ)^2 * exp(λ[i])^2), β[i])
  end
  return lpdf
end

function prior_λ(λ::AbstractVector{Y})::Float64 where {Y<:AbstractFloat}
  lpdf::Float64 = 0.0
  for i in eachindex(λ)
      lpdf += logpdf(Cauchy(0,1), exp(λ[i])) + λ[i]
  end
  return lpdf
end

function log_posterior(β::AbstractVector{Y}, τ::Y, λ::AbstractVector{Y}, σ_sq::Y, X::AbstractMatrix{Y}, y::AbstractVector{Y})::Float64 where {Y<:AbstractFloat}
  lpdf::Float64 = 0.0
  ## Likelihood
  lpdf = logpdf(MvNormal(X * β, exp(σ_sq)), y)
  ## Prior 
  lpdf += prior_β(β, τ, λ, σ_sq) + prior_λ(λ) + logpdf(Cauchy(0,1), exp(τ)) + τ

  return lpdf
end

function gen_data(N::T, P::T; sparsity::Y = 0.8, ρ::Y = 0.2, σ_sq::Y = 1.0) where {Y<:AbstractFloat, T<:Integer}
    Σ = ones(P, P) * ρ
    Σ[diagind(Σ)] .= 1
    X = zeros(N, P)
    X .= rand(MultivariateNormal(zeros(P), Σ), N)'
    β = zeros(P)
    for i in 1:P
        if rand(Bernoulli(1 - sparsity)) == 1
            β[i] = rand(TDist(2.0) * 2)
        end
    end

    Y_obs = rand(MultivariateNormal(X * β, σ_sq * diagm(ones(N))))

    return X, Y_obs, β
end


######################################
############### P < N  ###############
######################################


### Low correlation

N = 100
P = 50

X,Y, β = gen_data(N, P, ρ = 0.3, sparsity = 0.9)


data = Dict("N" => N, "P" => P, "X" => X, "y" => Y)
sm_HS = SampleModel("HorseShoe", model);

### STAN
t1 = time()
rc = stan_sample(sm_HS; num_chains=1, num_warmups=100000, num_samples=100000, data);
stan_time = time() - t1
df = read_samples(sm_HS, :array);
df = df[:,:,1]

Stan_β = plot(df[:, findall(β .!= 0)], legend = false)
Stan_β_0 = plot(df[:, findall(β .== 0)], legend = false)

### AGESS
MCMC_iters = 200000
x_AGESS = zeros(MCMC_iters, 2*P+2)
Σ = diagm(ones(2*P+2))
μ_AGESS = zeros(2*P+2)
AGESS_time = AGESS(x_AGESS, b -> log_posterior(b[1:P], b[2*P+1], b[(P+1):(2*P)], b[2*P+2], data["X"], data["y"]), 
      μ_AGESS, Σ, true)

AGESS_β = plot(x_AGESS[100000:end, findall(β .!= 0)], legend = false)
AGESS_β_0 = plot(x_AGESS[100000:end, findall(β .== 0)], legend = false)

### GESS
x_GESS = zeros(MCMC_iters, 2*P+2)
x_GESS[:,(2*P+1):end] .= 0.5
μ_GESS = zeros(2*P+2)
GESS_time = GESS(x_GESS,  b -> log_posterior(b[1:P], b[2*P+1], b[(P+1):(2*P)], b[2*P+2], data["X"], data["y"]),
                 μ_GESS, Σ)

GESS_β = plot(x_GESS[100000:end, findall(β .!= 0)], legend = false)
GESS_β_0 = plot(x_GESS[100000:end, findall(β .== 0)], legend = false)

@rput X
@rput Y
R"""
library(CoupledHalfT)
library(horseshoe)

X_transpose <- t(X)
burnin <- 0
chain_length <- 200000
chain <- NA
time1 = Sys.time()
hs_chain <- horseshoe(Y, X, method.tau = "halfCauchy", method.sigma = "Jeffreys", nmc = chain_length, burn = 0)
#try(chain <- half_t_mcmc(chain_length, burnin, X, X_transpose, y, t_dist_df=2))
time_end = Sys.time() - time1
beta_samps_HS = hs_chain$BetaSamples
HS_time = time_end

time1 = Sys.time()
half_t_chain <- half_t_mcmc(chain_length, burnin, X, X_transpose, Y, t_dist_df=3)
time_end = Sys.time() - time1
half_t_time = time_end
beta_samps_half_t = half_t_chain$beta_samples
"""
@rget beta_samps_HS
@rget HS_time
@rget half_t_time
@rget beta_samps_half_t
HS_β = plot(beta_samps_HS[findall(β .!= 0), 100000:end]', legend = false)
HS_β_0 = plot(beta_samps_HS[findall(β .== 0), 100000:end]', legend = false)

half_t_β = plot(beta_samps_half_t[100000:end, findall(β .!= 0)], legend = false)
half_t_β_0 = plot(beta_samps_half_t[100000:end, findall(β .== 0)], legend = false)


plot(GESS_β, AGESS_β, Stan_β, HS_β, half_t_β, GESS_β_0, AGESS_β_0, Stan_β_0, HS_β_0, half_t_β_0, layout = @layout([A B C D E ; F G H I J]), margin= 5Plots.mm)
plot!(size = (2500, 1000))

savefig(string(dir, "/low_dim_low_corr2.pdf"))

x_GESS1 = x_GESS[100001:end,1:50]
x_AGESS1 = x_AGESS[100001:end,1:50]
x_stan1 = df[:,1:50]
x_HS1 = beta_samps_HS[:,100001:end]'
x_half_t1 = beta_samps_half_t[100001:end,:]
@rput x_GESS1
@rput x_AGESS1
@rput x_HS1
@rput x_stan1
@rput x_half_t1
R"""
library(mcmcse)
mats = rbind(x_GESS1, x_AGESS1, x_HS1, x_stan1, x_half_t1)
sigma = mcse.multi(x_GESS1)$cov
ess_GESS <- multiESS(mats, covmat = sigma) / 5
multiESS(x_ESS1)
sigma = mcse.multi(x_AGESS1)$cov
ess_AGESS <- multiESS(mats, covmat = sigma) / 5
sigma = mcse.multi(x_stan1)$cov
ess_Stan <- multiESS(mats, covmat = sigma) / 5
sigma = mcse.multi(x_HS1)$cov
ess_HS <- multiESS(mats, covmat = sigma) / 5
sigma = mcse.multi(x_half_t1)$cov
ess_half_t <- multiESS(mats, covmat = sigma) / 5
"""


#### High correlation
N = 100
P = 50

X,Y, β = gen_data(N, P, ρ = 0.9, sparsity = 0.9)


data = Dict("N" => N, "P" => P, "X" => X, "y" => Y)
sm_HS = SampleModel("HorseShoe", model);

### STAN
t1 = time()
rc = stan_sample(sm_HS; num_chains=1, num_warmups=100000, num_samples=100000, data);
stan_time = time() - t1
df = read_samples(sm_HS, :array);
df = df[:,:,1]

Stan_β_HC = plot(df[:, findall(β .!= 0)], legend = false)
Stan_β_0_HC = plot(df[:, findall(β .== 0)], legend = false)

### AGESS
MCMC_iters = 200000
x_AGESS = zeros(MCMC_iters, 2*P+2)
Σ = diagm(ones(2*P+2))
μ_AGESS = zeros(2*P+2)
AGESS_time = AGESS(x_AGESS, b -> log_posterior(b[1:P], b[2*P+1], b[(P+1):(2*P)], b[2*P+2], data["X"], data["y"]), 
      μ_AGESS, Σ, true)

AGESS_β_HC = plot(x_AGESS[100000:end, findall(β .!= 0)], legend = false)
AGESS_β_0_HC = plot(x_AGESS[100000:end, findall(β .== 0)], legend = false)

### GESS
x_GESS = zeros(MCMC_iters, 2*P+2)
x_GESS[:,(2*P+1):end] .= 0.5
μ_GESS = zeros(2*P+2)
GESS_time = GESS(x_GESS,  b -> log_posterior(b[1:P], b[2*P+1], b[(P+1):(2*P)], b[2*P+2], data["X"], data["y"]),
                 μ_GESS, Σ)

GESS_β_HC = plot(x_GESS[100000:end, findall(β .!= 0)], legend = false)
GESS_β_0_HC = plot(x_GESS[100000:end, findall(β .== 0)], legend = false)

@rput X
@rput Y
R"""
library(CoupledHalfT)
library(horseshoe)

X_transpose <- t(X)
burnin <- 0
chain_length <- 200000
chain <- NA
time1 = Sys.time()
hs_chain <- horseshoe(Y, X, method.tau = "halfCauchy", method.sigma = "Jeffreys", nmc = chain_length, burn = 0)
#try(chain <- half_t_mcmc(chain_length, burnin, X, X_transpose, y, t_dist_df=2))
time_end = Sys.time() - time1
beta_samps_HS = hs_chain$BetaSamples
HS_time = time_end

time1 = Sys.time()
half_t_chain <- half_t_mcmc(chain_length, burnin, X, X_transpose, Y, t_dist_df=3)
time_end = Sys.time() - time1
half_t_time = time_end
beta_samps_half_t = half_t_chain$beta_samples
"""
@rget beta_samps_HS
@rget HS_time
@rget half_t_time
@rget beta_samps_half_t
HS_β_HC = plot(beta_samps_HS[findall(β .!= 0), 100000:end]', legend = false)
HS_β_0_HC = plot(beta_samps_HS[findall(β .== 0), 100000:end]', legend = false)

half_t_β_HC = plot(beta_samps_half_t[100000:end, findall(β .!= 0)], legend = false)
half_t_β_0_HC = plot(beta_samps_half_t[100000:end, findall(β .== 0)], legend = false)


plot(GESS_β_HC, AGESS_β_HC, Stan_β_HC, HS_β_HC, half_t_β_HC, GESS_β_0_HC, AGESS_β_0_HC, Stan_β_0_HC, HS_β_0_HC, half_t_β_0_HC, layout = @layout([A B C D E ; F G H I J]), margin= 5Plots.mm)
plot!(size = (2500, 1000))

savefig(string(dir, "/low_dim_high_corr.pdf"))

false_positive = 0
for i in 1:length(findall(β .== 0))
  quant = quantile(x_AGESS1[:,findall(β .== 0)[i]], [0.025 0.975])
  if quant[1]> 0
    false_positive += 1
  end
  if quant[2] < 0
    false_positive += 1
  end
end

x_GESS1 = x_GESS[100001:end,1:50]
x_AGESS1 = x_AGESS[100001:end,1:50]
x_stan1 = df[:,1:50]
x_HS1 = beta_samps_HS[:,100001:end]'
x_half_t1 = beta_samps_half_t[100001:end,:]
@rput x_GESS1
@rput x_AGESS1
@rput x_HS1
@rput x_stan1
@rput x_half_t1
R"""
library(mcmcse)
mats = rbind(x_GESS1, x_AGESS1, x_HS1, x_stan1, x_half_t1)
sigma = mcse.multi(x_GESS1)$cov
ess_GESS <- multiESS(mats, covmat = sigma) / 5
multiESS(x_ESS1)
sigma = mcse.multi(x_AGESS1)$cov
ess_AGESS <- multiESS(mats, covmat = sigma) / 5
sigma = mcse.multi(x_stan1)$cov
ess_Stan <- multiESS(mats, covmat = sigma) / 5
sigma = mcse.multi(x_HS1)$cov
ess_HS <- multiESS(mats, covmat = sigma) / 5
sigma = mcse.multi(x_half_t1)$cov
ess_half_t <- multiESS(mats, covmat = sigma) / 5
"""




######################################
############### P > N  ###############
######################################


N = 100
P = 500

X,Y,β = gen_data(N, P, ρ = 0.5, sparsity = 0.98)

data = Dict("N" => N, "P" => P, "X" => X, "y" => Y)


sm_HS = SampleModel("HorseShoe", model);

t1 = time()
rc = stan_sample(sm_HS; num_chains=1, num_warmups=10000, num_samples=40000, data);
stan_time = time() - t1
df = read_samples(sm_HS, :array);
df = df[:,:,1]




MCMC_iters = 200000
x_AGESS = zeros(MCMC_iters, 2*P+2)
Σ = diagm(ones(2*P+2))
μ_AGESS = zeros(2*P+2)
AGESS_time = AGESS(x_AGESS, b -> log_posterior(b[1:P], b[2*P+1], b[(P+1):(2*P)], b[2*P+2], data["X"], data["y"]), 
      μ_AGESS, Σ, true)


x_GESS = zeros(MCMC_iters, 2*P+2)
x_GESS[:,(2*P+1):end] .= 0.5
μ_GESS = zeros(2*P+2)
GESS_time = GESS(x_GESS,  b -> (logpdf(MvNormal(data["X"] * b[1:P], Σ_I .* exp(b[2*P+2])), data["y"]) + prior_β(b[1:P], b[2*P+1], b[2*P+2], b[(P+1):(2*P)]) + prior_λ(b[(P+1):(2*P)]) + logpdf(Cauchy(0, 1), exp(b[2*P+1])) + logpdf(InverseGamma(3, 1), exp(b[2*P+2])) + b[2*P+1] + b[2*P+2]),
                 μ_GESS, Σ)



x_AGESS1 = x_AGESS[60000:end,:]
@rput X
@rput Y
@rput x_AGESS1
@rput df
R"""
library(CoupledHalfT)
library(horseshoe)

X_transpose <- t(X)
burnin <- 0
chain_length <- 100000
chain <- NA
time1 = Sys.time()
hs_chain <- horseshoe(Y, X, method.tau = "halfCauchy", method.sigma = "Jeffreys", nmc = chain_length, burn = 0)
#try(chain <- half_t_mcmc(chain_length, burnin, X, X_transpose, y, t_dist_df=2))
time_end = Sys.time() - time1


library(mcmcse)
mats = rbind(x_AGESS1,df)
sigma = mcse.multi(x_AGESS1)$cov
ess_AGESS <- multiESS(mats, covmat = sigma) / 2
sigma = mcse.multi(df)$cov
ess_Stan <- multiESS(mats, covmat = sigma) / 2

library(stableGR)
ess_AGESS1 <- n.eff(x_AGESS1)$n.eff
ess_Stan1 <- n.eff(df)

"""


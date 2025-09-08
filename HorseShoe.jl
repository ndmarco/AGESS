using StanBase
#set_cmdstan_home!("/Users/ndm34/Projects/cmdstan")
set_cmdstan_home!("C:\\Users\\ndmar\\Projects\\cmdstan")
using StanSample, DataFrames, Stan
include("AGESS.jl")
using LinearAlgebra, LogExpFunctions, Distributions, LinearAlgebra, JLD2, Random, StatsBase, RCall, StatsPlots, KernelDensity

#dir = "/Users/ndm34/Projects/AGESS_Simulation/Horseshoe"
dir = "C:\\Users\\ndmar\\Projects\\AGESS_Simulation\\Horseshoe"
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
  lambda ~ cauchy(0,1);
  tau ~ cauchy(0,1);
  beta ~ normal(0, sigma * tau * lambda);
  y ~ normal_id_glm(X, 0, beta, sigma);
}
";

function prior_β(β::AbstractVector{Y}, τ::Y, λ::AbstractVector{Y}, σ::Y)::Float64 where {Y<:AbstractFloat}
  lpdf::Float64 = 0.0
  for i in eachindex(β)
      @views lpdf += logpdf(Normal(0, exp(σ) * exp(τ) * exp(λ[i])), β[i])
  end
  return lpdf
end

function prior_λ(λ::AbstractVector{Y})::Float64 where {Y<:AbstractFloat}
  lpdf::Float64 = 0.0
  cauchy_d = Cauchy(0,1)
  for i in eachindex(λ)
      @views lpdf += logpdf(cauchy_d, exp(λ[i])) + λ[i]
  end
  return lpdf
end

function log_posterior(β::AbstractVector{Y}, τ::Y, λ::AbstractVector{Y}, σ::Y, X::AbstractMatrix{Y}, y::AbstractVector{Y})::Float64 where {Y<:AbstractFloat}
  lpdf::Float64 = 0.0
  ## Likelihood
  for i in eachindex(y)
    @views lpdf += logpdf(Normal(dot(X[i,:], β), exp(σ)), y[i])
  end
  ## Prior 
  lpdf += prior_β(β, τ, λ, σ) + prior_λ(λ) + logpdf(Cauchy(0,1), exp(τ)) + τ - σ

  return lpdf
end

function gen_data(N::T, P::T; sparsity::Y = 0.8, ρ::Y = 0.2, σ_sq::Y = 1.0) where {Y<:AbstractFloat, T<:Integer}
    Σ = ones(P, P) 
    for i in 1:P
      for j in 1:P
        Σ[i,j] = ρ^(abs(i - j))
      end
    end
    Σ[diagind(Σ)] .= 1
    X = zeros(N, P)
    X .= rand(MultivariateNormal(zeros(P), Σ), N)'
    β = zeros(P)
    for i in 1:P
        if rand(Bernoulli(1 - sparsity)) == 1
          β[i] = randn() * 2
        end
    end

    Y_obs = rand(MultivariateNormal(X * β, σ_sq * diagm(ones(N))))

    return X, Y_obs, β
end


######################################
############### P < N  ###############
######################################


### Low correlation

N = 50
P = 100

X,Y, β = gen_data(N, P, ρ = 0.9, sparsity = 0.8, σ_sq = 0.5)
data = Dict("N" => N, "P" => P, "X" => X, "y" => Y)


### STAN
sm_HS = SampleModel("HorseShoe", model);
t1 = time()
rc = stan_sample(sm_HS; num_chains=1, num_warmups=10000, num_samples=100000, data);
stan_time = time() - t1
df = read_samples(sm_HS, :array);
df = df[:,:,1]

Stan_β = plot(df[1:10:end, findall(β .!= 0)], legend = false)
Stan_β_0 = plot(df[1:10:end, findall(β .== 0)], legend = false)

### AGESS
MCMC_iters = 200000
x_AGESS = zeros(MCMC_iters, 2*P+2)
Σ = diagm(ones(2*P+2))
μ_AGESS = zeros(2*P+2)
ph = zeros(N)
@time AGESS_time = AGESS(x_AGESS, b -> log_posterior(b[1:P], b[2*P+1], b[(P+1):(2*P)], b[2*P+2], data["X"], data["y"]), 
      μ_AGESS, Σ, true, burnin = 0.5)

AGESS_β = plot(x_AGESS[1:10:end, findall(β .!= 0)], legend = false, dpi = 300)
hline!(β[findall(β .!= 0)], line = :dash, color =:black)
AGESS_β_0 = plot(x_AGESS[1:10:end, findall(β .== 0)], legend = false, dpi = 300)

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
@rget beta_samps_half_t
HS_β = plot(beta_samps_HS[findall(β .!= 0), 100000:10:end]', legend = false)
hline!(β[findall(β .!= 0)], line = :dash, color =:black)
HS_β_0 = plot(beta_samps_HS[findall(β .== 0), 100000:10:end]', legend = false)

ind = findall(β .!= 0)
ind1 = ind[5]
ind2 = ind[2]

plot(kde((beta_samps_HS[ind1,:], beta_samps_HS[ind2,:])))
scatter!((β[ind1], β[ind2]))
plot(kde((x_AGESS[100000:end,ind1], x_AGESS[100000:end,ind2])))
scatter!((β[ind1], β[ind2]))

scatter(mean(beta_samps_HS, dims = 2))
scatter!(β)
scatter!(median(x_AGESS[100000:end, 1:P], dims = 1)')

histogram(β)

half_t_β = plot(beta_samps_half_t[100000:end, findall(β .!= 0)], legend = false)
half_t_β_0 = plot(beta_samps_half_t[100000:end, findall(β .== 0)], legend = false)

MCMC_iters = 200000
x_AGESS_BR = zeros(MCMC_iters, P+2)
Σ = diagm(ones(P+2))
μ_AGESS = zeros(P+2)
AGESS_time_BR = AGESS(x_AGESS_BR, b -> log_posterior_BR(b[1:P], b[P+1], b[P+2], data["X"], data["y"]), 
      μ_AGESS, Σ, true)

AGESS_β_BR = plot(x_AGESS_BR[100000:end, findall(β .!= 0)], legend = false)
AGESS_β_0_BR = plot(x_AGESS_BR[100000:end, findall(β .== 0)], legend = false)


plot(GESS_β, AGESS_β, Stan_β, HS_β, half_t_β, AGESS_β_BR, GESS_β_0, AGESS_β_0, Stan_β_0, HS_β_0, half_t_β_0, AGESS_β_0_BR, layout = @layout([A B C D E F; G H I J K L]), margin= 5Plots.mm)
plot!(size = (2500, 1000))

savefig(string(dir, "\\low_dim_low_corr2.jpg"))

#x_GESS1 = x_GESS[100001:end,1:P]
x_AGESS1 = x_AGESS[50001:end,1:P]
x_stan1 = df[:,1:P]
x_HS1 = beta_samps_HS[1:P,100001:end]'
#x_half_t1 = beta_samps_half_t[100001:end,P]
#@rput x_GESS1
@rput x_AGESS1
@rput x_HS1
@rput x_stan1
#@rput x_half_t1
R"""
library(mcmcse)
library(stable)
mats = rbind(x_AGESS1, x_HS1, x_stan1)
#sigma = mcse.multi(x_GESS1)$cov
#ess_GESS <- multiESS(mats, covmat = sigma) / 5
#multiESS(x_ESS1)
sigma = mcse.multi(x_AGESS1)$cov
ess_AGESS <- multiESS(mats, covmat = sigma) / 3
sigma = mcse.multi(x_stan1)$cov
ess_Stan <- multiESS(mats, covmat = sigma) / 3
sigma = mcse.multi(x_HS1)$cov
ess_HS <- multiESS(mats, covmat = sigma) / 3
#sigma = mcse.multi(x_half_t1)$cov
#ess_half_t <- multiESS(mats, covmat = sigma) / 5
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
HS_β_HC = plot(beta_samps_HS[findall(β .!= 0), 100000:10:end]', legend = false)
hline!(β[findall(β .!= 0)], color = "black", line = (3, :dash))
HS_β_0_HC = plot(beta_samps_HS[findall(β .== 0), 100000:10:end]', legend = false)

half_t_β_HC = plot(beta_samps_half_t[100000:end, findall(β .!= 0)], legend = false)
half_t_β_0_HC = plot(beta_samps_half_t[100000:end, findall(β .== 0)], legend = false)


plot(GESS_β_HC, AGESS_β_HC, Stan_β_HC, HS_β_HC, half_t_β_HC, GESS_β_0_HC, AGESS_β_0_HC, Stan_β_0_HC, HS_β_0_HC, half_t_β_0_HC, layout = @layout([A B C D E ; F G H I J]), margin= 5Plots.mm)
plot!(size = (2500, 1000))

savefig(string(dir, "/low_dim_high_corr.jpg"))

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

X,Y,β = gen_data(N, P, ρ = 0.9, sparsity = 0.98)

data = Dict("N" => N, "P" => P, "X" => X, "y" => Y)


sm_HS = SampleModel("HorseShoe", model);

t1 = time()
rc = stan_sample(sm_HS; num_chains=1, num_warmups=10000, num_samples=100000, data);
stan_time = time() - t1
df = read_samples(sm_HS, :array);
df = df[:,:,1]

Stan_β_HC = plot(df[1:5:end, findall(β .!= 0)], legend = false, dpi = 300)
Stan_β_0_HC = plot(df[1:5:end, findall(β .== 0)], legend = false, dpi = 300)

savefig(Stan_β_HC, string(dir, "\\stan_beta.pdf"))
savefig(Stan_β_0_HC, string(dir, "\\stan_beta_0.pdf"))

MCMC_iters = 200000
x_AGESS = zeros(MCMC_iters, 2*P+2)
Σ = diagm(ones(2*P+2))
μ_AGESS = zeros(2*P+2)
AGESS_time = AGESS(x_AGESS, b -> log_posterior(b[1:P], b[2*P+1], b[(P+1):(2*P)], b[2*P+2], data["X"], data["y"]), 
      μ_AGESS, Σ, true)

AGESS_β_HC = plot(x_AGESS[100000:5:end, findall(β .!= 0)], legend = false, dpi = 300)
AGESS_β_0_HC = plot(x_AGESS[100000:5:end, findall(β .== 0)], legend = false, dpi = 300)

savefig(AGESS_β_HC, string(dir, "\\AGESS_beta.pdf"))
savefig(AGESS_β_0_HC, string(dir, "\\AGESS_beta_0.pdf"))



x_AGESS1 = x_AGESS[60000:end,:]
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
HS_β_HC = plot(beta_samps_HS[findall(β .!= 0), 100000:5:end]', legend = false)
HS_β_0_HC = plot(beta_samps_HS[findall(β .== 0), 100000:5:end]', legend = false)

half_t_β_HC = plot(beta_samps_half_t[100000:5:end, findall(β .!= 0)], legend = false)
half_t_β_0_HC = plot(beta_samps_half_t[100000:5:end, findall(β .== 0)], legend = false)


function get_coverage(β::AbstractVector{Y}, x_AGESS::AbstractMatrix{Y}, x_STAN::AbstractMatrix{Y}, x_HS::AbstractMatrix{Y}, x_CHT::AbstractMatrix{Y}) where {Y<:AbstractFloat}
    inc_β_AGESS = 0
    inc_β_0_AGESS = 0
    inc_β_STAN = 0
    inc_β_0_STAN = 0
    inc_β_HS = 0
    inc_β_0_HS = 0
    inc_β_CHT = 0
    inc_β_0_CHT = 0
    for i in eachindex(β)
      if β[i] != 0
        CI = quantile(x_AGESS[100000:end,i], [0.025 0.975])
        if CI[1] <= β[i]
          if CI[2] >= β[i]
            inc_β_AGESS += 1
          end
        end
        
        CI = quantile(x_STAN[100000:end,i], [0.025 0.975])
        if CI[1] <= β[i]
          if CI[2] >= β[i]
            inc_β_STAN += 1
          end
        end

        CI = quantile(x_HS[100000:end,i], [0.025 0.975])
        if CI[1] <= β[i]
          if CI[2] >= β[i]
            inc_β_HS += 1
          end
        end

        CI = quantile(x_CHT[100000:end,i], [0.025 0.975])
        if CI[1] <= β[i]
          if CI[2] >= β[i]
            inc_β_CHT += 1
          end
        end
      else
        CI = quantile(x_AGESS[100000:end,i], [0.025 0.975])
        if CI[1] <= β[i]
          if CI[2] >= β[i]
            inc_β_0_AGESS += 1
          end
        end
        
        CI = quantile(x_STAN[100000:end,i], [0.025 0.975])
        if CI[1] <= β[i]
          if CI[2] >= β[i]
            inc_β_0_STAN += 1
          end
        end

        CI = quantile(x_HS[100000:end,i], [0.025 0.975])
        if CI[1] <= β[i]
          if CI[2] >= β[i]
            inc_β_0_HS += 1
          end
        end

        CI = quantile(x_CHT[100000:end,i], [0.025 0.975])
        if CI[1] <= β[i]
          if CI[2] >= β[i]
            inc_β_0_CHT += 1
          end
        end
      end
    end
    inc_β_AGESS = inc_β_AGESS / sum(β .!= 0)
    inc_β_STAN = inc_β_STAN / sum(β .!= 0)
    inc_β_HS = inc_β_HS / sum(β .!= 0)
    inc_β_CHT = inc_β_CHT / sum(β .!= 0)
    inc_β_0_AGESS = inc_β_0_AGESS / sum(β .== 0)
    inc_β_0_STAN = inc_β_0_STAN / sum(β .== 0)
    inc_β_0_HS = inc_β_0_HS / sum(β .== 0)
    inc_β_0_CHT = inc_β_0_CHT / sum(β .== 0)

    return inc_β_AGESS, inc_β_0_AGESS, inc_β_STAN, inc_β_0_STAN, inc_β_HS, inc_β_0_HS, inc_β_CHT, inc_β_0_CHT
end

coverage = get_coverage(β, x_AGESS[100000:end,:], df, beta_samps_HS[:, 100000:end]', beta_samps_half_t[100000:end,:])

##########################
### Poisson Case #########
##########################

function log_posterior_Poisson(β::AbstractVector{Y}, τ::Y, λ::AbstractVector{Y}, X::AbstractMatrix{Y}, y::AbstractVector{T})::Float64 where {Y<:AbstractFloat, T<:Integer}
  lpdf::Float64 = 0.0
  ## Likelihood
  for i in eachindex(y)
    @views lpdf += logpdf(Poisson(exp(dot(X[i,:], β))), y[i])
  end
  ## Prior 
  lpdf += prior_β(β, τ, λ, 0.0) + prior_λ(λ) + logpdf(Cauchy(0,1), exp(τ)) + τ

  return lpdf
end

function gen_data_Poisson(N::T, P::T; sparsity::Y = 0.8, ρ::Y = 0.2) where {Y<:AbstractFloat, T<:Integer}
    Σ = ones(P, P) 
    for i in 1:P
      for j in 1:P
        Σ[i,j] = ρ^(abs(i - j))
      end
    end
    X = zeros(N, P)
    X .= rand(MultivariateNormal(zeros(P), Σ), N)'
    β = zeros(P)
    for i in 1:P
        if rand(Bernoulli(1 - sparsity)) == 1
            β[i] = rand(TDist(2.0))
        end
    end

    Y_obs = zeros(Integer, N)
    for i in 1:N
      Y_obs[i] = rand(Poisson(exp(dot(X[i,:], β))))
    end

    return X, Y_obs, β
end

N = 50
P = 100

X, Y, β = gen_data_Poisson(N, P, ρ = 0.2, sparsity = 0.95)
data = Dict("N" => N, "P" => P, "X" => X, "y" => Y)

MCMC_iters = 200000
x_AGESS = zeros(MCMC_iters, 2*P+1)
Σ = diagm(ones(2*P+1))
μ_AGESS = zeros(2*P+1)
AGESS_time = AGESS(x_AGESS, b -> log_posterior_Poisson(b[1:P], b[2*P+1], b[(P+1):(2*P)], data["X"], data["y"]), 
      μ_AGESS, Σ, true)

AGESS_β_HC = plot(x_AGESS[100000:10:end, findall(β .!= 0)], legend = false)
AGESS_β_0_HC = plot(x_AGESS[100000:10:end, findall(β .== 0)], legend = false)


@rput X
@rput Y
R"""
library(bayesreg)
df1 <- data.frame(X,Y)
time1 = Sys.time()
rv.pois <- bayesreg(Y~., data=df1, model="poisson", prior="hs", burnin=1e5, n.samples=1e5)
time_end = Sys.time() - time1

beta_samples <- rv.pois$beta
"""

@rget beta_samples
br_β_HC = plot(beta_samples[findall(β .!= 0), 1:10:end]', legend = false)
br_β_0_HC = plot(beta_samples[findall(β .== 0), 1:10:end,], legend = false)


#######################
###### Real Data ######
#######################
### Biscuit dough dataset 

R"""
library(fds)
library(horseshoe)
data(labc)
y_pred = labc[1,]
data(labp)
y_obs = labp[1,]
y_pred = y_pred - mean(y_obs)
y_obs = y_obs - mean(y_obs)
data(nirp)
data(nirc)
X = t(nirp$y)
X_pred = t(nirc$y)

## Center and rescale
for(i in 1:ncol(X)){
  X_pred[,i] <- (X_pred[,i] - mean(X[,i])) / sd(X[,i])
  X[,i] <- (X[,i] - mean(X[,i])) / sd(X[,i])
}



X_transpose <- t(X)
# Run Horseshoe
burnin <- 0
chain_length <- 400000
chain <- NA
time1 = Sys.time()
hs_chain <- horseshoe(y_obs, X, method.tau = "halfCauchy", method.sigma = "Jeffreys", nmc = chain_length, burn = 0)
time_end = Sys.time() - time1
beta_samps_HS = hs_chain$BetaSamples
HS_time = time_end
sigma_samps_HS = hs_chain$Sigma2Samples

time1 = Sys.time()
half_t_chain <- half_t_mcmc(chain_length, burnin, X, X_transpose, y_obs, t_dist_df=3)
time_end = Sys.time() - time1
half_t_time = time_end
beta_samps_half_t = half_t_chain$beta_samples
sigma_samps_half_t = half_t_chain$sigma2_samples
"""
@rget beta_samps_HS
@rget beta_samps_half_t
@rget HS_time
@rget half_t_time
@rget X
@rget y_obs
@rget X_pred
@rget y_pred
@rget sigma_samps_HS
@rget sigma_samps_half_t
##

index_order = sortperm(abs.(mean(beta_samps_HS, dims = 2)), dims = 1)

g1 = scatter(mean(beta_samps_HS, dims = 2))
scatter!(median(beta_samps_HS2, dims = 2))
g1 = scatter(median(beta_samps_half_t1, dims = 1)')
scatter(median(beta_samps_half_t1, dims = 1)')
scatter!(median(x_AGESS1[100000:end, 1:P], dims = 1)')

ind = sortperm(abs.(median(beta_samps_HS2, dims = 2)), dims = 1)

sortperm(abs.(median(x_AGESS1[100000:end, 1:P], dims = 1)), dims = 2)

P = size(X)[2]
MCMC_iters = 400000
x_AGESS = zeros(MCMC_iters, 2*P+2)
Σ = diagm(ones(2*P+2))
μ_AGESS = zeros(2*P+2)
t1 = time()
AGESS_time = AGESS(x_AGESS, b -> log_posterior(b[1:P], b[2*P+1], b[(P+1):(2*P)], b[2*P+2], X, y_obs), 
      μ_AGESS, Σ, true, burnin = 0.5)
AGESS_total_time = time() - t1



function elppd(X::AbstractMatrix{Y}, Y_obs::AbstractVector{Y}, β_samples::AbstractMatrix{Y}, σ_samples::AbstractVector{Y}) where {Y<:AbstractFloat}
    lppd = zeros(length(Y_obs))
    for i in 1:length(Y_obs)
      for j in 1:size(β_samples)[1]
        @views lppd[i] += pdf(Normal(dot(X[i,:], β_samples[j,:]), σ_samples[j]), Y_obs[i])
      end
      lppd[i] /= size(β_samples)[1]
      lppd[i] = log(lppd[i])
    end

  return lppd
end

elppd_agess = elppd(X_pred, y_pred, x_AGESS[200000:end,1:P], exp.(x_AGESS[200000:end,2*P +2]))
elppd_HS = elppd(X_pred, y_pred, beta_samps_HS[:,200000:end]', sqrt.(sigma_samps_HS[200000:end]))
elppd_half_t = elppd(X_pred, y_pred, beta_samps_half_t[200000:end,:], sqrt.(sigma_samps_half_t[200000:end]))

save(string(dir ,"//Biscuit//Sim", i,".jld2"), Dict("x_AGESS" => x_AGESS, 
                                                    "beta_samps_HS" => beta_samps_HS,
                                                    "beta_samps_half_t" => beta_samps_half_t,
                                                    "sigma_samps_HS" => sigma_samps_HS,
                                                    "sigma_samps_half_t" => sigma_samps_half_t,
                                                    "elppd_agess" => elppd_agess1,
                                                    "elppd_HS" => elppd_HS,
                                                    "elppd_half_t" => elppd_half_t,
                                                    "AGESS_total_time" => AGESS_total_time,
                                                    "HS_time" => HS_time,
                                                    "half_t_time" => half_t_time))



#### Riboflavin
R"""
library(hdi)
data(riboflavin)
dim2 <- dim(riboflavin$x)[2]
corr <- rep(0, dim2)
for(i in 1:dim2){
  corr[i] <- cor(riboflavin$y, riboflavin$x[,i])
}
corr <- abs(corr)
ind <- order(corr)
X <- riboflavin$x[,ind[(dim2-499):dim2]]
X <- scale(X)
y_obs <- riboflavin$y - mean(riboflavin$y)

X_train <- X[1:50,]
y_train <- y_obs[1:50,]
X_test <- X[51:length(y_obs),]
y_test <- y_obs[51:length(y_obs),]

burnin <- 0
chain_length <- 400000
chain <- NA
time1 = Sys.time()
hs_chain <- horseshoe(y_train, X_train, method.tau = "halfCauchy", method.sigma = "Jeffreys", nmc = chain_length, burn = 0)
time_end = Sys.time() - time1
beta_samps_HS = hs_chain$BetaSamples
HS_time = time_end
sigma_samps_HS = hs_chain$Sigma2Samples
"""
@rget beta_samps_HS
@rget sigma_samps_HS

g1 = scatter(mean(beta_samps_HS[:,200000:end], dims = 2))

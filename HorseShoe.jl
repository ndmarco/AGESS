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
      @views lpdf += logpdf(Normal(0.0, exp(σ) * exp(τ) * exp(λ[i])), β[i])
  end
  return lpdf
end

function prior_λ(λ::AbstractVector{Y})::Float64 where {Y<:AbstractFloat}
  lpdf::Float64 = 0.0
  cauchy_d = Cauchy(0.0,1.0)
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
  lpdf += prior_β(β, τ, λ, σ) + prior_λ(λ) + logpdf(Cauchy(0.0,1.0), exp(τ)) + τ - σ

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
          β[i] = (rand() * 3 + 1) * (-1)^i
        end
    end

    Y_obs = rand(MultivariateNormal(X * β, σ_sq * diagm(ones(N))))

    return X, Y_obs, β
end


######################################
############### P > N  ###############
######################################


### Low correlation

N = 50
P = 100
Stan_β = plot()
Stan_β_0 = plot()
AGESS_β = plot()
AGESS_β_0 = plot()
GESS_β = plot()
GESS_β_0 = plot()
HS_β = plot()
HS_β_0 = plot()
ESS_PS = zeros(10, 3)
time_run = zeros(10, 3)

for i in 1:10
  X,Y, β = gen_data(N, P, ρ = 0.8, sparsity = 0.95, σ_sq = 1.0)
  data = Dict("N" => N, "P" => P, "X" => X, "y" => Y)
  
  
  ### STAN
  sm_HS = SampleModel("HorseShoe", model);
  t1 = time()
  rc = stan_sample(sm_HS; num_chains=1, num_warmups=10000, num_samples=100000, data);
  stan_time = time() - t1
  df = read_samples(sm_HS, :array);
  df = df[:,:,1]
  
  if i == 1
    Stan_β = plot(df[1:10:end, findall(β .!= 0)], legend = false, dpi = 300)
    hline!(β[findall(β .!= 0)], line = :dash, color =:black)
    Stan_β_0 = plot(df[1:10:end, findall(β .== 0)], legend = false, dpi = 300)
  end
  
  ### AGESS
  MCMC_iters = 200000
  x_AGESS = zeros(MCMC_iters, 2*P+2)
  Σ = diagm(ones(2*P+2))
  μ_AGESS = zeros(2*P+2)
  ph = zeros(N)
  @time AGESS_time = AGESS(x_AGESS, b -> log_posterior(b[1:P], b[2*P+1], b[(P+1):(2*P)], b[2*P+2], data["X"], data["y"]), 
        μ_AGESS, Σ, true, burnin = 0.5)
  if i == 1
    AGESS_β = plot(x_AGESS[100000:10:end, findall(β .!= 0)], legend = false, dpi = 300)
    hline!(β[findall(β .!= 0)], line = :dash, color =:black)
    AGESS_β_0 = plot(x_AGESS[100000:10:end, findall(β .== 0)], legend = false, dpi = 300)
  end
  ### GESS
  if i == 1
    x_GESS = zeros(MCMC_iters, 2*P+2)
    μ_GESS = zeros(2*P+2)
    GESS_time = GESS(x_GESS,  b -> log_posterior(b[1:P], b[2*P+1], b[(P+1):(2*P)], b[2*P+2], data["X"], data["y"]),
                    μ_GESS, Σ)
    
    
    GESS_β = plot(x_GESS[100000:10:end, findall(β .!= 0)], legend = false, dpi = 300)
    hline!(β[findall(β .!= 0)], line = :dash, color =:black)
    GESS_β_0 = plot(x_GESS[100000:10:end, findall(β .== 0)], legend = false, dpi = 300)
  end
  @rput X
  @rput Y
  R"""
  
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
  
  """
  @rget beta_samps_HS
  @rget HS_time
  if i == 1
    HS_β = plot(beta_samps_HS[findall(β .!= 0), 100000:10:end]', legend = false, dpi = 300)
    hline!(β[findall(β .!= 0)], line = :dash, color =:black)
    HS_β_0 = plot(beta_samps_HS[findall(β .== 0), 100000:10:end]', legend = false, dpi = 300)
  end
  x_AGESS1 = x_AGESS[100001:end,1:P]
  x_stan1 = df[:,1:P]
  x_HS1 = beta_samps_HS[1:P,100001:end]'
  @rput x_AGESS1
  @rput x_HS1
  @rput x_stan1
  R"""
  library(mcmcse)
  mats = rbind(x_AGESS1, x_HS1, x_stan1)
  sigma = mcse.multi(x_AGESS1)$cov
  ess_AGESS <- multiESS(mats, covmat = sigma) / 3
  sigma = mcse.multi(x_stan1)$cov
  ess_Stan <- multiESS(mats, covmat = sigma) / 3
  sigma = mcse.multi(x_HS1)$cov
  ess_HS <- multiESS(mats, covmat = sigma) / 3
  """
  @rget ess_Stan
  @rget ess_HS
  @rget ess_AGESS
  
  ESS_PS[i,1] = ess_AGESS / (AGESS_time[1])
  ESS_PS[i,2] = ess_Stan / (stan_time * (10/11))
  ESS_PS[i,3] = ess_HS / (HS_time * 0.5)
  time_run[i,1] = AGESS_time[1]
  time_run[i,2] = (stan_time * (10/11))
  time_run[i,3] = HS_time

  save(string(dir ,"//Sim", i,".jld2"), Dict("x_AGESS" => x_AGESS, 
                                             "x_Stan" => df,
                                             "x_HS" => beta_samps_HS,
                                             "stan_time" => stan_time,
                                             "AGESS_time" => AGESS_time[1],
                                             "HS_time" => HS_time,
                                             "ess_Stan" => ess_Stan,
                                             "ess_AGESS" => ess_AGESS,
                                             "ess_HS" => ess_HS,
                                             "β" => β,
                                             "X" => X,
                                             "Y" => Y))
end
time_run1 = deepcopy(time_run)
time_run1[:,1] .= time_run[:,2]
time_run1[:,2] .= time_run[:,1]
colors = [:green :orange :navy]
box1 = boxplot(["AGESS" "HMC" "HS"], ESS_PS, title = "Effective Sample Size (P = 202)", legend = false, color=[:green :orange :navy], markerstrokewidth=0,  yscale=:log10)
ylabel!("Effective Sample Size per Second")
box2 = boxplot(["AGESS" "HMC" "HS"], time_run, title = "Computational Time", legend = false, yscale=:log10, yticks = [10, 100, 1000],  ylims = [10, 1000], markerstrokewidth=0, color=[:green :orange :navy])
ylabel!("Seconds")

plot(GESS_β, AGESS_β, Stan_β, HS_β, box1, GESS_β_0, AGESS_β_0, Stan_β_0, HS_β_0, box2, layout = @layout([A B C D E; F G H I J]), margin= 5Plots.mm)
plot!(size = (2500, 1000))
savefig(string(dir ,"//SimulationResults.pdf"))

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
"""
@rget beta_samps_HS
@rget HS_time
@rget X
@rget y_obs
@rget X_pred
@rget y_pred
@rget sigma_samps_HS
##


P = size(X)[2]
MCMC_iters = 400000
x_AGESS = zeros(MCMC_iters, 2*P+2)
Σ = diagm(ones(2*P+2))
μ_AGESS = zeros(2*P+2)
t1 = time()
AGESS_time = AGESS(x_AGESS, b -> log_posterior(b[1:P], b[2*P+1], b[(P+1):(2*P)], b[2*P+2], X, y_obs), 
      μ_AGESS, Σ, true, burnin = 0.5)
AGESS_total_time = time() - t1


data = Dict("N" => 32, "P" => 700, "X" => X, "y" => y_obs)
  
  
### STAN
sm_HS = SampleModel("HorseShoe", model);
t1 = time()
rc = stan_sample(sm_HS; num_chains=1, num_warmups=10000, num_samples=400000, data);
stan_time = time() - t1
df = read_samples(sm_HS, :array);
df = df[:,:,1]


index_order = sortperm(abs.(mean(beta_samps_HS, dims = 2)), dims = 1)

g1 = scatter(mean(beta_samps_HS, dims = 2))
scatter!(median(beta_samps_HS2, dims = 2))
g1 = scatter(median(beta_samps_half_t1, dims = 1)')
scatter(median(beta_samps_half_t1, dims = 1)')
scatter!(median(x_AGESS[100000:end, 1:P], dims = 1)')

ind = sortperm(abs.(median(beta_samps_HS2, dims = 2)), dims = 1)

sortperm(abs.(mean(x_AGESS1[100000:end, 1:P], dims = 1)), dims = 2)


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

function RMSE(X::AbstractMatrix{Y}, Y_obs::AbstractVector{Y}, β_samples::AbstractMatrix{Y}) where {Y<:AbstractFloat}
  β = mean(β_samples, dims = 1)
  Y_pred = X * β'
  RMSE = sum((Y_obs - Y_pred).^2)

  return RMSE
end

elppd_agess = elppd(X_pred, y_pred, x_AGESS[100000:end,1:P], exp.(x_AGESS[100000:end,2*P +2]))
elppd_HS = elppd(X_pred, y_pred, beta_samps_HS[:,100000:end]', sqrt.(sigma_samps_HS[100000:end]))
elppd_stan = elppd(X_pred, y_pred, df[:,1:P], df[:,2*P +2])
sum(elppd_agess)
sum(elppd_HS)
sum(elppd_stan)

RMSE_agess = RMSE(X_pred, y_pred, x_AGESS[100000:end,1:P])
RMSE_HS = RMSE(X_pred, y_pred, beta_samps_HS[:,100000:end]')
RMSE_stan = RMSE(X_pred, y_pred, df[:,1:P])

g1 = scatter!(mean(beta_samps_HS[:,100000:end], dims = 2))
scatter!(mean(x_AGESS[100000:end, 1:P], dims = 1)')

AGESS_order = sortperm(abs.(mean(x_AGESS[100000:end, 1:P], dims = 1)), dims = 2)
plot(x_AGESS[100000:30:end,AGESS_order[P-9:P]], labels = false)
plot!(size = (500,300))


HS_order = sortperm(abs.(mean(beta_samps_HS[:, 100000:end], dims = 2)), dims = 1)
plot(beta_samps_HS[HS_order[P-9:P], 100000:30:end]', labels = false)
plot!(size = (500,300))

Stan_order = sortperm(abs.(mean(df[:, 1:P], dims = 1)), dims = 2)
plot(x_AGESS[1:30:end,Stan_order[P-9:P]], labels = false)
plot!(size = (500,300))


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
library(horseshoe)
data(riboflavin)
dim2 <- dim(riboflavin$x)[2]
X <-  riboflavin$x
y_obs <- riboflavin$y
set.seed(1)
inc_ind = sample(1:71, 50)
X_train <- X[inc_ind,]
y_train <- y_obs[inc_ind]
X_test <- X[-inc_ind,]
y_test <- y_obs[-inc_ind]
corr <- rep(0, dim2)
for(i in 1:dim2){
  corr[i] <- cor(y_train, X_train[,i])
}
corr <- abs(corr)
ind <- order(corr)
X_test <- X_test[,ind[(dim2-299):dim2]]
X_train <- X_train[,ind[(dim2-299):dim2]]


for(i in 1:ncol(X_train)){
  X_test[,i] <- (X_test[,i] - mean(X_train[,i])) / sd(X_train[,i])
  X_train[,i] <- (X_train[,i] - mean(X_train[,i])) / sd(X_train[,i])
}
y_test <- y_test - mean(y_train)
y_train <- y_train - mean(y_train)

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
@rget HS_time
@rget X_train
@rget X_test
@rget y_train
@rget y_test



g1 = scatter(mean(beta_samps_HS[:,100000:end], dims = 2))
scatter!(mean(x_AGESS[100000:end, 1:P], dims = 1)')


P = size(X_train)[2]
MCMC_iters = 400000
x_AGESS = zeros(MCMC_iters, 2*P+2)
Σ = diagm(ones(2*P+2))
μ_AGESS = zeros(2*P+2)
t1 = time()
AGESS_time = AGESS(x_AGESS, b -> log_posterior(b[1:P], b[2*P+1], b[(P+1):(2*P)], b[2*P+2], X_train, y_train), 
      μ_AGESS, Σ, true, burnin = 0.5)
AGESS_total_time = time() - t1

elppd_agess = elppd(X_test, y_test, x_AGESS[100000:end,1:P], exp.(x_AGESS[100000:end,2*P +2]))
elppd_HS = elppd(X_test, y_test, beta_samps_HS[:,100000:end]', sqrt.(sigma_samps_HS[100000:end]))
sum(elppd_agess)
sum(elppd_HS)
RMSE_agess = RMSE(X_test, y_test, x_AGESS[100000:end,1:P])
RMSE_HS = RMSE(X_test, y_test, beta_samps_HS[:,100000:end]')
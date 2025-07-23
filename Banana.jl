using StanBase
set_cmdstan_home!("/Users/ndm34/Projects/cmdstan")

using StanSample, DataFrames, Stan
include("AGESS.jl")
using LinearAlgebra, LogExpFunctions, Distributions, LinearAlgebra, JLD2, Random, StatsBase, RCall, StatsPlots, KernelDensity

model = "
data {
  int N;
  array[N] real y;
}

parameters {
  real theta1;
  real theta2;
}

model {
  for (i in 1:N)
   y[i] ~ normal(theta1 + theta2^2 ,1);
  theta1 ~ normal(0,1);
  theta2 ~ normal(0.5,1);
}
";

sm = SampleModel("banana", model);

data = Dict("N" => 100, "y" => randn(100) .+ 1);

t1 = time()
rc = stan_sample(sm; num_cpp_chains=1, num_chains=1, num_warmups=100000, num_samples=100000, data);
stan_time = time() - t1 

df = read_samples(sm, :array);


MCMC_iters = 200000

x_AGESS = zeros(MCMC_iters, 2)

Σ_I = diagm(ones(length(data["y"])))
ones_N = ones(length(data["y"]))
Σ =  diagm(ones(2))

AGESS_time = AGESS(x_AGESS, b -> logpdf(MvNormal((b[1] + b[2]^2) * ones_N, Σ_I), data["y"]), 
      c -> (logpdf(Normal(0, 1), c[1]) + logpdf(Normal(0.5, 1), c[2])), 
      [0.5, 0], Σ, true)

x_GESS = zeros(MCMC_iters, 2)
GESS_time = GESS(x_GESS,  b -> (logpdf(MvNormal((b[1] + b[2]^2) * ones_N, Σ_I), data["y"]) + logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0.5, 1), b[2])),
     [0, 0.5], Σ)

x_ESS = zeros(MCMC_iters, 2)
ESS_time = ESS(x_ESS,  b -> logpdf(MvNormal((b[1] + b[2]^2) * ones_N, Σ_I), data["y"]),
     [0, 0.5], Σ)
  
x_ARW = zeros(MCMC_iters, 2)
ARW_time = ARW(x_ARW, b -> logpdf(MvNormal((b[1] + b[2]^2) * ones_N, Σ_I), data["y"]), c -> (logpdf(Normal(0, 1), c[1]) + logpdf(Normal(0.5, 1), c[2])), 1000,
             0.1, [0, 0.5], Σ)




dens_AGESS = kde(x_GESS[100001:end,:])
plot(dens_AGESS)

scatter(x_AGESS[100001:end, 1], x_AGESS[100001:end, 2], alpha = 0.4, label = "AGESS", legend=:outerbottom)
scatter!(x_GESS[100001:end, 1], x_GESS[100001:end, 2] .+ 5, alpha = 0.4, label = "GESS")
scatter!(df[:, 1], df[:, 2] .+ 10, alpha = 0.4, label = "HMC")
scatter!(x_ESS[100001:end, 1], x_ESS[100001:end, 2] .+ 15, alpha = 0.4, label = "ESS")
scatter!(x_ARW[100001:end, 1], x_ARW[100001:end, 2] .+ 20, alpha = 0.4, label = "ARW")

savefig("/Users/ndm34/Projects/AGESS_Simulation/Banana/Standard_Banana.pdf")

x_AGESS1 = x_AGESS[100001:end,:]
x_GESS1 = x_GESS[100001:end,:]
x_ESS1 = x_ESS[100001:end,:]
x_ARW1 = x_ARW[100001:end,:]
x_STAN = df[:,:,1]

@rput x_AGESS1
@rput x_GESS1
@rput x_ESS1
@rput x_ARW1
@rput x_STAN

R"""
library(stableGR)
ess_ESS <- n.eff(x_ESS1)$n.eff
ess_GESS <- n.eff(x_GESS1)$n.eff
ess_AGESS <- n.eff(x_AGESS1)$n.eff
ess_ARW <- n.eff(x_ARW1)$n.eff
ess_HMC <- n.eff(x_STAN)$n.eff
"""

@rget ess_ESS
@rget ess_GESS
@rget ess_AGESS
@rget ess_ARW
@rget ess_HMC

ess_per_second_ESS_banana = ess_ESS / ESS_time
ess_per_second_GESS_banana = ess_GESS / GESS_time
ess_per_second_AGESS_banana = ess_AGESS / AGESS_time
ess_per_second_ARW_banana = ess_ARW / ARW_time
ess_per_second_HMC_banana = ess_HMC / (stan_time * 0.5)


####################
### Twin Bananas ###
####################

model = "
data {
  int N;
  array[N] real y;
}

parameters {
  real theta1;
  real theta2;
}

model {
  for (i in 1:N)
   y[i] ~ normal(0.1 * theta1^2 - 0.5 * theta2^4 - (10 * theta1 * theta2 ),10);
  theta1 ~ normal(0,1);
  theta2 ~ normal(0,1);
}
";

sm = SampleModel("Pair_bananas", model);

data = Dict("N" => 100, "y" => randn(100)* 10 .+ 100);

t1 = time()
rc = stan_sample(sm; num_cpp_chains=1, num_chains=1, num_warmups=250000, num_samples=250000, data);
stan_time = time() - t1 

df = read_samples(sm, :array);


MCMC_iters = 500000

x_AGESS = zeros(MCMC_iters, 2)

Σ_I = diagm(ones(length(data["y"])))
ones_N = ones(length(data["y"]))
Σ =  diagm(ones(2))

AGESS_time = AGESS(x_AGESS, b -> logpdf(MvNormal((0.1 * b[1]^2  -  0.5 * b[2]^4 - 10 * b[1] * b[2]) * ones_N, 100 * Σ_I), data["y"]), 
      c -> (logpdf(Normal(0, 1), c[1]) + logpdf(Normal(0, 1), c[2])), 
      [0.0, 0.0], Σ, true)

x_GESS = zeros(MCMC_iters, 2)
GESS_time = GESS(x_GESS,  b -> (logpdf(MvNormal((0.1 * b[1]^2  -  0.5 * b[2]^4 - 10 * b[1] * b[2]) * ones_N, 100 * Σ_I), data["y"]) + logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 1), b[2])),
     [0.0, 0.0], Σ)

x_ESS = zeros(MCMC_iters, 2)
ESS_time = ESS(x_ESS,  b -> logpdf(MvNormal((0.1 * b[1]^2  -  0.5 * b[2]^4 - 10 * b[1] * b[2]) * ones_N, 100 * Σ_I), data["y"]),
              [0.0, 0.0], Σ)
  
x_ARW = zeros(MCMC_iters, 2)
ARW_time = ARW(x_ARW, b -> logpdf(MvNormal((0.1 * b[1]^2  -  0.5 * b[2]^4 - 10 * b[1] * b[2]) * ones_N, 100 * Σ_I), data["y"]), c -> (logpdf(Normal(0, 1), c[1]) + logpdf(Normal(0, 1), c[2])), 1000,
             0.1, [0.0, 0.0], Σ)


scatter(x_AGESS[250001:end, 1], x_AGESS[250001:end, 2], alpha = 0.4, label = "AGESS", legend=:outerbottom)
scatter!(x_GESS[250001:end, 1], x_GESS[250001:end, 2] .+ 5, alpha = 0.4, label = "GESS")
scatter!(df[:, 1], df[:, 2] .+ 10, alpha = 0.4, label = "HMC")
scatter!(x_ESS[250001:end, 1], x_ESS[250001:end, 2] .+ 15, alpha = 0.4, label = "ESS")
scatter!(x_ARW[250001:end, 1], x_ARW[250001:end, 2] .+ 20, alpha = 0.4, label = "ARW")

savefig("/Users/ndm34/Projects/AGESS_Simulation/Banana/Pair_Bananas.pdf")

x_AGESS1 = x_AGESS[250001:end,:]
x_GESS1 = x_GESS[250001:end,:]
x_ESS1 = x_ESS[250001:end,:]
x_ARW1 = x_ARW[250001:end,:]
x_STAN = df[:,:,1]

@rput x_AGESS1
@rput x_GESS1
@rput x_ESS1
@rput x_ARW1
@rput x_STAN

R"""
library(stableGR)
ess_ESS <- n.eff(x_ESS1)$n.eff
ess_GESS <- n.eff(x_GESS1)$n.eff
ess_AGESS <- n.eff(x_AGESS1)$n.eff
ess_ARW <- n.eff(x_ARW1)$n.eff
ess_HMC <- n.eff(x_STAN)$n.eff
"""

@rget ess_ESS
@rget ess_GESS
@rget ess_AGESS
@rget ess_ARW
@rget ess_HMC

ess_per_second_ESS_banana_pair = ess_ESS / ESS_time
ess_per_second_GESS_banana_pair = ess_GESS / GESS_time
ess_per_second_AGESS_banana_pair = ess_AGESS / AGESS_time
ess_per_second_ARW_banana_pair = ess_ARW / ARW_time
ess_per_second_HMC_banana_pair = ess_HMC / (stan_time * 0.5)



##########################
### Spoon Distribution ###
##########################

model = "
data {
  int N;
  array[N] real y;
}

parameters {
  real theta1;
  real theta2;
}

model {
  for (i in 1:N)
   y[i] ~ normal((exp(0.5 * theta1) - (0.1 * (exp(theta2 + 3))^4)), sqrt(1 + abs(theta1)));
  theta1 ~ normal(0.5,1);
  theta2 ~ normal(0,1);
}
";

sm = SampleModel("spoon", model);

data = Dict("N" => 100, "y" => randn(100) .+ 10);

t1 = time()
rc = stan_sample(sm; num_cpp_chains=1, num_chains=1, num_warmups=100000, num_samples=100000, data);
stan_time = time() - t1 

df = read_samples(sm, :array);


MCMC_iters = 200000

x_AGESS = zeros(MCMC_iters, 2)

Σ_I = diagm(ones(length(data["y"])))
ones_N = ones(length(data["y"]))
Σ =  diagm(ones(2))

AGESS_time = AGESS(x_AGESS, b -> logpdf(MvNormal((exp(0.5 * b[1]) - (0.1 * (exp(b[2] + 3))^4)) * ones_N, Σ_I  + Σ_I .* abs(b[1])), data["y"]), 
      c -> (logpdf(Normal(0.5, 1), c[1]) + logpdf(Normal(0, 1), c[2])), 
      [0.5, 0], Σ, true)

x_GESS = zeros(MCMC_iters, 2)
GESS_time = GESS(x_GESS,  b -> (logpdf(MvNormal((exp(0.5 * b[1]) - (0.1 * (exp(b[2] + 3))^4)) * ones_N, Σ_I  + Σ_I .* abs(b[1])), data["y"]) + logpdf(Normal(0.5, 1), b[1]) + logpdf(Normal(0, 1), b[2])),
     [0.5, 0], Σ)

x_ESS = zeros(MCMC_iters, 2)
ESS_time = ESS(x_ESS,  b -> logpdf(MvNormal((exp(0.5 * b[1]) - (0.1 * (exp(b[2] + 3))^4)) * ones_N, Σ_I  + Σ_I .* abs(b[1])), data["y"]),
     [0.5, 0], Σ)
  
x_ARW = zeros(MCMC_iters, 2)
ARW_time = ARW(x_ARW, b -> logpdf(MvNormal((exp(0.5 * b[1]) - (0.1 * (exp(b[2] + 3))^4)) * ones_N, Σ_I  + Σ_I .* abs(b[1])), data["y"]), c -> (logpdf(Normal(0.5, 1), c[1]) + logpdf(Normal(0, 1), c[2])), 1000,
             0.1, [0.5, 0], Σ)




dens_AGESS = kde(x_AGESS[250001:end,:])
plot(dens_AGESS)

scatter(x_AGESS[100001:end, 1], x_AGESS[100001:end, 2], alpha = 0.4, label = "AGESS", legend=:outerbottom)
scatter!(x_GESS[100001:end, 1], x_GESS[100001:end, 2] .+ 5, alpha = 0.4, label = "GESS")
scatter!(df[:, 1], df[:, 2] .+ 10, alpha = 0.4, label = "HMC")
scatter!(x_ESS[100001:end, 1], x_ESS[100001:end, 2] .+ 15, alpha = 0.4, label = "ESS")
scatter!(x_ARW[100001:end, 1], x_ARW[100001:end, 2] .+ 20, alpha = 0.4, label = "ARW")

savefig("/Users/ndm34/Projects/AGESS_Simulation/Banana/Spoon2.pdf")

x_AGESS1 = x_AGESS[100001:end,:]
x_GESS1 = x_GESS[100001:end,:]
x_ESS1 = x_ESS[100001:end,:]
x_ARW1 = x_ARW[100001:end,:]
x_STAN = df[:,:,1]

@rput x_AGESS1
@rput x_GESS1
@rput x_ESS1
@rput x_ARW1
@rput x_STAN

R"""
library(stableGR)
ess_ESS <- n.eff(x_ESS1)$n.eff
ess_GESS <- n.eff(x_GESS1)$n.eff
ess_AGESS <- n.eff(x_AGESS1)$n.eff
ess_ARW <- n.eff(x_ARW1)$n.eff
ess_HMC <- n.eff(x_STAN)$n.eff
"""

@rget ess_ESS
@rget ess_GESS
@rget ess_AGESS
@rget ess_ARW
@rget ess_HMC

ess_per_second_ESS_spoon = ess_ESS / ESS_time
ess_per_second_GESS_spoon = ess_GESS / GESS_time
ess_per_second_AGESS_spoon = ess_AGESS / AGESS_time
ess_per_second_ARW_spoon = ess_ARW / ARW_time
ess_per_second_HMC_spoon = ess_HMC / (stan_time * 0.5)

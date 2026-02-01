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
  lpdf += prior_β(β, τ, λ, σ) + prior_λ(λ) + logpdf(Cauchy(0.0,1.0), exp(τ)) + τ

  return lpdf
end

function gen_data_AR1(N::T, P::T; sparsity::Y = 0.8, ρ::Y = 0.2, σ_sq::Y = 1.0) where {Y<:AbstractFloat, T<:Integer}
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
    if sum(β) == 0
      β[1] = (rand() * 3 + 1) * (-1)^i
    end

    Y_obs = rand(MultivariateNormal(X * β, σ_sq * diagm(ones(N))))

    return X, Y_obs, β
end


####################################
############### AR1  ###############
####################################


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
ESS_PS = zeros(25, 3)
time_run = zeros(25, 3)

for i in 1:25
  if !isfile(string(dir ,"//Sim", i,".jld2"))
    Random.seed!(i)
    X,Y, β = gen_data_AR1(N, P, ρ = 0.8, sparsity = 0.95, σ_sq = 1.0)
    data = Dict("N" => N, "P" => P, "X" => X, "y" => Y)
    
    
    ### STAN
    sm_HS = SampleModel("HorseShoe", model);
    t1 = time()
    rc = stan_sample(sm_HS; num_chains=1, num_warmups=10000, num_samples=200000, data);
    stan_time = time() - t1
    df = read_samples(sm_HS, :array);
    df = df[:,:,1]
    
    if i == 16
      Stan_β = plot(df[1:10:end, findall(β .!= 0)], legend = false, dpi = 300, ylims = [-3, 6.5])
      hline!(β[findall(β .!= 0)], line = :dash, color =:black)
      Stan_β_0 = plot(df[1:10:end, findall(β .== 0)], legend = false, dpi = 300, ylims = [-1.5, 1.5])
    end
    
    ### AGESS
    MCMC_iters = 1500000
    x_AGESS = zeros(MCMC_iters, 2*P+2)
    Σ = diagm(ones(2*P+2))
    μ_AGESS = zeros(2*P+2)
    ph = zeros(N)
    t1 = time()
    AGESS_time = AGESS(x_AGESS, b -> log_posterior(b[1:P], b[2*P+1], b[(P+1):(2*P)], b[2*P+2], data["X"], data["y"]), 
          μ_AGESS, Σ, true, burnin = 1/6)
    AGESS_total_time = time() - t1
    if i == 16
      AGESS_β = plot(x_AGESS[250001:10:end, findall(β .!= 0)], legend = false, dpi = 300,  ylims=[-3, 6.5])
      hline!(β[findall(β .!= 0)], line = :dash, color =:black)
      AGESS_β_0 = plot(x_AGESS[250001:10:end, findall(β .== 0)], legend = false, dpi = 300, ylims = [-1.5, 1.5])
    end

    ### GESS
    if i == 16
      x_GESS = zeros(MCMC_iters, 2*P+2)
      μ_GESS = zeros(2*P+2)
      GESS_time = GESS(x_GESS,  b -> log_posterior(b[1:P], b[2*P+1], b[(P+1):(2*P)], b[2*P+2], data["X"], data["y"]),
                      μ_GESS, Σ)
      
      
      GESS_β = plot(x_GESS[250000:10:end, findall(β .!= 0)], legend = false, dpi = 300, ylims=[-3, 6.5])
      hline!(β[findall(β .!= 0)], line = :dash, color =:black)
      GESS_β_0 = plot(x_GESS[250000:10:end, findall(β .== 0)], legend = false, dpi = 300, ylims = [-1.5, 1.5])
    end
    @rput X
    @rput Y
    R"""
    library(stableGR)
    library(horseshoe)
    time1 = Sys.time()
    hs_chain <- horseshoe(Y, X, method.tau = "halfCauchy", method.sigma = "Jeffreys", burn=1e5, nmc=1e5)
    time_end = Sys.time() - time1


    beta_samps_HS <- hs_chain$BetaSamples
    HS_time = as.numeric(time_end)
    """

    @rget beta_samps_HS
    @rget HS_time
    if i == 16
      HS_β = plot(beta_samps_HS[findall(β .!= 0), 1:10:end]', legend = false, dpi = 300, ylims=[-3, 6.5])
      hline!(β[findall(β .!= 0)], line = :dash, color =:black)
      HS_β_0 = plot(beta_samps_HS[findall(β .== 0), 1:10:end]', legend = false, dpi = 300, ylims = [-1.5, 1.5])
    end
    x_AGESS1 = x_AGESS[250001:end,1:P]
    x_stan1 = df[:,1:P]
    x_HS1 = beta_samps_HS'
    @rput x_AGESS1
    @rput x_HS1
    @rput x_stan1
    R"""
    library(stableGR)
    library(stableGR)
    ess_AGESS <- -1
    ess_HS <- -1
    ess_Stan <- -1
    rg_AGESS <- NA
    rg_Stan <- NA
    rg_HS <- NA
    num_iters_AGESS <- 0
    num_iters_Stan <- 0
    try(rg_AGESS <- n.eff(x_AGESS1, epsilon = 0.1))
    try(rg_Stan <- n.eff(x_stan1, epsilon = 0.1))
    try(rg_HS <- n.eff(x_HS1, epsilon = 0.1))

    try(if(rg_AGESS$converged == TRUE){
          ess_AGESS <- rg_AGESS$n.eff
      }else{num_iters_AGESS <- rg_AGESS$n.target})
      
    try(if(rg_Stan$converged == TRUE){
        ess_Stan <- rg_Stan$n.eff
      }else{
        num_iters_Stan <- rg_Stan$n.target
      })
    
    try(if(rg_HS$converged == TRUE){
        ess_HS <- rg_HS$n.eff
      })
    """
    @rget ess_Stan
    @rget ess_HS
    @rget ess_AGESS
    
    ESS_PS[i,1] = ess_AGESS / (AGESS_time[1])
    ESS_PS[i,2] = ess_Stan / (stan_time * (20/21))
    ESS_PS[i,3] = ess_HS / (HS_time * 0.5)
    time_run[i,1] = AGESS_total_time
    time_run[i,2] = stan_time
    time_run[i,3] = HS_time 

    save(string(dir ,"//Sim", i,".jld2"), Dict("x_AGESS" => x_AGESS, 
                                              "x_Stan" => df,
                                              "x_HS" => beta_samps_HS,
                                              "stan_time" => stan_time,
                                              "AGESS_time" => AGESS_time[1],
                                              "AGESS_total_Time" => AGESS_total_time,
                                              "HS_time" => HS_time,
                                              "ess_Stan" => ess_Stan,
                                              "ess_AGESS" => ess_AGESS,
                                              "ess_HS" => ess_HS,
                                              "β" => β,
                                              "X" => X,
                                              "Y" => Y))
  else
    sim = load(string(dir ,"//Sim", i,".jld2")) 

    ESS_PS[i,1] = sim["ess_AGESS"] / (sim["AGESS_time"])
    ESS_PS[i,2] = sim["ess_Stan"] / (sim["stan_time"] * (20/21))
    ESS_PS[i,3] = sim["ess_HS"] / (sim["HS_time"] * 0.5)
    time_run[i,1] = sim["AGESS_total_time"]
    time_run[i,2] = sim["stan_time"]
    time_run[i,3] = sim["HS_time"] 
  end
end


colors = [:green :orange :navy]
box1 = boxplot(["AGESS" "HMC" "HS"], ESS_PS, title = "Effective Sample Size (P = 202)", legend = false, color=[:green :orange :navy], markerstrokewidth=0,  yscale=:log10, yticks = [1, 10, 100, 1000, 10000],  ylims = [0.5, 10000])
ylabel!("Effective Sample Size per Second")
box2 = boxplot(["AGESS" "HMC" "HS"], time_run, title = "Total Computational Time", legend = false,  yscale=:log10 ,ylim = [10,5000], yticks = [10, 100, 1000], markerstrokewidth=0, color=[:green :orange :navy])
ylabel!("Seconds")

plot(GESS_β, AGESS_β, Stan_β, HS_β, box1, GESS_β_0, AGESS_β_0, Stan_β_0, HS_β_0, box2, layout = @layout([A B C D E; F G H I J]), margin= 5Plots.mm)
plot!(size = (2500, 1000))
savefig(string(dir ,"//SimulationResults_AR1.pdf"))
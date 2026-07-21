using StanBase
# Set path to STAN
set_cmdstan_home!("C:\\cmdstan")
using StanSample, DataFrames, Stan
include("AGESS.jl")
using LinearAlgebra, LogExpFunctions, Distributions, LinearAlgebra, JLD2, Random, StatsBase, RCall, StatsPlots, KernelDensity

dir = ".\\Horseshoe"
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

function stan_sampling_time(sm)
    csv_files = filter(f -> endswith(f, ".csv"), readdir(sm.tmpdir, join=true))
    sampling_time = 0.0
    for f in csv_files
        for line in eachline(f)
            if occursin("(Sampling)", line)
                m = match(r"([\d.]+)\s+seconds\s+\(Sampling\)", line)
                sampling_time += parse(Float64, m.captures[1])
            end
        end
    end
    return sampling_time
end

function summarize_column(v::AbstractVector{<:Real})
    return (mean = mean(v), std = std(v), median = median(v),
            q10 = quantile(v, 0.10), q90 = quantile(v, 0.90))
end
 
function write_ess_summary_table(filepath::AbstractString,
                                  ESS_per_second::AbstractMatrix{<:Real},
                                  ESS_per_iter::AbstractMatrix{<:Real},
                                  ESS_per_likelihood::AbstractMatrix{<:Real},
                                  total_time::AbstractMatrix{<:Real};
                                  method_names = ["AGESS", "STAN", "HS"])
    metrics = [
        ("ESS per Second",               ESS_per_second),
        ("ESS per Iteration",            ESS_per_iter),
        ("ESS per Likelihood Evaluation", ESS_per_likelihood),
        ("Total Computational Time ", total_time),
    ]
 
    open(filepath, "w") do io
        println(io, "Summary Statistics: Deep GP (n = ", size(ESS_per_second, 1), " reps)")
        println(io, "="^70)
        println(io)
        for (metric_name, M) in metrics
            println(io, metric_name)
            println(io, "-"^70)
            @printf(io, "%-10s %20s %24s\n", "Method", "Mean ± SD", "Median [Q10, Q90]")
            for (j, name) in enumerate(method_names)
                s = summarize_column(view(M, :, j))
                mean_sd_str = @sprintf("%.4g ± %.4g", s.mean, s.std)
                median_iqr_str = @sprintf("%.4g [%.4g, %.4g]", s.median, s.q10, s.q90)
                @printf(io, "%-10s %20s %24s\n", name, mean_sd_str, median_iqr_str)
            end
            println(io)
        end
    end
 
    println("Summary table written to: ", filepath)
end

function sci_format(v::Real)
    v == 0 && return "0"
    s = @sprintf("%.1e", v)
    mantissa, expstr = split(s, 'e')
    e = parse(Int, expstr)
    mantissa = replace(mantissa, r"\.0$" => "")
    return string(mantissa, "e", e)
end

function trace_plot(samples::AbstractMatrix{<:Real}, β_true::Union{AbstractVector{<:Real}, Nothing}, ylims;
                     title::AbstractString = "", ylabel::AbstractString = "", thin::Int = 10, show_xlabel::Bool = true)
    x = thin .* (1:size(samples, 1))
    p = plot(x, samples, legend = false, ylims = ylims, title = title, linewidth = 1,
             xlabel = show_xlabel ? "MCMC Iterations" : "", xformatter = sci_format,
             fontfamily = "Computer Modern", titlefontsize = 22, guidefontsize = 18, tickfontsize = 15,
             framestyle = :axes, grid = false)
    if β_true !== nothing
        hline!(β_true, line = :dash, color = :black, linewidth = 2)
    end
    if !isempty(ylabel)
        ylabel!(ylabel)
    end
    return p
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
ESS_per_second = zeros(25, 3)
ESS_per_iteration = zeros(25, 3)
ESS_per_likelihood = zeros(25, 3)
total_evals = zeros(25, 3)
total_time = zeros(25, 3)

for i in 1:25
  if !isfile(string(dir ,"//Sim", i,".jld2"))
    Random.seed!(i)
    X,Y, β = gen_data_AR1(N, P, ρ = 0.8, sparsity = 0.95, σ_sq = 1.0)
    data = Dict("N" => N, "P" => P, "X" => X, "y" => Y)
    
    
    ### STAN
    sm_HS = SampleModel("HorseShoe", model);
    t1 = time()
    rc = stan_sample(sm_HS; num_chains=1, num_warmups=10_000, num_samples=200_000, data);
    stan_total_time = time() - t1
    df = read_samples(sm_HS, :array);
    df = df[:,:,1]
    
    df_diag = read_samples(sm_HS, :dataframe; include_internals=true)

    # Sum the number of leapfrog steps (which will count the number of gradient evaluations)
    # Add 1 per iteration to initialize the Hamiltonian in each step
    total_evals[i,2] = sum(df_diag.n_leapfrog__ .+ 1)

    stan_time = stan_sampling_time(sm_HS)
        
    if i == 6
      Stan_β = trace_plot(df[1:10:end, findall(β .!= 0)], β[findall(β .!= 0)], (-5, 5); title = "HMC", ylabel = "Nonzero β", show_xlabel = false)
      Stan_β_0 = trace_plot(df[1:10:end, findall(β .== 0)], nothing, (-1.5, 1.5); title = "HMC", ylabel = "Zero β")
    end

    ### AGESS
    MCMC_iters = 1_500_000
    x_AGESS = zeros(MCMC_iters, 2*P+2)
    Σ = diagm(ones(2*P+2))
    μ_AGESS = zeros(2*P+2)
    ph = zeros(N)
    t1 = time()
    AGESS_time, Σ_adapt, μ_adapt, total_evals[i,1] = AGESS(x_AGESS, b -> log_posterior(b[1:P], b[2*P+1], b[(P+1):(2*P)], b[2*P+2], data["X"], data["y"]),
          μ_AGESS, Σ, true, burnin = 1/6)
    AGESS_total_time = time() - t1
    if i == 16
      AGESS_β = trace_plot(x_AGESS[250_001:10:end, findall(β .!= 0)], β[findall(β .!= 0)], (-5, 5); title = "AGESS", ylabel = "Nonzero β", show_xlabel = false)
      AGESS_β_0 = trace_plot(x_AGESS[250_001:10:end, findall(β .== 0)], nothing, (-1.5, 1.5); title = "AGESS", ylabel = "Zero β")
    end

    ### GESS
    if i == 6
      x_GESS = zeros(MCMC_iters, 2*P+2)
      μ_GESS = zeros(2*P+2)
      GESS_time, total_evals_GESS = GESS(x_GESS,  b -> log_posterior(b[1:P], b[2*P+1], b[(P+1):(2*P)], b[2*P+2], data["X"], data["y"]),
                      μ_GESS, Σ)


      GESS_β = trace_plot(x_GESS[250_001:10:end, findall(β .!= 0)], β[findall(β .!= 0)], (-5, 5); title = "GESS", ylabel = "Nonzero β", show_xlabel = false)
      GESS_β_0 = trace_plot(x_GESS[250_001:10:end, findall(β .== 0)], nothing, (-1.5, 1.5); title = "GESS", ylabel = "Zero β")
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
    if i == 6
      HS_β = trace_plot(beta_samps_HS[findall(β .!= 0), 1:10:end]', β[findall(β .!= 0)], (-5, 5); title = "HS", ylabel = "Nonzero β", show_xlabel = false)
      HS_β_0 = trace_plot(beta_samps_HS[findall(β .== 0), 1:10:end]', nothing, (-1.5, 1.5); title = "HS", ylabel = "Zero β")
    end
    x_AGESS1 = x_AGESS[250_001:end,1:P]
    x_stan1 = df[:,1:P]
    x_HS1 = beta_samps_HS'
    @rput x_AGESS1
    @rput x_HS1
    @rput x_stan1
    R"""
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
    
    ESS_per_second[i,1] = ess_AGESS / (AGESS_time)
    ESS_per_second[i,2] = ess_Stan / (stan_time)
    ESS_per_second[i,3] = ess_HS / (HS_time * 0.5)

    ESS_per_iteration[i,1] = ess_AGESS / 1_250_000
    ESS_per_iteration[i,2] = ess_Stan / 200_000
    ESS_per_iteration[i,3] = ess_HS / 1e5

    ESS_per_likelihood[i, 1] = ess_AGESS / (total_evals[i,1])
    ESS_per_likelihood[i, 2] = ess_Stan / (total_evals[i,2])

    total_time[i,1] = AGESS_total_time
    total_time[i,2] = stan_time
    total_time[i,3] = HS_time 

    save(string(dir ,"//Sim", i,".jld2"), Dict("x_AGESS" => x_AGESS, 
                                              "x_Stan" => df,
                                              "x_HS" => beta_samps_HS,
                                              "stan_time" => stan_time,
                                              "stan_total_time" => stan_total_time,
                                              "AGESS_time" => AGESS_time,
                                              "AGESS_total_time" => AGESS_total_time,
                                              "HS_time" => HS_time,
                                              "ess_Stan" => ess_Stan,
                                              "ess_AGESS" => ess_AGESS,
                                              "ess_HS" => ess_HS,
                                              "stan_total_likelihood" => total_evals[i,2],
                                              "AGESS_total_likelihood" => total_evals[i,1],
                                              "β" => β,
                                              "X" => X,
                                              "Y" => Y))
  else
    sim = load(string(dir ,"//Sim", i,".jld2")) 

    ESS_per_second[i,1] = sim["ess_AGESS"] / (sim["AGESS_time"])
    ESS_per_second[i,2] = sim["ess_Stan"] / (sim["stan_time"] * (20/21))
    ESS_per_second[i,3] = sim["ess_HS"] / (sim["HS_time"] * 0.5)

    ESS_per_iteration[i,1] = sim["ess_AGESS"] / 1250000
    ESS_per_iteration[i,2] = sim["ess_Stan"] / 200000
    ESS_per_iteration[i,3] = sim["ess_HS"] / 1e5

    ESS_per_likelihood[i, 1] = sim["ess_AGESS"] / (sim["AGESS_total_likelihood"])
    ESS_per_likelihood[i, 2] = sim["ess_Stan"] / (sim["stan_total_likelihood"])

    total_time[i,1] = sim["AGESS_total_Time"]
    total_time[i,2] = sim["stan_time"]
    total_time[i,3] = sim["HS_time"] 

    if i == 6
      β = sim["β"]
      df = sim["x_Stan"]
      Stan_β = trace_plot(df[1:10:end, findall(β .!= 0)], β[findall(β .!= 0)], (-5, 5); title = "HMC", ylabel = "", show_xlabel = false)
      Stan_β_0 = trace_plot(df[1:10:end, findall(β .== 0)], nothing, (-1.5, 1.5); title = "HMC", ylabel = "")

      x_AGESS = sim["x_AGESS"]
      AGESS_β = trace_plot(x_AGESS[250_001:10:end, findall(β .!= 0)], β[findall(β .!= 0)], (-5, 5); title = "AGESS", ylabel = "", show_xlabel = false)
      AGESS_β_0 = trace_plot(x_AGESS[250_001:10:end, findall(β .== 0)], nothing, (-1.5, 1.5); title = "AGESS", ylabel = "")

      beta_samps_HS = sim["x_HS"]
      HS_β = trace_plot(beta_samps_HS[findall(β .!= 0), 1:10:end]', β[findall(β .!= 0)], (-5, 5); title = "HS", ylabel = "", show_xlabel = false)
      HS_β_0 = trace_plot(beta_samps_HS[findall(β .== 0), 1:10:end]', nothing, (-1.5, 1.5); title = "HS", ylabel = "")

      ### Run GESS
      MCMC_iters = 1500000
      data = Dict("N" => N, "P" => P, "X" => sim["X"], "y" => sim["Y"])
      x_GESS = zeros(MCMC_iters, 2*P+2)
      μ_GESS = zeros(2*P+2)
      Σ = diagm(ones(2*P+2))
      GESS_time, total_evals_GESS = GESS(x_GESS,  b -> log_posterior(b[1:P], b[2*P+1], b[(P+1):(2*P)], b[2*P+2], data["X"], data["y"]),
                      μ_GESS, Σ)


      GESS_β = trace_plot(x_GESS[250_001:10:end, findall(β .!= 0)], β[findall(β .!= 0)], (-5, 5); title = "GESS", ylabel = "Nonzero β", show_xlabel = false)
      GESS_β_0 = trace_plot(x_GESS[250_001:10:end, findall(β .== 0)], nothing, (-1.5, 1.5); title = "GESS", ylabel = "Zero β")
    end
  end
end


plot(GESS_β, AGESS_β, Stan_β, HS_β, GESS_β_0, AGESS_β_0, Stan_β_0, HS_β_0,
     layout = @layout([A B C D; E F G H]),
     left_margin = 15Plots.mm, right_margin = 5Plots.mm,
     top_margin = 5Plots.mm, bottom_margin = 15Plots.mm,
     tickfontsize = 15)
plot!(size = (2100, 1050), dpi = 300)
savefig(string(dir ,"//SimulationResults_AR1.png"))

write_ess_summary_table(string(dir, "//summary_stats.txt"),
                         ESS_per_second, ESS_per_iteration, ESS_per_likelihood, total_time)
using StanBase
#set_cmdstan_home!("/Users/ndm34/Projects/cmdstan")
set_cmdstan_home!("C:\\Users\\ndmar\\Projects\\cmdstan")
using StanSample, DataFrames, Stan
include("AGESS.jl")
using LinearAlgebra, LogExpFunctions, Distributions, LinearAlgebra, JLD2, Random, StatsBase, RCall, StatsPlots, KernelDensity, Trapz


dir = "C:\\Users\\ndmar\\Projects\\AGESS_Simulation\\Banana"

model = "
data {
  int N;
  array[N] real y;
  real mu1;
  real mu2;
}

parameters {
  real theta1;
  real theta2;
}

model {
  for (i in 1:N){
    y[i] ~ normal((theta1 - mu1) + (theta2 - mu2)^2 ,1);
  }
  theta1 ~ normal(0,2);
  theta2 ~ normal(0.5,2);
}
";

## Create Directories
if !isdir(string(dir ,"//Banana"))
    mkdir(string(dir ,"//Banana"))
end
if !isdir(string(dir ,"//Twin_Bananas"))
    mkdir(string(dir ,"//Twin_Bananas"))
end


sm_banana = SampleModel("banana", model);

MCMC_iters = 200000

ESS_per_second_banana = zeros(100, 3)
colors = [:red :blue :green :orange :purple]
KL_dist_banana = zeros(100, 5)

mean_norm = zeros(100)
p1 = scatter()
p2 = scatter()
p3 = scatter()
p4 = scatter()
p5 = scatter()
p_density = scatter()

Random.seed!(123)
for i in 1:100
  if !isfile(string(dir ,"//Banana//Sim", i,".jld2"))
    μ1 = randn() * 3
    μ2 = randn() * 3
    data = Dict("N" => 100, "y" => randn(100) .+ 1, "mu1" => μ1, "mu2" => μ2);

    t1 = time()
    rc = stan_sample(sm_banana; num_cpp_chains=1, num_chains=1, num_warmups=100000, num_samples=100000, data);
    stan_time = time() - t1 

    df = read_samples(sm_banana, :array);


    x_AGESS = zeros(MCMC_iters, 2)

    Σ_I = diagm(ones(length(data["y"])))
    ones_N = ones(length(data["y"]))
    Σ =  diagm(ones(2) * 4)

    mean_norm[i] = (μ1^2 + μ2^2)^0.5
    
    AGESS_time, Σ_adapt = AGESS(x_AGESS, b -> (logpdf(MvNormal(((b[1] - μ1) + (b[2] - μ2)^2) * ones_N, Σ_I), data["y"]) + logpdf(Normal(0, 2), b[1]) + logpdf(Normal(0.5, 2), b[2])), 
          [0, 0.5], Σ, true)
    
    x_GESS = zeros(MCMC_iters, 2)
    GESS_time = GESS(x_GESS,  b -> (logpdf(MvNormal(((b[1] - μ1) + (b[2] - μ2)^2)  * ones_N, Σ_I), data["y"]) + logpdf(Normal(0, 2), b[1]) + logpdf(Normal(0.5, 2), b[2])),
        [0, 0.5], Σ)
    
    x_ESS = zeros(MCMC_iters, 2)
    ESS_time = ESS(x_ESS,  b -> logpdf(MvNormal(((b[1] - μ1) + (b[2] - μ2)^2)  * ones_N, Σ_I), data["y"]),
        [0, 0.5], Σ)
      
    x_ARW = zeros(MCMC_iters, 2)
    ARW_time = ARW(x_ARW, b -> logpdf(MvNormal(((b[1] - μ1) + (b[2] - μ2)^2)  * ones_N, Σ_I), data["y"]), c -> (logpdf(Normal(0, 2), c[1]) + logpdf(Normal(0.5, 2), c[2])), 1000,
                0.1, [0, 0.5], Σ)
    
    x_min = minimum([x_ESS[100001:end, 1] x_GESS[100001:end, 1] x_AGESS[100001:end, 1] df[:, 1] x_ARW[100001:end, 1]])
    x_max = maximum([x_ESS[100001:end, 1] x_GESS[100001:end, 1] x_AGESS[100001:end, 1] df[:, 1] x_ARW[100001:end, 1]])
    y_min = minimum([x_ESS[100001:end, 2] x_GESS[100001:end, 2] x_AGESS[100001:end, 2] df[:, 2] x_ARW[100001:end, 2]])
    y_max = maximum([x_ESS[100001:end, 2] x_GESS[100001:end, 2] x_AGESS[100001:end, 2] df[:, 2] x_ARW[100001:end, 2]])
    if i == 1
      p1 = scatter(x_ESS[100001:end, 1], x_ESS[100001:end, 2], alpha = 0.4, color = colors[1], legend = false, ylim = (y_min, y_max), xlim = (x_min, x_max), markerstrokewidth=0)
      p2 = scatter(x_GESS[100001:end, 1], x_GESS[100001:end, 2], alpha = 0.4, color = colors[2], legend = false, ylim = (y_min, y_max), xlim = (x_min, x_max), markerstrokewidth=0)
      p3 = scatter(x_AGESS[100001:end, 1], x_AGESS[100001:end, 2], alpha = 0.4, color = colors[3], legend = false, ylim = (y_min, y_max), xlim = (x_min, x_max), markerstrokewidth=0)
      mean_ell = [mean(x_AGESS[100001:end, 1]) mean(x_AGESS[100001:end, 2])]
      p3 = covellipse!(mean_ell[1,:], Σ_adapt, level = 0.68)
      p4 = scatter(df[:, 1], df[:, 2], alpha = 0.4, color = colors[4], legend = false, ylim = (y_min, y_max), xlim = (x_min, x_max), markerstrokewidth=0)
      p5 = scatter(x_ARW[100001:end, 1], x_ARW[100001:end, 2], alpha = 0.4, color = colors[5], legend = false, ylim = (y_min, y_max), xlim = (x_min, x_max), markerstrokewidth=0)
    end
    
    
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

    x_grid = range(x_min, x_max, 1000)
    y_grid = range(y_min, y_max, 1000)

    M = [(logpdf(MvNormal(((x - μ1) + (y - μ2)^2)  * ones_N, Σ_I), data["y"]) + logpdf(Normal(0, 2), x) + logpdf(Normal(0.5, 2), y)) for x = x_grid, y = y_grid]
    Integral = trapz((x_grid,y_grid), exp.(M))

    if i == 1
      p_density = contour(x_grid, y_grid, exp.(M .- log(Integral))', legend = :none)
    end
    
    B = kde(x_AGESS1)
    B_ESS = kde(x_ESS1)
    B_GESS = kde(x_GESS1)
    B_STAN = kde(x_STAN)
    B_ARW = kde(x_ARW1)
    pdf_AGESS = pdf(B, x_grid, y_grid)
    int_agess = trapz((x_grid,y_grid), pdf_AGESS)
    pdf_ESS = pdf(B_ESS, x_grid, y_grid)
    pdf_GESS = pdf(B_GESS, x_grid, y_grid)
    pdf_STAN = pdf(B_STAN, x_grid, y_grid)
    pdf_ARW = pdf(B_ARW, x_grid, y_grid)
    KL_AGESS = 0.0
    KL_ESS = 0.0
    KL_GESS = 0.0
    KL_STAN = 0.0
    KL_ARW = 0.0
    q = 0.0
    q1 = 0.0
    q2 = 0.0
    q3 = 0.0
    q4 = 0.0
    for k in 1:1000
      for j in 1:1000
        p = M[k,j] - log(Integral)
        q = max(pdf_AGESS[k,j], 1e-60)
        KL_AGESS += (exp(p) * (p - log(q))) 

        q1 = max(pdf_ESS[k,j], 1e-60)
        KL_ESS += (exp(p) * (p - log(q1)))

        q2 = max(pdf_GESS[k,j], 1e-60)
        KL_GESS +=(exp(p) * (p - log(q2)))

        q3 = max(pdf_STAN[k,j], 1e-60)
        KL_STAN += (exp(p) * (p - log(q3))) 

        q4 = max(pdf_ARW[k,j], 1e-60)
        KL_ARW += (exp(p) * (p - log(q4)))
      end
    end

    KL_dist_banana[i, 1] = KL_ESS
    KL_dist_banana[i, 2] = KL_GESS
    KL_dist_banana[i, 3] = KL_AGESS
    KL_dist_banana[i, 4] = KL_STAN
    KL_dist_banana[i, 5] = KL_ARW

    R"""
    library(stableGR)
    ess_ESS <- n.eff(x_ESS1)$n.eff
    ess_GESS <- n.eff(x_GESS1)$n.eff
    ess_AGESS <- n.eff(x_AGESS1)$n.eff
    """
    
    @rget ess_ESS
    @rget ess_GESS
    @rget ess_AGESS

    
    ESS_per_second_banana[i,1] = ess_ESS / ESS_time
    ESS_per_second_banana[i,2] = ess_GESS / GESS_time
    ESS_per_second_banana[i,3] = ess_AGESS / AGESS_time

    save(string(dir ,"//Banana//Sim", i,".jld2"), Dict("x_ESS" => x_ESS, 
                                                       "x_GESS" => x_GESS,
                                                       "x_AGESS" => x_AGESS,
                                                       "x_ARW" => x_ARW,
                                                       "x_STAN" => df[:,:,1],
                                                       "KL_ESS" => KL_ESS,
                                                       "KL_GESS" => KL_GESS,
                                                       "KL_AGESS" => KL_AGESS,
                                                       "KL_STAN" => KL_STAN,
                                                       "KL_ARW" => KL_ARW,
                                                       "ess_ESS" => ess_ESS,
                                                       "ess_GESS" => ess_GESS,
                                                       "ess_AGESS" => ess_AGESS,
                                                       "ESS_time" => ESS_time,
                                                       "GESS_time" => GESS_time,
                                                       "AGESS_time" => AGESS_time,
                                                       "HMC_time" => stan_time * 0.5,
                                                       "ARW_time" => AGESS_time,
                                                       "μ1" => μ1,
                                                       "μ2" => μ2,
                                                       "y" => data["y"]))
  else
    sim = load(string(dir ,"//Banana//Sim", i,".jld2"))
    
    KL_dist_banana[i, 1] = sim["KL_ESS"]
    KL_dist_banana[i, 2] = sim["KL_GESS"]
    KL_dist_banana[i, 3] = sim["KL_AGESS"]
    KL_dist_banana[i, 4] = sim["KL_STAN"]
    KL_dist_banana[i, 5] = sim["KL_ARW"]

    ESS_per_second_banana[i,1] = sim["ess_ESS"] /  sim["ESS_time"]
    ESS_per_second_banana[i,2] = sim["ess_GESS"] / sim["GESS_time"]
    ESS_per_second_banana[i,3] = sim["ess_AGESS"] / sim["AGESS_time"]

    mean_norm[i] = (sim["μ1"]^2 + sim["μ2"]^2)^0.5
  end
end

p6 = scatter(mean_norm, ESS_per_second_banana, labels = ["ESS" "GESS" "AGESS"], markerstrokewidth=0, color=[:red :blue :green], legend = false)
ylabel!("Effective Sample Size per Second")
xlabel!("Norm of μ")
relative_KL = zeros(100,4)
relative_KL[:,1:2] .= KL_dist_banana[:,1:2]
relative_KL[:,3:4] .= KL_dist_banana[:,4:5]
relative_KL .-= KL_dist_banana[:,3]

p7 = boxplot(["ESS" "GESS" "HMC" "ARW"], relative_KL, legend = false, color=[:red :blue :orange :purple], markerstrokewidth=0)
p7 = hline!([0], color =:green)
ylabel!("Relative KL Divergence")

plot(p_density, p1, p2, p6, p3, p4, p5, p7, layout = @layout([A B C D; E F G H]), margin= 5Plots.mm)
plot!(size = (1800, 1000))
  
savefig(string(dir ,"//Banana//Results.pdf"))

####################
### Twin Bananas ###
####################

model = "
data {
  int N;
  array[N] real y;
  real mu1;
  real mu2;
}

parameters {
  real theta1;
  real theta2;
}

model {
  for (i in 1:N){
    y[i] ~ normal(0.1 * (theta1 - mu1)^2 - 0.5 * (theta2 - mu2)^4 - (10 * (theta1 - mu1) * (theta2 - mu2)),10);
  }
  theta1 ~ normal(0,1);
  theta2 ~ normal(0,1);
}
";

sm_twin_bananas = SampleModel("Twin_bananas", model);

MCMC_iters = 500000
ESS_per_second_twin_bananas = zeros(100, 3)
colors = [:red :blue :green :orange :purple]
KL_dist_twin_banana = zeros(100, 3)

mean_norm = zeros(100)
Random.seed!(1234)
p1 = scatter()
p2 = scatter()
p3 = scatter()
p4 = scatter()
p5 = scatter()
p_density = scatter()

Random.seed!(123)
for i in 1:100
  if !isfile(string(dir ,"//Twin_Bananas//Sim", i,".jld2"))
    μ1 = randn() * 3
    μ2 = randn() * 3
    data = Dict("N" => 100, "y" => randn(100)* 10 .+ 100, "mu1" => μ1, "mu2" => μ2);
    mean_norm[i] = (μ1^2 + μ2^2)^0.5
    x_AGESS = zeros(MCMC_iters, 2)

    Σ_I = diagm(ones(length(data["y"])))
    ones_N = ones(length(data["y"]))
    Σ =  diagm(ones(2) .* 4)

    AGESS_time, Σ_adapt = AGESS(x_AGESS, b -> (logpdf(MvNormal((0.1 * (b[1] - μ1)^2  -  0.5 * (b[2] - μ2)^4 - 10 * (b[1]- μ1) * (b[2]- μ2)) * ones_N, 100 * Σ_I), data["y"]) + logpdf(Normal(0, 2), b[1]) + logpdf(Normal(0, 2), b[2])), 
          [0.0, 0.0], Σ, true)

    x_GESS = zeros(MCMC_iters, 2)
    GESS_time = GESS(x_GESS,  b -> (logpdf(MvNormal((0.1 * (b[1] - μ1)^2  -  0.5 * (b[2] - μ2)^4 - 10 * (b[1]- μ1) * (b[2]- μ2)) * ones_N, 100 * Σ_I), data["y"]) + logpdf(Normal(0, 2), b[1]) + logpdf(Normal(0, 2), b[2])),
        [0.0, 0.0], Σ)

    x_ESS = zeros(MCMC_iters, 2)
    ESS_time = ESS(x_ESS,  b -> logpdf(MvNormal((0.1 * (b[1] - μ1)^2  -  0.5 * (b[2] - μ2)^4 - 10 * (b[1]- μ1) * (b[2]- μ2)) * ones_N, 100 * Σ_I), data["y"]),
                  [0.0, 0.0], Σ)
      

    if i == 1
      ## Only fitting STAN and ARW models once because they get stuck in one of the modes, so ESS information is not useful

      ## Fit model using STAN
      t1 = time()
      rc = stan_sample(sm_twin_bananas; num_cpp_chains=1, num_chains=1, num_warmups=250000, num_samples=250000, data);
      stan_time = time() - t1 

      df = read_samples(sm_twin_bananas, :array);

      ## Fit model using ARW
      x_ARW = zeros(MCMC_iters, 2)
      ARW_time = ARW(x_ARW, b -> logpdf(MvNormal((0.1 * (b[1] - μ1)^2  -  0.5 * (b[2] - μ2)^4 - 10 * (b[1]- μ1) * (b[2]- μ2)) * ones_N, 100 * Σ_I), data["y"]), c -> (logpdf(Normal(0, 2), c[1]) + logpdf(Normal(0, 2), c[2])), 1000,
                  0.1, [0.0, 0.0], Σ)

    
      x_min = minimum([x_ESS[250001:end, 1] x_GESS[250001:end, 1] x_AGESS[250001:end, 1] df[:, 1] x_ARW[250001:end, 1]]) - 0.1
      x_max = maximum([x_ESS[250001:end, 1] x_GESS[250001:end, 1] x_AGESS[250001:end, 1] df[:, 1] x_ARW[250001:end, 1]]) + 0.1
      y_min = minimum([x_ESS[250001:end, 2] x_GESS[250001:end, 2] x_AGESS[250001:end, 2] df[:, 2] x_ARW[250001:end, 2]]) - 0.1
      y_max = maximum([x_ESS[250001:end, 2] x_GESS[250001:end, 2] x_AGESS[250001:end, 2] df[:, 2] x_ARW[250001:end, 2]]) + 0.1

      x_grid = range(x_min, x_max, 1000)
      y_grid = range(y_min, y_max, 1000)

      M = [(logpdf(MvNormal((0.1 * (x - μ1)^2  -  0.5 * (y - μ2)^4 - 10 * (x- μ1) * (y- μ2)) * ones_N, 100 * Σ_I), data["y"]) + logpdf(Normal(0, 2), x) + logpdf(Normal(0, 2), y)) for x = x_grid, y = y_grid]
      Integral = trapz((x_grid,y_grid), exp.(M))

      p_density = contour(x_grid, y_grid, exp.(M .- log(Integral))', legend = :none)
      p1 = scatter(x_ESS[250001:end, 1], x_ESS[250001:end, 2], alpha = 0.4, color = colors[1], legend = false, ylim = (y_min, y_max), xlim = (x_min, x_max), markerstrokewidth=0)
      p2 = scatter(x_GESS[250001:end, 1], x_GESS[250001:end, 2], alpha = 0.4, color = colors[2], legend = false,ylim = (y_min, y_max), xlim = (x_min, x_max), markerstrokewidth=0)
      p3 = scatter(x_AGESS[250001:end, 1], x_AGESS[250001:end, 2], alpha = 0.4, color = colors[3], legend = false, ylim = (y_min, y_max), xlim = (x_min, x_max), markerstrokewidth=0)
      mean_ell = [mean(x_AGESS[250001:end, 1]) mean(x_AGESS[250001:end, 2])]
      p3 = covellipse!(mean_ell[1,:], Σ_adapt, level = 0.68)
      p4 = scatter(df[:, 1], df[:, 2], alpha = 0.4, color = colors[4], legend = false, ylim = (y_min, y_max), xlim = (x_min, x_max), markerstrokewidth=0)
      p5 = scatter(x_ARW[250001:end, 1], x_ARW[250001:end, 2], alpha = 0.4, color = colors[5], legend = false, ylim = (y_min, y_max), xlim = (x_min, x_max), markerstrokewidth=0)
    end

    x_AGESS1 = x_AGESS[250001:end,:]
    x_GESS1 = x_GESS[250001:end,:]
    x_ESS1 = x_ESS[250001:end,:]

    x_min = minimum([x_ESS[250001:end, 1] x_GESS[250001:end, 1] x_AGESS[250001:end, 1]]) - 0.1
    x_max = maximum([x_ESS[250001:end, 1] x_GESS[250001:end, 1] x_AGESS[250001:end, 1]]) + 0.1
    y_min = minimum([x_ESS[250001:end, 2] x_GESS[250001:end, 2] x_AGESS[250001:end, 2]]) - 0.1
    y_max = maximum([x_ESS[250001:end, 2] x_GESS[250001:end, 2] x_AGESS[250001:end, 2]]) + 0.1

    x_grid = range(x_min, x_max, 1000)
    y_grid = range(y_min, y_max, 1000)

    M = [(logpdf(MvNormal((0.1 * (x - μ1)^2  -  0.5 * (y - μ2)^4 - 10 * (x- μ1) * (y- μ2)) * ones_N, 100 * Σ_I), data["y"]) + logpdf(Normal(0, 2), x) + logpdf(Normal(0, 2), y)) for x = x_grid, y = y_grid]
    Integral = trapz((x_grid,y_grid), exp.(M))

    B = kde(x_AGESS1)
    B_ESS = kde(x_ESS1)
    B_GESS = kde(x_GESS1)
    pdf_AGESS = pdf(B, x_grid, y_grid)
    pdf_ESS = pdf(B_ESS, x_grid, y_grid)
    pdf_GESS = pdf(B_GESS, x_grid, y_grid)
    KL_AGESS = 0.0
    KL_ESS = 0.0
    KL_GESS = 0.0
    q = 0.0
    q1 = 0.0
    q2 = 0.0
    for k in 1:1000
      for j in 1:1000
        p = M[k,j] - log(Integral)
        q = max(pdf_AGESS[k,j], 1e-60)
        KL_AGESS += (exp(p) * (p - log(q))) 

        q1 = max(pdf_ESS[k,j], 1e-60)
        KL_ESS += (exp(p) * (p - log(q1)))

        q2 = max(pdf_GESS[k,j], 1e-60)
        KL_GESS +=(exp(p) * (p - log(q2)))
      end
    end

    KL_dist_twin_banana[i, 1] = KL_ESS
    KL_dist_twin_banana[i, 2] = KL_GESS
    KL_dist_twin_banana[i, 3] = KL_AGESS

    @rput x_AGESS1
    @rput x_GESS1
    @rput x_ESS1

    R"""
    library(stableGR)
    ess_ESS <- n.eff(x_ESS1)$n.eff
    ess_GESS <- n.eff(x_GESS1)$n.eff
    ess_AGESS <- n.eff(x_AGESS1)$n.eff
    """

    @rget ess_ESS
    @rget ess_GESS
    @rget ess_AGESS

    ESS_per_second_twin_bananas[i,1] = ess_ESS / ESS_time
    ESS_per_second_twin_bananas[i,2] = ess_GESS / GESS_time
    ESS_per_second_twin_bananas[i,3] = ess_AGESS / AGESS_time

    save(string(dir ,"//Twin_Bananas//Sim", i,".jld2"), Dict("x_ESS" => x_ESS, 
                                                             "x_GESS" => x_GESS,
                                                             "x_AGESS" => x_AGESS,
                                                             "KL_ESS" => KL_ESS,
                                                             "KL_GESS" => KL_GESS,
                                                             "KL_AGESS" => KL_AGESS,
                                                             "ess_ESS" => ess_ESS,
                                                             "ess_GESS" => ess_GESS,
                                                             "ess_AGESS" => ess_AGESS,
                                                             "ESS_time" => ESS_time,
                                                             "GESS_time" => GESS_time,
                                                             "AGESS_time" => AGESS_time,
                                                             "μ1" => μ1,
                                                             "μ2" => μ2,
                                                             "y" => data["y"]))
  else
    sim = load(string(dir ,"//Twin_Bananas//Sim", i,".jld2"))

    ESS_per_second_twin_bananas[i,1] = sim["ess_ESS"] /  sim["ESS_time"]
    ESS_per_second_twin_bananas[i,2] = sim["ess_GESS"] / sim["GESS_time"]
    ESS_per_second_twin_bananas[i,3] = sim["ess_AGESS"] / sim["AGESS_time"]

    KL_dist_twin_banana[i, 1] =sim["KL_ESS"]
    KL_dist_twin_banana[i, 2] = sim["KL_GESS"]
    KL_dist_twin_banana[i, 3] = sim["KL_AGESS"]

    mean_norm[i] = (sim["μ1"]^2 + sim["μ2"]^2)^0.5
  end
end

p6 = boxplot(["ESS" "GESS" "AGESS"], ESS_per_second_twin_bananas, markerstrokewidth=0, color=[:red :blue :green], legend = false)
ylabel!("Effective Sample Size per Second")
relative_KL = zeros(100,2)
relative_KL[:,1:2] .= KL_dist_twin_banana[:,1:2]
relative_KL .-= KL_dist_twin_banana[:,3]

p7 = boxplot(["ESS" "GESS"], relative_KL, legend = false, color=[:red :blue :orange :purple], markerstrokewidth=0)
p7 = hline!([0], color =:green)
ylabel!("Relative KL Divergence")


plot(p_density, p1, p2, p6, p3, p4, p5,  p7, layout = @layout([A B C D; E F G H] ), margin= 5Plots.mm)
plot!(size = (1800, 1000))
savefig(string(dir ,"//Twin_Bananas//Results.pdf"))

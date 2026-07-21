using KernelFunctions, LinearAlgebra, LogExpFunctions, Distributions, LinearAlgebra, JLD2, Random, StatsBase, RCall, StatsPlots, Printf
using StanBase
# Set path to STAN
set_cmdstan_home!("C:\\cmdstan")
using StanSample, DataFrames, Stan
include("AGESS.jl")

dir = ".\\DeepGP"

## Create Directories
if !isdir(string(dir ,"//Large_Nugget"))
    mkdir(string(dir ,"//Large_Nugget"))
end
if !isdir(string(dir ,"//Small_Nugget"))
    mkdir(string(dir ,"//Small_Nugget"))
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
            min = minimum(v), max = maximum(v))
end
 
function write_ess_summary_table(filepath::AbstractString,
                                  ESS_per_second::AbstractMatrix{<:Real},
                                  ESS_per_iter::AbstractMatrix{<:Real},
                                  ESS_per_likelihood::AbstractMatrix{<:Real},
                                  total_time::AbstractMatrix{<:Real};
                                  method_names = ["ESS", "GESS", "AGESS", "STAN"])
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
            @printf(io, "%-10s %20s %24s\n", "Method", "Mean ± SD", "Median [Min, Max]")
            for (j, name) in enumerate(method_names)
                s = summarize_column(view(M, :, j))
                mean_sd_str = @sprintf("%.4g ± %.4g", s.mean, s.std)
                median_iqr_str = @sprintf("%.4g [%.4g, %.4g]", s.median, s.min, s.max)
                @printf(io, "%-10s %20s %24s\n", name, mean_sd_str, median_iqr_str)
            end
            println(io)
        end
    end
 
    println("Summary table written to: ", filepath)
end


function gen_data(N::T, min_eval::Y, max_eval::Y) where {Y<:AbstractFloat, T<:Integer}
    X = collect(LinRange(min_eval, max_eval, N))
    Y_N = sin.(X) .+ 2 * exp.(-30 * X.^2)
    return X, Y_N
end

function gen_data(X::AbstractVector{Y}) where {Y<:AbstractFloat}
    Y_N = sin.(X) .+ 2 * exp.(-30 * X.^2)
    return Y_N
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
                      Σ::AbstractMatrix{Y}, ν_y::Y)::Float64 where {Y<:AbstractFloat}
    construct_Kernel_Mat_y!(Σ, X, W, θ_y_x, θ_y_w)
    Σ[diagind(Σ)] .+=  g
    cholesky!(Σ)
    ph .= UpperTriangular(Σ)' \ Y_N
    lpdf =  -sum(log.(diag(Σ)))  - (0.5 * (ν_y + length(ph)) * log1p(dot(ph, ph) / ν_y))

    return lpdf
end

function likelihood_W(W::AbstractVector{Y}, X_N::AbstractVector{Y}, θ_w::Y, g::Y, ph::AbstractVector{Y}, Σ::AbstractMatrix{Y})::Float64 where {Y<:AbstractFloat}
    construct_Kernel_Mat!(Σ, X_N, θ_w)
    Σ[diagind(Σ)] .+= g
    cholesky!(Σ)

    ph .= UpperTriangular(Σ)' \ W
    lpdf = -sum(log.(diag(Σ)))  - 0.5 * dot(ph, ph)

    return lpdf
end


function sample_aux_parameters(θ_y_x::AbstractVector{Y}, θ_y_w::AbstractVector{Y},
                               θ_w::AbstractVector{Y}, ph1::AbstractVector{Y}, 
                               ph2::AbstractVector{Y}, Σ1::AbstractMatrix{Y},
                               Σ2::AbstractMatrix{Y}, X_N::AbstractVector{Y}, 
                               W::AbstractVector{Y}, Y_N::AbstractVector{Y},
                               σ_θ_w::Y, σ_θ_y_x::Y, σ_θ_y_w::Y,
                               prior_θ::Function, g::Y, g_x::Y, i::T, accept_vec::AbstractVector{Y}, ν_y::Y) where {Y<:AbstractFloat, T<:Integer}

    θ_y_x_prop = rand(LogNormal(log(θ_y_x[i]), σ_θ_y_x))
    accept_prob = likelihood_Y(W, X_N, Y_N, g, θ_y_x_prop, θ_y_w[i], ph1, Σ1, ν_y) + prior_θ(θ_y_x_prop) - (likelihood_Y(W, X_N, Y_N, g, θ_y_x[i], θ_y_w[i], ph2, Σ2, ν_y) + prior_θ(θ_y_x[i]))
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
    accept_prob = likelihood_Y(W, X_N, Y_N, g, θ_y_x[i], θ_y_w_prop, ph1, Σ1, ν_y) + prior_θ(θ_y_w_prop) - (likelihood_Y(W, X_N, Y_N, g, θ_y_x[i], θ_y_w[i], ph2, Σ2, ν_y) + prior_θ(θ_y_w[i]))
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
    accept_prob = likelihood_W(W, X_N, θ_w_prop, g_x, ph1, Σ1) + prior_θ(θ_w_prop) - (likelihood_W(W, X_N, θ_w[i], g_x, ph2, Σ2) + prior_θ(θ_w[i]))
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
                          θ_y_x::AbstractVector{Y}, θ_y_w::AbstractVector{Y}, g::Y, g_x::Y, Y_N::AbstractVector{Y}, 
                          X_N::AbstractVector{Y}, ν_y::Y;  burnin::Y=0.5, thinning::T = 1) where {Y<:AbstractFloat, T<:Integer}
    n_MCMC = size(W)[1]
    burnin_num = floor(Int64, (burnin * n_MCMC))
    P_out = length(time_points)
    P = length(X_N)
    steps = collect(range(burnin_num + 1, n_MCMC, step = thinning))
    Y_out = zeros(length(steps), P_out)
    W_out = zeros(P_out)
    k_out_X_const = zeros(P_out, P)
    k_out_X = zeros(P_out, P)
    k_out_Y = zeros(P_out, P)
    Σ = zeros(P,P)
    ph = zeros(P_out, P)
    ph1 = zeros(P)

    μ_out = zeros(length(steps), P_out)
    Σ_out = zeros(P_out, P_out) 

    for j in 1:P_out
        for k in 1:P
            k_out_X_const[j,k] = -(time_points[j] - X_N[k])^2
        end
    end

    iter = 1
    for i in steps
        ## get distribution of W_out
        k_out_X .= exp.(k_out_X_const ./ θ_w[i])
        construct_Kernel_Mat!(Σ, X_N, θ_w[i])
        Σ[diagind(Σ)] .+=  g_x
        ph .= (Σ \ k_out_X')'
        @views μ_out[iter,:] .= ph * W[i,:]
        construct_Kernel_Mat!(Σ_out, time_points, θ_w[i])
        Σ_out .-= ph * k_out_X'
        Σ_out[diagind(Σ_out)] .+= g_x
        Σ_out .= Hermitian(Σ_out)
        cholesky!(Σ_out)

        ## Generate sample of W_out
        W_out .= UpperTriangular(Σ_out)' * randn(P_out) 
        W_out .+= μ_out[iter,:]

        ## Generate sample of Y_out
        for j in 1:P_out
            for k in 1:P
                k_out_Y[j,k] = exp(-(((W_out[j] - W[i,k])^2 / θ_y_w[i]) + ((time_points[j] - X_N[k])^2 / θ_y_x[i])))
            end
        end


        @views construct_Kernel_Mat_y!(Σ, X_N, W[i,:], θ_y_x[i], θ_y_w[i])
        Σ[diagind(Σ)] .+=  g
        ph .= (Σ \ k_out_Y')'
        μ_out[iter,:] .= ph * Y_N

        cholesky!(Σ)
        ph1 .= UpperTriangular(Σ)' \ Y_N
        construct_Kernel_Mat_y!(Σ_out, time_points, W_out, θ_y_x[i], θ_y_w[i])
        Σ_out[diagind(Σ_out)] .+= g
        Σ_out .-= ph * k_out_Y'
        Σ_out .= Hermitian(Σ_out)

        d = dot(ph1, ph1)
        Σ_out .*= (ν_y + d) / (ν_y + P)
        cholesky!(Σ_out)

        
        Y_out[iter,:] .= UpperTriangular(Σ_out)' * randn(P_out)
        Y_out[iter,:] .*= 1 / sqrt(rand(Gamma((ν_y + P) / 2, 2 / (ν_y + P))))
        @views Y_out[iter,:] .+= μ_out[iter,:]
        iter = iter + 1
    end

    return Y_out, μ_out
end

function plot_CI(Y_N::AbstractVector{Y}, X_N::AbstractVector{Y}, Y_out::AbstractMatrix{Y},
                 time_points::AbstractVector{Y}, truth::AbstractVector{Y}) where {Y<:AbstractFloat}
    p = plot(X_N, Y_N, seriestype=:scatter, color = "red", label = "Observed Data",
             markersize = 6, markerstrokewidth = 0.5, markerstrokecolor = :black)
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

    p = plot!(size = (2000,2000))
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
                     θ_y_x::AbstractVector{Y}, θ_y_w::AbstractVector{Y}, g::Y, g_x::Y, prior_θ::Function,
                     σ_θ_y_x::Y = 0.1, σ_θ_y_w::Y = 0.1, σ_θ_w::Y = 0.1, ν_y::Y = 6.0, tuning_step::T = 25,
                     burnin::Y = 0.5) where {Y<:AbstractFloat, T<:Integer}
    accept_vec = zeros(3)
    n_MCMC = size(W)[1]
    P = length(Y_N)

    Σ1 = diagm(ones(P))
    Σ2 = diagm(ones(P))

    ph = ones(P)
    ph1 = ones(P)
    ph2 = ones(P)
    log_post = zeros(n_MCMC)
    lpdf = 0.0
    total_num_likevals = 0

    burnin_num = floor(Int64, burnin * n_MCMC)
    t1 = time()

    for i in 2:n_MCMC
        if i == burnin_num
            t1 = time()
            total_num_likevals = 0
        end
        ## Sample auxillary parameters
        @views sample_aux_parameters(θ_y_x, θ_y_w, θ_w, ph1, ph2, Σ1, Σ2, X_N, W[i,:], Y_N,
                                     σ_θ_w, σ_θ_y_x, σ_θ_y_w, prior_θ, g, g_x, i, accept_vec, ν_y)

        
        ## Sample W
        construct_Kernel_Mat!(Σ2, X_N, θ_w[i])
        Σ2[diagind(Σ2)] .+= g_x
        cholesky!(Σ2) 
        @views lpdf = likelihood_Y(W[i,:], X_N, Y_N, g, θ_y_x[i], θ_y_w[i], ph1, Σ1, ν_y)
        lpdf, num_lik_iter = ESS_SingleStep(W, ph, b -> likelihood_Y(b, X_N, Y_N, g, θ_y_x[i], θ_y_w[i], ph1, Σ1, ν_y), zeros(P), UpperTriangular(Σ2)', lpdf, i)
        total_num_likevals += num_lik_iter + 1 + 6 # Add 6 for evaluations in auxiliary variable sampling

        log_post[i] = lpdf + likelihood_W(W[i,:], X_N, θ_w[i], g_x, ph2, Σ2) + 
            logpdf(Gamma(1.0,2.0), θ_y_x[i]) + logpdf(Gamma(1.0,2.0), θ_y_w[i]) + logpdf(Gamma(1.0,2.0), θ_w[i])
        if (i % tuning_step) == 0
            println("MCMC iter: ", i)
            println("Acceptance θ_y_x: ", round(accept_vec[1] / tuning_step, digits=3))
            println("Acceptance θ_y_w: ", round(accept_vec[2] / tuning_step, digits=3))
            println("Acceptance θ_w: ", round(accept_vec[3] / tuning_step, digits=3))
            log_lik = @sprintf("%.2f", log_post[i])
            println("Log Posterior: ", log_lik)
            σ_θ_w = exp(log(σ_θ_w) + ((accept_vec[3] / tuning_step) - 0.44) / i)
            σ_θ_y_x = exp(log(σ_θ_y_x) + ((accept_vec[1] / tuning_step) - 0.44) / i)
            σ_θ_y_w = exp(log(σ_θ_y_w) + ((accept_vec[2] / tuning_step) - 0.44) / i)
            accept_vec .= 0
        end

        ## Update next state
        if i < n_MCMC
            @views W[i+1,:] .= W[i,:]
            θ_w[i+1] = θ_w[i]
            θ_y_x[i+1] = θ_y_x[i]
            θ_y_w[i+1] = θ_y_w[i]
        end

    end
    return time() - t1, total_num_likevals
end


#### Stan implementation
model = "
data {
  int N;
  real g;
  real g_x;
  vector[N] Y;
  vector[N] X;
}

parameters {
  vector[N] W;
  real<lower = 0> theta_y_x;
  real<lower = 0> theta_y_w;
  real<lower = 0> theta_w;
}

model {
  matrix[N, N] K_y;
  for (i in 1:(N - 1)) {
    K_y[i, i] = 1 + g;
    for (j in (i + 1):N) {
      K_y[i, j] = exp(-((X[i] - X[j])^2 / theta_y_x) - ((W[i] - W[j])^2 / theta_y_w));
      K_y[j, i] = K_y[i, j];
    }
  }
  K_y[N, N] = 1 + g;

  matrix[N, N] K_w;
  for (i in 1:(N - 1)) {
    K_w[i, i] = 1 + g_x;
    for (j in (i + 1):N) {
      K_w[i, j] = exp(- ((X[i] - X[j])^2 / theta_w));
      K_w[j, i] = K_w[i, j];
    }
  }
  K_w[N, N] = 1 + g_x;

  vector[N] mu = rep_vector(0, N);

  Y ~ multi_student_t(6, mu, K_y);
  W ~ multi_normal(mu, K_w);
  theta_w ~ gamma(1, 0.5);
  theta_y_w ~ gamma(1, 0.5);
  theta_y_x ~ gamma(1, 0.5);
}
";

N_obs = 35 
X_N, Y_N = gen_data(N_obs, -5.0, 5.0)
g = 1e-8 
g_x = 1e-8 

ESS_per_second = zeros(10,4)
ESS_per_iter = zeros(10,4)
ESS_per_likelihood = zeros(10,4)
times = zeros(10,4)

n_reps = 10
MCMC_iters = 250_000

sd_Y_pred = zeros(4, n_reps, 498)
Random.seed!(1234)

total_evals = zeros(10,4)

p = plot()
p1 = plot()
p2 = plot()
p3 = plot()

for i in 1:10
    if !isfile(string(dir ,"//Small_Nugget//Sim", i,".jld2"))
        θ_w_i = rand() 
        θ_y_x_i = rand() 
        θ_y_w_i = rand()
        ##########################
        ### STAN implementation ##
        ##########################
        sm = SampleModel("deepGP", model);

        data = Dict("N" => N_obs, "g" => g, "g_x" => g_x, "Y" => Y_N, "X" => X_N);

        stan_init = [(W = X_N, theta_y_x = θ_y_x_i, theta_y_w = θ_y_w_i, theta_w = θ_w_i)]

        t1 = time()
        rc = stan_sample(sm; num_cpp_chains=1, num_chains=1, 
                        num_warmups=floor(Int, 0.25 * MCMC_iters), 
                        num_samples=floor(Int, 0.25 * MCMC_iters), 
                        init = stan_init, data);
        total_stan_time = time() - t1 

        df_Stan = read_samples(sm, :array);

        df_diag = read_samples(sm, :dataframe; include_internals=true)

        # Sum the number of leapfrog steps (which will count the number of gradient evaluations)
        # Add 1 per iteration to initialize the Hamiltonian in each step
        total_evals[i,4] = sum(df_diag.n_leapfrog__ .+ 1)

        stan_time = stan_sampling_time(sm)

        df_Stan = df_Stan[:,:,1]

        #########################
        ### ESS implementation ##
        #########################

        W = ones(MCMC_iters, N_obs) 
        ## Initialize with W = X_N
        W[1,:] .= X_N
        W[2,:] .= X_N

        θ_w = ones(MCMC_iters) * θ_w_i
        θ_y_x = ones(MCMC_iters) * θ_y_x_i
        θ_y_w = ones(MCMC_iters) * θ_y_w_i

        t1 = time()
        ESS_time, total_evals[i,1] = sampler_ESS(Y_N, X_N, W, θ_w, θ_y_x, θ_y_w, g, g_x, k -> logpdf(Gamma(1.0,2.0), k))
        total_ESS_time = time() - t1 

        df_ESS = hcat(W, θ_y_x, θ_y_w, θ_w)
        df_ESS = df_ESS[(floor(Int, 0.5 * MCMC_iters) + 1):MCMC_iters,:]

        ##########################
        ### GESS implementation ##
        ##########################

        W_GESS = ones(MCMC_iters, N_obs + 3) 
        W_GESS[1,1:N_obs] .= X_N
        W_GESS[2,1:N_obs] .= X_N

        W_GESS[1:2,N_obs + 1] .= log(θ_y_x_i)
        W_GESS[1:2,N_obs + 2] .= log(θ_y_w_i)
        W_GESS[1:2,N_obs + 3] .= log(θ_w_i)

        μ_0 = zeros(N_obs + 3)
        Σ_0 = diagm(ones(N_obs + 3))
        ph1 = ones(N_obs)
        Σ1 = diagm(ones(N_obs))
        ph2 = ones(N_obs)
        Σ2 = diagm(ones(N_obs))
        ν_y = 6.0

        t1 = time()
        GESS_time, total_evals[i,2] = GESS(W_GESS, b -> (likelihood_Y(b[1:N_obs], X_N, Y_N, g, exp(b[N_obs+1]), exp(b[N_obs+2]), ph1, Σ1, ν_y) + 
                            likelihood_W(b[1:N_obs], X_N, exp(b[N_obs+3]), g_x, ph2, Σ2) + logpdf(Gamma(1.0,2.0), exp(b[N_obs+1])) + 
                            b[N_obs+1] + logpdf(Gamma(1.0,2.0), exp(b[N_obs+2])) + b[N_obs+2] + logpdf(Gamma(1.0,2.0), exp(b[N_obs+3])) + b[N_obs+3]), μ_0, Σ_0)
        total_GESS_time = time() - t1
        df_GESS = W_GESS[(floor(Int, 0.5 * MCMC_iters) + 1):MCMC_iters,:]
        df_GESS[:,(N_obs + 1):(N_obs + 3)] .= exp.(df_GESS[:,(N_obs + 1):(N_obs + 3)])

        ##########################
        ## AGESS implementation ##
        ##########################

        W_AGESS = ones(MCMC_iters, N_obs + 3) 
        ## Initialize with W = X_N
        W_AGESS[1,1:N_obs] .= X_N
        W_AGESS[2,1:N_obs] .= X_N

        W_AGESS[1:2,N_obs + 1] .= log(θ_y_x_i)
        W_AGESS[1:2,N_obs + 2] .= log(θ_y_w_i)
        W_AGESS[1:2,N_obs + 3] .= log(θ_w_i)

        μ_0 = zeros(N_obs + 3)
        Σ_0 = diagm(ones(N_obs + 3))
        ph1 = ones(N_obs)
        Σ1 = diagm(ones(N_obs))
        ph2 = ones(N_obs)
        Σ2 = diagm(ones(N_obs))
        t1 = time()
        AGESS_time, Σ_adapt, μ_adapt, total_evals[i,3] = AGESS(W_AGESS, b -> (likelihood_Y(b[1:N_obs], X_N, Y_N, g, exp(b[N_obs+1]), exp(b[N_obs+2]), ph1, Σ1, ν_y) + 
                                                likelihood_W(b[1:N_obs], X_N, exp(b[N_obs+3]), g_x, ph2, Σ2) + logpdf(Gamma(1.0,2.0), exp(b[N_obs+1])) + 
                                                b[N_obs+1] + logpdf(Gamma(1.0,2.0), exp(b[N_obs+2])) + b[N_obs+2] + logpdf(Gamma(1.0,2.0), exp(b[N_obs+3])) + b[N_obs+3]), μ_0, Σ_0, true)
        total_AGESS_time = time() -t1
        df_AGESS = W_AGESS[(floor(Int, 0.5 * MCMC_iters) + 1):MCMC_iters,:]
        df_AGESS[:,(N_obs + 1):(N_obs + 3)] .= exp.(df_AGESS[:,(N_obs + 1):(N_obs + 3)])

        time_points = collect(collect(LinRange(-5.0, 5.0, 500)))
        time_points = setdiff(time_points, X_N)
        Y_pred_AGESS, mu_out_AGESS = predictive_draws(time_points, df_AGESS[:,1:N_obs], df_AGESS[:,N_obs + 3], df_AGESS[:,N_obs + 1], df_AGESS[:,N_obs + 2], g, g_x, Y_N, X_N, 6.0, burnin = 0.0)
        Y_pred_GESS, mu_out_GESS = predictive_draws(time_points, df_GESS[:,1:N_obs], df_GESS[:,N_obs + 3], df_GESS[:,N_obs + 1], df_GESS[:,N_obs + 2], g, g_x, Y_N, X_N, 6.0, burnin = 0.0)
        Y_pred_Stan, mu_out_Stan = predictive_draws(time_points, df_Stan[:,1:N_obs], df_Stan[:,N_obs + 3], df_Stan[:,N_obs + 1], df_Stan[:,N_obs + 2], g, g_x, Y_N, X_N, 6.0, burnin = 0.0)
        Y_pred_ESS, mu_out_ESS = predictive_draws(time_points, df_ESS[:,1:N_obs], df_ESS[:,N_obs + 3], df_ESS[:,N_obs + 1], df_ESS[:,N_obs + 2], g, g_x, Y_N, X_N, 6.0, burnin = 0.0)
        truth = gen_data(time_points)
        if i == 1
            p = plot_CI(Y_N, X_N, Y_pred_ESS, time_points, truth)
            p1 = plot_CI(Y_N, X_N, Y_pred_GESS, time_points, truth)
            p2 = plot_CI(Y_N, X_N, Y_pred_AGESS, time_points, truth)
            p3 = plot_CI(Y_N, X_N, Y_pred_Stan, time_points, truth)
        end
        for j in 1:498
            sd_Y_pred[1, i, j] = sqrt(var(Y_pred_ESS[:,j]))
            sd_Y_pred[2, i, j] = sqrt(var(Y_pred_GESS[:,j]))
            sd_Y_pred[3, i, j] = sqrt(var(Y_pred_AGESS[:,j]))
            sd_Y_pred[4, i, j] = sqrt(var(Y_pred_Stan[:,j]))
        end

        @rput mu_out_AGESS
        @rput mu_out_GESS
        @rput mu_out_Stan
        @rput mu_out_ESS

        R"""
        library(stableGR)
        ess_AGESS <- n.eff(mu_out_AGESS[,seq(1,500,25)])$n.eff
        ess_ESS <- n.eff(mu_out_ESS[,seq(1,500,25)])$n.eff
        ess_GESS <- n.eff(mu_out_GESS[,seq(1,500,25)])$n.eff
        ess_Stan <- n.eff(mu_out_Stan[,seq(1,500,25)])$n.eff
        """
        @rget ess_AGESS
        @rget ess_ESS
        @rget ess_GESS
        @rget ess_Stan

        ESS_per_second[i,1] = ess_ESS / (ESS_time)
        ESS_per_second[i,2] = ess_GESS / (GESS_time)
        ESS_per_second[i,3] = ess_AGESS / (AGESS_time)
        ESS_per_second[i,4] = ess_Stan/ (stan_time)

        ESS_per_iter[i,1] = ess_ESS / (0.5 * MCMC_iters)
        ESS_per_iter[i,2] = ess_GESS / (0.5 * MCMC_iters)
        ESS_per_iter[i,3] = ess_AGESS / (0.5 * MCMC_iters)
        ESS_per_iter[i,4] = ess_Stan / (0.25 * MCMC_iters)

        ESS_per_likelihood[i,1] = ess_ESS / total_evals[i,1]
        ESS_per_likelihood[i,2] = ess_GESS / total_evals[i,2]
        ESS_per_likelihood[i,3] = ess_AGESS / total_evals[i,3]
        ESS_per_likelihood[i,4] = ess_Stan / total_evals[i,4]

        times[i,1] = total_ESS_time
        times[i,2] = total_GESS_time
        times[i,3] = total_AGESS_time
        times[i,4] = total_stan_time
        save(string(dir ,"//Small_Nugget//Sim", i,".jld2"), Dict("df_Stan" => df_Stan, 
                                                                 "df_ESS" => df_ESS,
                                                                 "df_GESS" => df_GESS,
                                                                 "df_AGESS" => df_AGESS,
                                                                 "ess_ESS" => ess_ESS,
                                                                 "ess_GESS" => ess_GESS,
                                                                 "ess_AGESS" => ess_AGESS,
                                                                 "ess_Stan" => ess_Stan,
                                                                 "ESS_time" => ESS_time,
                                                                 "total_ESS_time" => total_ESS_time,
                                                                 "GESS_time" => GESS_time,
                                                                 "total_GESS_time" => total_GESS_time,
                                                                 "AGESS_time" => AGESS_time,
                                                                 "total_AGESS_time" => total_AGESS_time,
                                                                 "stan_time" => stan_time,
                                                                 "total_stan_time" => total_stan_time,
                                                                 "ESS_evals" => total_evals[i,1],
                                                                 "GESS_evals" => total_evals[i,2],
                                                                 "AGESS_evals" => total_evals[i,3],
                                                                 "stan_evals" => total_evals[i,4],
                                                                 "θ_w" => θ_w_i,
                                                                 "θ_y_x" => θ_y_x_i,
                                                                 "θ_y_w" => θ_y_w_i,))
    else
        sim = load(string(dir ,"//Small_Nugget//Sim", i,".jld2")) 
        df_AGESS = sim["df_AGESS"]
        df_ESS = sim["df_ESS"]
        df_GESS = sim["df_GESS"]
        df_Stan = sim["df_Stan"]
        time_points = collect(collect(LinRange(-5.0, 5.0, 500)))
        time_points = setdiff(time_points, X_N)
        Y_pred_AGESS, mu_out_AGESS = predictive_draws(time_points, df_AGESS[:,1:N_obs], df_AGESS[:,N_obs + 3], df_AGESS[:,N_obs + 1], df_AGESS[:,N_obs + 2], g, g_x, Y_N, X_N, 6.0, burnin = 0.0)
        Y_pred_GESS, mu_out_GESS = predictive_draws(time_points, df_GESS[:,1:N_obs], df_GESS[:,N_obs + 3], df_GESS[:,N_obs + 1], df_GESS[:,N_obs + 2], g, g_x, Y_N, X_N, 6.0, burnin = 0.0)
        Y_pred_Stan, mu_out_Stan = predictive_draws(time_points, df_Stan[:,1:N_obs], df_Stan[:,N_obs + 3], df_Stan[:,N_obs + 1], df_Stan[:,N_obs + 2], g, g_x, Y_N, X_N, 6.0, burnin = 0.0)
        Y_pred_ESS, mu_out_ESS = predictive_draws(time_points, df_ESS[:,1:N_obs], df_ESS[:,N_obs + 3], df_ESS[:,N_obs + 1], df_ESS[:,N_obs + 2], g, g_x, Y_N, X_N, 6.0, burnin = 0.0)
        truth = gen_data(time_points)
        if i == 1
            p = plot_CI(Y_N, X_N, Y_pred_ESS, time_points, truth)
            p1 = plot_CI(Y_N, X_N, Y_pred_GESS, time_points, truth)
            p2 = plot_CI(Y_N, X_N, Y_pred_AGESS, time_points, truth)
            p3 = plot_CI(Y_N, X_N, Y_pred_Stan, time_points, truth)
        end
        for j in 1:498
            sd_Y_pred[1, i, j] = sqrt(var(Y_pred_ESS[:,j]))
            sd_Y_pred[2, i, j] = sqrt(var(Y_pred_GESS[:,j]))
            sd_Y_pred[3, i, j] = sqrt(var(Y_pred_AGESS[:,j]))
            sd_Y_pred[4, i, j] = sqrt(var(Y_pred_Stan[:,j]))
        end

        @rput mu_out_AGESS
        @rput mu_out_GESS
        @rput mu_out_Stan
        @rput mu_out_ESS

        R"""
        library(stableGR)
        ess_AGESS <- n.eff(mu_out_AGESS[,seq(1,500,25)])$n.eff
        ess_ESS <- n.eff(mu_out_ESS[,seq(1,500,25)])$n.eff
        ess_GESS <- n.eff(mu_out_GESS[,seq(1,500,25)])$n.eff
        ess_Stan <- n.eff(mu_out_Stan[,seq(1,500,25)])$n.eff
        """
        @rget ess_AGESS
        @rget ess_ESS
        @rget ess_GESS
        @rget ess_Stan

        ESS_per_second[i,1] = ess_ESS / (sim["ESS_time"])
        ESS_per_second[i,2] = ess_GESS / (sim["GESS_time"])
        ESS_per_second[i,3] = ess_AGESS / (sim["AGESS_time"])
        ESS_per_second[i,4] = ess_Stan/ (sim["stan_time"])

        ESS_per_iter[i,1] = ess_ESS / (0.5 * MCMC_iters)
        ESS_per_iter[i,2] = ess_GESS / (0.5 * MCMC_iters)
        ESS_per_iter[i,3] = ess_AGESS / (0.5 * MCMC_iters)
        ESS_per_iter[i,4] = ess_Stan / (0.5 * MCMC_iters)

        ESS_per_likelihood[i,1] = ess_ESS / sim["ESS_evals"]
        ESS_per_likelihood[i,2] = ess_GESS / sim["GESS_evals"]
        ESS_per_likelihood[i,3] = ess_AGESS /sim["AGESS_evals"]
        ESS_per_likelihood[i,4] = ess_Stan / sim["stan_evals"]

        times[i,1] = sim["total_ESS_time"]
        times[i,2] = sim["total_GESS_time"]
        times[i,3] = sim["total_AGESS_time"]
        times[i,4] = sim["total_stan_time"]
        println(i)
    end
end

box_ess = boxplot(["AGESS"], ESS_per_second[1:10], title = "Effective Sample Size", legend = false, color =:green, markerstrokewidth=0)
ylabel!("Effective Sample Size per Second")

box_time =  boxplot(["ESS" "GESS" "AGESS" "HMC"], times, title = "Total Computation Time", legend = false, yscale=:log10, yticks = [10, 100, 1000, 10000, 100000], ylims = [10, 100000], markerstrokewidth=0, color = [:red :blue :green :orange])
ylabel!("Time (Seconds)")

time_points = collect(collect(LinRange(-5.0, 5.0, 500)))
time_points = setdiff(time_points, X_N)

ESS_sd = plot(time_points, sd_Y_pred[1,:,:]', legend = false, title = "ESS & ARW")
ylabel!("Standard Deviations")

GESS_sd = plot(time_points, sd_Y_pred[2,:,:]', legend = false, title = "GESS")
ylabel!("Standard Deviations")

AGESS_sd = plot(time_points, sd_Y_pred[3,:,:]', legend = false, title = "AGESS")
ylabel!("Standard Deviations")

HMC_sd = plot(time_points, sd_Y_pred[4,:,:]', legend = false, title = "HMC")
ylabel!("Standard Deviations")

truth = gen_data(time_points)
p_out = plot(time_points, truth, color = "black", label ="Truth", legend = :outerbottom)
p_out = plot!(X_N, Y_N, seriestype=:scatter, color = "red", label = "Observed Data",
             markersize = 5, markerstrokewidth = 0.75, markerstrokecolor = :black)
plot(p_out, ESS_sd, GESS_sd, AGESS_sd, HMC_sd, layout = @layout([a{0.33w} [b c; d e]]), margin= 8Plots.mm,
     fontfamily = "Computer Modern", titlefontsize = 28, guidefontsize = 24, tickfontsize = 20,
     legendfontsize = [10 20 20 20 20], linewidth = 1.5, framestyle = :axes, grid = false)
plot!(size = (2000, 1000))
savefig(string(dir, "//Small_Nugget//Results.pdf"))

write_ess_summary_table(string(dir, "//Small_Nugget//summary_stats.txt"),
                         ESS_per_second[1:9,3:4], ESS_per_iter[1:9,3:4], ESS_per_likelihood[1:9,3:4], 
                         times[1:9,3:4], method_names = ["AGESS", "HMC"])

conv_ESS_ind = norm.(eachcol(sd_Y_pred[1,:,:]' .- sd_Y_pred[3,1,:])) .< maximum(norm.(eachcol(sd_Y_pred[4,1:9,:]' .- sd_Y_pred[3,1,:])))
write_ess_summary_table(string(dir, "//Small_Nugget//summary_stats_ESS_ARW.txt"),
                         ESS_per_second[conv_ESS_ind,1:1], ESS_per_iter[conv_ESS_ind,1:1], ESS_per_likelihood[conv_ESS_ind,1:1], 
                         times[conv_ESS_ind,1:1], method_names = ["ESS & ARW"])                    
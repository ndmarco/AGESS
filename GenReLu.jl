using LogExpFunctions, Distributions, LinearAlgebra, JLD2, Random, StatsBase, RCall, StatsPlots

include("AGESS.jl")
dir = ".\\GenReLU"

## Function to generate the data
function generate_data(N::T, P::T, ν::Y = 6.0) where {Y<:AbstractFloat, T<:Integer}
    β = randn(P) * (2 * log(P))^(1.0 / 4)
    β .*= sqrt(rand(Gamma(ν/2, 2/ ν)))
    x::Matrix{typeof(ν)} = randn(N, P) .+ randn() * 0.5
    μ = zeros(N)
    y::Vector{typeof(N)} = zeros(Int64, N)

    for i in 1:N
        μ[i] = max(0, dot(x[i,:], β))
        y[i] = rand(Binomial(1, logistic(μ[i])), 1)[1]
    end


    return β, x, μ, y
end

## Likelihood
function log_likelihood(β::AbstractVector{Y}, x::Matrix{Y}, y::Vector{T})::Float64 where {Y <:AbstractFloat, T<:Integer}
    log_lik::Float64 = 0.0
    z::Float64 = 0.0
    for i in eachindex(y)
        @views z = dot(x[i,:], β)
        if z < 0.0
            z = 0.0
        end
        log_lik -= log1p(exp(-(sign(y[i] - 0.5) * z)))
    end

    return log_lik
end

## Prior on β
function log_prior(β::AbstractVector{Y})::Float64 where {Y <:AbstractFloat}
    log_p::Float64 = -0.5 * dot(β, β)
    return log_p
end


function log_posterior(β::AbstractVector{Y}, x::Matrix{Y}, y::Vector{T})::Float64 where {Y <:AbstractFloat, T<:Integer}
    log_lik::Float64 = log_likelihood(β, x, y) + log_prior(β)
    return log_lik
end

function summarize_column(v::AbstractVector{<:Real})
    return (mean = mean(filter(!isnan, v)), std = std(filter(!isnan, v)), median = median(filter(!isnan, v)),
            q10 = quantile(filter(!isnan, v), 0.1), q90 = quantile(filter(!isnan, v), 0.9))
end
 
function write_ess_summary_table(filepath::AbstractString,
                                  ESS_per_second::AbstractMatrix{<:Real},
                                  ESS_per_iter::AbstractMatrix{<:Real},
                                  ESS_per_likelihood::AbstractMatrix{<:Real}, 
                                  total_times::AbstractMatrix{<:Real},
                                  method_names;
                                  P = [2, 10, 50])
    metrics = [
        ("ESS per Second",               ESS_per_second),
        ("ESS per Iteration",            ESS_per_iter),
        ("ESS per Likelihood Evaluation", ESS_per_likelihood),
        ("Total Computational Time ", total_times),
    ]
 
    open(filepath, "w") do io
        println(io, "Summary Statistics: Generalized ReLU (n = ", size(ESS_per_second, 1), " reps, method = ", method_names, ")")
        println(io, "="^70)
        println(io)
        for (metric_name, M) in metrics
            println(io, metric_name)
            println(io, "-"^70)
            @printf(io, "%-10s %20s %24s\n", "Dimension", "Mean ± SD", "Median [Q10, Q90]")
            for (j, dim) in enumerate(P)
                s = summarize_column(view(M, :, j))
                mean_sd_str = @sprintf("%.4g ± %.4g", s.mean, s.std)
                median_iqr_str = @sprintf("%.4g [%.4g, %.4g]", s.median, s.q10, s.q90)
                @printf(io, "%-10s %20s %24s\n", dim, mean_sd_str, median_iqr_str)
            end
            println(io)
        end
        println(io, "-"^70)
        @printf(io,"%30s %24s\n", "Number of Chains not Converged (p = 2)", sum(isnan.(ESS_per_second[:,1])))
        @printf(io,"%30s %24s\n", "Number of Chains not Converged (p = 10)", sum(isnan.(ESS_per_second[:,2])))
        @printf(io,"%30s %24s\n", "Number of Chains not Converged (p = 50)", sum(isnan.(ESS_per_second[:,3])))
    end
 
    println("Summary table written to: ", filepath)
end

### Run Simulation
## Settings:
##       P = 2, 10, 50
##       N = 1000
##       MCMC_iters = 10000
##       N_Data_Sets = 100

## Create Directories
if !isdir(string(dir ,"//2_Cov"))
    mkdir(string(dir ,"//2_Cov"))
end
if !isdir(string(dir ,"//10_Cov"))
    mkdir(string(dir ,"//10_Cov"))
end
if !isdir(string(dir ,"//50_Cov"))
    mkdir(string(dir ,"//50_Cov"))
end

percent_zero = zeros(3, 100)
ESS_per_second_ESS = zeros(3,100)
ESS_per_second_GESS = zeros(3,100)
ESS_per_second_AGESS = zeros(3,100)
ESS_per_second_ARW = zeros(3, 100)

ESS_per_iteration_ESS = zeros(3,100)
ESS_per_iteration_GESS = zeros(3,100)
ESS_per_iteration_AGESS = zeros(3,100)
ESS_per_iteration_ARW = zeros(3, 100)

ESS_per_likelihood_ESS = zeros(3,100)
ESS_per_likelihood_GESS = zeros(3,100)
ESS_per_likelihood_AGESS = zeros(3,100)
ESS_per_likelihood_ARW = zeros(3, 100)


total_time_ESS = zeros(3,100)
total_time_GESS = zeros(3,100)
total_time_AGESS = zeros(3,100)
total_time_ARW = zeros(3, 100)

total_evals = zeros(3,4,100)

P_vec = [2, 10, 50]

Random.seed!(1234)
for j in 1:3
    Σ = diagm(ones(P_vec[j]))
    μ_j = zeros(P_vec[j])
    for i in 1:100
        if !isfile(string(dir ,"//", P_vec[j], "_Cov//Sim", i,".jld2"))
            Random.seed!(i)
            β, x, μ, y = generate_data(1000, P_vec[j])
            sum(μ .== 0)

            ## Elliptical Slice Sampling
            β_samp_ESS = zeros(10000 * P_vec[j], P_vec[j])
            t2 = time()
            time_non_burnin_ESS, total_evals[j,1,i] = ESS(β_samp_ESS, β -> log_likelihood(β, x, y), μ_j, Σ, burnin = 0.25)
            total_time_ESS[j,i] = time() - t2
            

            ## Generalized Elliptical Slice Sampling
            β_samp_GESS = zeros(10000 * P_vec[j], P_vec[j])
            t2 = time()
            time_non_burnin_GESS, total_evals[j,2,i] = GESS(β_samp_GESS,  β -> log_posterior(β, x, y), μ_j, Σ, burnin = 0.25)
            total_time_GESS[j,i] = time() - t2


            ## Adaptive Generalized Elliptical Slice Sampling
            β_samp_AGESS = zeros(10000 * P_vec[j], P_vec[j])
            t2 = time()
            time_non_burnin_AGESS, Σ_adapt, μ_adapt_AGESS, total_evals[j,3,i] = AGESS(β_samp_AGESS, β -> log_posterior(β, x, y), μ_j, Σ, true, burnin = 0.25)
            total_time_AGESS[j,i] = time() - t2
            

            ## Adaptive Random Walk
            β_samp_ARW = zeros(30000 * P_vec[j], P_vec[j])
            t2 = time()
            time_non_burnin_ARW, total_evals[j,4,i] = ARW(β_samp_ARW, β -> log_likelihood(β, x, y), log_prior, 10000, 0.01, μ_j, Σ, burnin = 25/300)
            total_time_ARW[j,i] = time() - t2
            

            ## Get Multivariate Effective Sample Size using R package "stableGR"
            beta_samp_ESS_1 = β_samp_ESS[2500*P_vec[j]:10000*P_vec[j],:]
            beta_samp_GESS_1 = β_samp_GESS[2500*P_vec[j]:10000*P_vec[j],:]
            beta_samp_AGESS_1 = β_samp_AGESS[2500*P_vec[j]:10000*P_vec[j],:]
            beta_samp_ARW_1 = β_samp_ARW[2500*P_vec[j]:30000*P_vec[j],:]

            @rput beta_samp_ESS_1
            @rput beta_samp_AGESS_1
            @rput beta_samp_GESS_1
            @rput beta_samp_ARW_1

            R"""
            library(stableGR)
            rg_ESS <- n.eff(beta_samp_ESS_1, epsilon = 0.1)
            rg_GESS <- n.eff(beta_samp_GESS_1, epsilon = 0.1)
            rg_AGESS <- n.eff(beta_samp_AGESS_1, epsilon = 0.1)
            rg_ARW <- n.eff(beta_samp_ARW_1, epsilon = 0.1)

            ess_ESS <- -1
            ess_GESS <- -1
            ess_AGESS <- -1
            ess_ARW <- -1
            if(rg_ESS$converged == TRUE){
                ess_ESS <- rg_ESS$n.eff
            }
            if(rg_GESS$converged == TRUE){
                ess_GESS <- rg_GESS$n.eff
            }
            if(rg_AGESS$converged == TRUE){
                ess_AGESS <- rg_AGESS$n.eff
            }
            if(rg_ARW$converged == TRUE){
                ess_ARW <- rg_ARW$n.eff
            }
            """

            @rget ess_ESS
            @rget ess_GESS
            @rget ess_AGESS
            @rget ess_ARW


            # Save output of Simulation
            save(string(dir ,"//", P_vec[j], "_Cov//Sim", i,".jld2"), Dict("β_samp_ESS" => β_samp_ESS, 
                                                                           "β_samp_GESS" => β_samp_GESS,
                                                                           "β_samp_AGESS" => β_samp_AGESS,
                                                                           "β_samp_ARW" => β_samp_ARW,
                                                                           "time_non_burnin_ESS" => time_non_burnin_ESS,
                                                                           "time_non_burnin_GESS" => time_non_burnin_GESS,
                                                                           "time_non_burnin_AGESS" => time_non_burnin_AGESS,
                                                                           "time_non_burnin_ARW" => time_non_burnin_ARW,
                                                                           "total_time_ESS" => total_time_ESS[j,i],
                                                                           "total_time_GESS" => total_time_GESS[j,i],
                                                                           "total_time_AGESS" => total_time_AGESS[j,i],
                                                                           "total_time_ARW" => total_time_ARW[j,i],
                                                                           "ESS_evals" => total_evals[j,1,i],
                                                                           "GESS_evals" => total_evals[j,2,i],
                                                                           "AGESS_evals" => total_evals[j,3,i],
                                                                           "ARW_evals" => total_evals[j,4,i],
                                                                           "Eff_SS_ESS" => ess_ESS,
                                                                           "Eff_SS_GESS" => ess_GESS,
                                                                           "Eff_SS_AGESS" => ess_AGESS,
                                                                           "Eff_SS_ARW" => ess_ARW,
                                                                           "β" => β,
                                                                           "x" => x,
                                                                           "μ" => μ,
                                                                           "y" => y))
            
            # Calculate Metrics
            percent_zero[j,i] = sum(μ .== 0) / 1000
            ESS_per_second_ESS[j,i] = ess_ESS / time_non_burnin_ESS
            ESS_per_second_GESS[j,i] = ess_GESS / time_non_burnin_GESS
            ESS_per_second_AGESS[j,i] = ess_AGESS / time_non_burnin_AGESS
            ESS_per_second_ARW[j,i] = ess_ARW / time_non_burnin_ARW

            ESS_per_iteration_ESS[j,i] = ess_ESS / (10000 * P_vec[j] * 0.75)
            ESS_per_iteration_GESS[j,i] = ess_GESS / (10000 * P_vec[j] * 0.75)
            ESS_per_iteration_AGESS[j,i] = ess_AGESS / (10000 * P_vec[j] * 0.75)
            ESS_per_iteration_ARW[j,i] = ess_ARW / (30000 * P_vec[j] * (275/300))

            ESS_per_likelihood_ESS[j,i] = ess_ESS / total_evals[j,1,i]
            ESS_per_likelihood_GESS[j,i] = ess_GESS / total_evals[j,2,i]
            ESS_per_likelihood_AGESS[j,i] = ess_AGESS / total_evals[j,3,i]
            ESS_per_likelihood_ARW[j,i] = ess_ARW / total_evals[j,4,i]
            
        else
            sim = load(string(dir ,"//", P_vec[j], "_Cov//Sim", i,".jld2")) 

            percent_zero[j,i] = sum(sim["μ"] .== 0) / 1000
            ESS_per_second_ESS[j,i] = sim["Eff_SS_ESS"] / sim["time_non_burnin_ESS"]
            ESS_per_second_GESS[j,i] = sim["Eff_SS_GESS"] / sim["time_non_burnin_GESS"]
            ESS_per_second_AGESS[j,i] = sim["Eff_SS_AGESS"] / sim["time_non_burnin_AGESS"]
            ESS_per_second_ARW[j,i] = sim["Eff_SS_ARW"] / sim["time_non_burnin_ARW"]

            ESS_per_iteration_ESS[j,i] = sim["Eff_SS_ESS"] / (10000 * P_vec[j] * 0.75)
            ESS_per_iteration_GESS[j,i] = sim["Eff_SS_GESS"] / (10000 * P_vec[j] * 0.75)
            ESS_per_iteration_AGESS[j,i] = sim["Eff_SS_AGESS"] / (10000 * P_vec[j] * 0.75)
            ESS_per_iteration_ARW[j,i] = sim["Eff_SS_ARW"] / (30000 * P_vec[j] * (275/300))

            ESS_per_likelihood_ESS[j,i] = sim["Eff_SS_ESS"] / sim["ESS_evals"]
            ESS_per_likelihood_GESS[j,i] = sim["Eff_SS_GESS"] / sim["GESS_evals"]
            ESS_per_likelihood_AGESS[j,i] = sim["Eff_SS_AGESS"] / sim["AGESS_evals"]
            ESS_per_likelihood_ARW[j,i] = sim["Eff_SS_ARW"] / sim["ARW_evals"]

            total_time_ESS[j,i] = sim["total_time_ESS"]
            total_time_GESS[j,i] = sim["total_time_GESS"]
            total_time_AGESS[j,i] = sim["total_time_AGESS"]
            total_time_ARW[j,i] = sim["total_time_ARW"]
        end
    end
end

## Set non-converged values equal to 0.01
ESS_per_second_GESS[ESS_per_second_GESS .< 0] .= 0.001
ESS_per_second_AGESS[ESS_per_second_AGESS .< 0] .= 0.001
ESS_per_second_ESS[ESS_per_second_ESS .< 0] .= 0.001
ESS_per_second_ARW[ESS_per_second_ARW .< 0] .= 0.001


colors = [:red :blue :green :purple]
shapes = [:circle :rect :utriangle :diamond]
labels = ["ESS" "GESS" "AGESS" "ARW"]

ESS_2 = [ESS_per_second_ESS[1,:]'; ESS_per_second_GESS[1,:]'; ESS_per_second_AGESS[1,:]'; ESS_per_second_ARW[1,:]']
scatter1 = scatter(percent_zero[1,:], ESS_2', title = "D = 2", yscale=:log10, yticks = [1000, 10000, 100000], ylims = [1000, 100000], markerstrokewidth=0, markersize = 6, color=colors, shape=shapes, framestyle = :axes, legend = false)
ylabel!("Effective Sample Size per Second")
xlabel!("Proportion of μ that are zero")

ESS_2 = [ESS_per_second_ESS[2,:]'; ESS_per_second_GESS[2,:]'; ESS_per_second_AGESS[2,:]'; ESS_per_second_ARW[2,:]']
scatter2 = scatter(percent_zero[2,:], ESS_2', title = "D = 10", yscale=:log10, yticks = [100, 1000, 10000, 100000], ylims = [100, 100000], markerstrokewidth=0, markersize = 6, color=colors, shape=shapes, framestyle = :axes, legend = false)
ylabel!("Effective Sample Size per Second")
xlabel!("Proportion of μ that are zero")

ESS_2 = [ESS_per_second_ESS[3,:]'; ESS_per_second_GESS[3,:]'; ESS_per_second_AGESS[3,:]'; ESS_per_second_ARW[3,:]']
scatter3 = scatter(percent_zero[3,:], ESS_2', title = "D = 50", yscale=:log10, yticks = [10, 100, 1000], ylims = [10, 1000], markerstrokewidth=0, markersize = 6, color=colors, shape=shapes, framestyle = :axes, legend = false)
ylabel!("Effective Sample Size per Second")
xlabel!("Proportion of μ that are zero")

## Dummy subplot holding only the shared legend, spanning the full figure width
legend_plot = scatter(fill(NaN, 1, 4), color = colors, shape = shapes, markerstrokewidth = 0, markersize = 8,
                       label = labels, legend = :inside, legend_columns = 4, framestyle = :none, grid = false)

plot(scatter1, scatter2, scatter3, legend_plot, layout = @layout([A B C; D{0.12h}]), margin = 10Plots.mm,
     fontfamily = "Computer Modern", titlefontsize = 28, guidefontsize = 24, tickfontsize = 20,
     legendfontsize = 20, grid = false)
plot!(size = (2100, 800))

savefig(string(dir ,"//Results.pdf"))

### Print out Summary Statistics 

ESS_per_second_GESS[ESS_per_second_GESS .==  0.001] .= NaN
ESS_per_second_AGESS[ESS_per_second_AGESS .==  0.001] .= NaN
ESS_per_second_ESS[ESS_per_second_ESS .==  0.001] .= NaN
ESS_per_second_ARW[ESS_per_second_ARW .==  0.001] .= NaN


ESS_per_iteration_GESS[ESS_per_iteration_GESS .< 0] .= NaN
ESS_per_iteration_AGESS[ESS_per_iteration_AGESS .< 0] .= NaN
ESS_per_iteration_ESS[ESS_per_iteration_ESS .< 0] .= NaN
ESS_per_iteration_ARW[ESS_per_iteration_ARW .< 0] .= NaN

ESS_per_likelihood_GESS[ESS_per_likelihood_GESS .< 0] .= NaN
ESS_per_likelihood_AGESS[ESS_per_likelihood_AGESS .< 0] .= NaN
ESS_per_likelihood_ESS[ESS_per_likelihood_ESS .< 0] .= NaN
ESS_per_likelihood_ARW[ESS_per_likelihood_ARW .< 0] .= NaN

write_ess_summary_table(string(dir, "//ESS.txt"),
                         ESS_per_second_ESS', ESS_per_iteration_ESS', ESS_per_likelihood_ESS', total_time_ESS', ["ESS"])

write_ess_summary_table(string(dir, "//GESS.txt"),
                         ESS_per_second_GESS', ESS_per_iteration_GESS', ESS_per_likelihood_GESS', total_time_GESS', ["GESS"])

write_ess_summary_table(string(dir, "//AGESS.txt"),
                         ESS_per_second_AGESS', ESS_per_iteration_AGESS', ESS_per_likelihood_AGESS', total_time_AGESS', ["AGESS"])

write_ess_summary_table(string(dir, "//ARW.txt"),
                         ESS_per_second_ARW', ESS_per_iteration_ARW', ESS_per_likelihood_ARW', total_time_ARW', ["ARW"])
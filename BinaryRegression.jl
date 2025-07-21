using LogExpFunctions, Distributions, LinearAlgebra, JLD2, Random, StatsBase, RCall, StatsPlots

include("AGESS.jl")
dir = "//Users//ndm34//Projects//AGESS_Simulation//BinaryRegression"

## Function to generate the data
function generate_data(N::T, P::T, ν::Y = 6.0) where {Y<:AbstractFloat, T<:Integer}
    β = randn(P) * (2 * log(P))^(1.0 / 4)
    β .*= sqrt(rand(Gamma(ν/2, 2/ ν)))
    x = randn(N, P) .+ randn() * 0.5
    μ = zeros(N)
    y = zeros(N)

    for i in 1:N
        μ[i] = max(0, dot(x[i,:], β))
        y[i] = rand(Binomial(1, logistic(μ[i])), 1)[1]
    end


    return β, x, μ, y
end

## Likelihood
function log_likelihood(β::AbstractVector{Y}, x::AbstractMatrix{Y}, y::AbstractVector{Y}) where {Y <:AbstractFloat}
    log_lik = 0
    for i in eachindex(y)
        p = logistic(max(0, dot(x[i,:], β)))
        log_lik += y[i] * log(p) + (1 - y[i]) * log(1-p)
    end

    return log_lik
end

## Prior on β
function log_prior(β::AbstractVector{Y}) where {Y <:AbstractFloat}
    log_p = -0.5 * dot(β, β)
    return log_p
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
median_ESS_per_second_ESS = zeros(3,100)
median_ESS_per_second_GESS = zeros(3,100)
median_ESS_per_second_AGESS = zeros(3,100)
median_ESS_per_second_ARW = zeros(3, 100)

P_vec = [2, 10, 50]

Random.seed!(1234)
for j in 1:3
    Σ = diagm(ones(P_vec[j]))
    for i in 1:100
        if !isfile(string(dir ,"//", P_vec[j], "_Cov//Sim", i,".jld2"))
            Random.seed!(i)
            β, x, μ, y = generate_data(1000, P_vec[j])
            sum(μ .== 0)

            ## Elliptical Slice Sampling
            β_samp_ESS = zeros(5000 * P_vec[j], P_vec[j])
            t2 = time()
            time_non_burnin_ESS = ESS(β_samp_ESS, β -> log_likelihood(β, x, y), Σ)
            total_time_ESS = time() - t2
            

            ## Generalized Elliptical Slice Sampling
            β_samp_GESS = zeros(5000 * P_vec[j], P_vec[j])
            t2 = time()
            μ1 = zeros(P_vec[j])
            time_non_burnin_GESS = GESS(β_samp_GESS, β -> (log_likelihood(β, x, y) + log_prior(β)), μ1, Σ)
            total_time_GESS = time() - t2


            ## Adaptive Generalized Elliptical Slice Sampling
            β_samp_AGESS = zeros(5000 * P_vec[j], P_vec[j])
            t2 = time()
            μ1 = zeros(P_vec[j])
            time_non_burnin_AGESS = AGESS(β_samp_AGESS, β -> log_likelihood(β, x, y), log_prior, μ1, Σ, true)
            total_time_AGESS = time() - t2
            

            ## Adaptive Random Walk
            β_samp_ARW = zeros(5000 * P_vec[j], P_vec[j])
            t2 = time()
            μ1 = zeros(P_vec[j])
            time_non_burnin_ARW = ARW(β_samp_ARW, β -> log_likelihood(β, x, y), log_prior, 1000, 0.01, μ1, Σ)
            total_time_ARW = time() - t2
            

            ## Get Multivariate Effective Sample Size using R package "stableGR"
            β_samp_ESS_1 = β_samp_ESS[2500*P_vec[j]:5000*P_vec[j],:]
            β_samp_GESS_1 = β_samp_GESS[2500*P_vec[j]:5000*P_vec[j],:]
            β_samp_AGESS_1 = β_samp_AGESS[2500*P_vec[j]:5000*P_vec[j],:]
            β_samp_ARW_1 = β_samp_ARW[2500*P_vec[j]:5000*P_vec[j],:]
            @rput β_samp_ESS_1
            @rput β_samp_AGESS_1
            @rput β_samp_GESS_1
            @rput β_samp_ARW_1

            R"""
            library(stableGR)
            ess_ESS <- n.eff(β_samp_ESS_1)$n.eff
            ess_GESS <- n.eff(β_samp_GESS_1)$n.eff
            ess_AGESS <- n.eff(β_samp_AGESS_1)$n.eff
            ess_ARW <- n.eff(β_samp_ARW_1)$n.eff

            ess <- matrix(0, 4, ncol(β_samp_ESS_1))
            for(p in 1:ncol(β_samp_ESS_1)){
                ess[1, p] <- n.eff(β_samp_ESS_1[,p])$n.eff
                ess[2, p] <- n.eff(β_samp_GESS_1[,p])$n.eff
                ess[3, p] <- n.eff(β_samp_AGESS_1[,p])$n.eff
                ess[4, p] <- n.eff(β_samp_ARW_1[,p])$n.eff
            }

            median_ess_ESS <- median(ess[1,])
            median_ess_GESS <- median(ess[2,])
            median_ess_AGESS <- median(ess[3,])
            median_ess_ARW <- median(ess[4,])
            """

            @rget ess_ESS
            @rget ess_GESS
            @rget ess_AGESS
            @rget ess_ARW
            @rget median_ess_ESS
            @rget median_ess_GESS
            @rget median_ess_AGESS
            @rget median_ess_ARW


            # Save output of Simulation
            save(string(dir ,"//", P_vec[j], "_Cov//Sim", i,".jld2"), Dict("β_samp_ESS" => β_samp_ESS, 
                                                                           "β_samp_GESS" => β_samp_GESS,
                                                                           "β_samp_AGESS" => β_samp_AGESS,
                                                                           "β_samp_ARW" => β_samp_ARW,
                                                                           "time_non_burnin_ESS" => time_non_burnin_ESS,
                                                                           "time_non_burnin_GESS" => time_non_burnin_GESS,
                                                                           "time_non_burnin_AGESS" => time_non_burnin_AGESS,
                                                                           "time_non_burnin_ARW" => time_non_burnin_ARW,
                                                                           "total_time_ESS" => total_time_ESS,
                                                                           "total_time_GESS" => total_time_GESS,
                                                                           "total_time_AGESS" => total_time_AGESS,
                                                                           "total_time_ARW" => total_time_ARW,
                                                                           "Eff_SS_ESS" => ess_ESS,
                                                                           "Eff_SS_GESS" => ess_GESS,
                                                                           "Eff_SS_AGESS" => ess_AGESS,
                                                                           "Eff_SS_ARW" => ess_ARW,
                                                                           "Median_Eff_SS_ESS" => median_ess_ESS,
                                                                           "Median_Eff_SS_GESS" => median_ess_GESS,
                                                                           "Median_Eff_SS_AGESS" => median_ess_AGESS,
                                                                           "Median_Eff_SS_ARW" => median_ess_ARW,
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

            median_ESS_per_second_ESS[j,i] = median_ess_ESS / time_non_burnin_ESS
            median_ESS_per_second_GESS[j,i] = median_ess_GESS / time_non_burnin_GESS
            median_ESS_per_second_AGESS[j,i] = median_ess_AGESS / time_non_burnin_AGESS
            median_ESS_per_second_ARW[j,i] = median_ess_ARW / time_non_burnin_ARW
            
        else
            sim = load(string(dir ,"//", P_vec[j], "_Cov//Sim", i,".jld2")) 

            percent_zero[j,i] = sum(sim["μ"] .== 0) / 1000
            ESS_per_second_ESS[j,i] = sim["Eff_SS_ESS"] / sim["time_non_burnin_ESS"]
            ESS_per_second_GESS[j,i] = sim["Eff_SS_GESS"] / sim["time_non_burnin_GESS"]
            ESS_per_second_AGESS[j,i] = sim["Eff_SS_AGESS"] / sim["time_non_burnin_AGESS"]
            ESS_per_second_ARW[j,i] = sim["Eff_SS_ARW"] / sim["time_non_burnin_ARW"]

            median_ESS_per_second_ESS[j,i] = sim["Median_Eff_SS_ESS"] / sim["time_non_burnin_ESS"]
            median_ESS_per_second_GESS[j,i] = sim["Median_Eff_SS_GESS"] / sim["time_non_burnin_GESS"]
            median_ESS_per_second_AGESS[j,i] = sim["Median_Eff_SS_AGESS"] / sim["time_non_burnin_AGESS"]
            median_ESS_per_second_ARW[j,i] = sim["Median_Eff_SS_ARW"] / sim["time_non_burnin_ARW"]
        end
    end
end

ESS_2 = [ESS_per_second_ESS[1,:]'; ESS_per_second_GESS[1,:]'; ESS_per_second_AGESS[1,:]'; ESS_per_second_ARW[1,:]']
scatter1 = scatter(percent_zero[1,:], ESS_2', label=["ESS" "GESS" "AGESS" "ARW"], title = "Effective Sample Size per Second (P = 2)", yscale=:log10, yticks = [100, 1000, 10000], ylims = [100, 12000], markerstrokewidth=0)
ylabel!("Effective Sample Size per Second")
xlabel!("Proportion of μ that are zero")
box1 = boxplot(["ESS" "GESS" "AGESS" "ARW"], ESS_2', title = "Effective Sample Size (P = 2)", legend = false, yscale=:log10, yticks = [100, 1000, 10000], ylims = [100, 12000], markerstrokewidth=0)
ylabel!("Effective Sample Size per Second")

ESS_2 = [median_ESS_per_second_ESS[1,:]'; median_ESS_per_second_GESS[1,:]'; median_ESS_per_second_AGESS[1,:]'; median_ESS_per_second_ARW[1,:]']
box1_single = boxplot(["ESS" "GESS" "AGESS" "ARW"], ESS_2', title = "Univariate Median Effective Sample Size (P = 2)", legend = false)
ylabel!("Effective Sample Size per Second")

ESS_2 = [ESS_per_second_ESS[2,:]'; ESS_per_second_GESS[2,:]'; ESS_per_second_AGESS[2,:]'; ESS_per_second_ARW[2,:]']
scatter2 = scatter(percent_zero[2,:], ESS_2', label=["ESS" "GESS" "AGESS" "ARW"], title = "Effective Sample Size per Second (P = 10)", yscale=:log10, yticks = [10, 100, 1000, 10000], ylims = [10, 10000], markerstrokewidth=0)
ylabel!("Effective Sample Size per Second")
xlabel!("Proportion of μ that are zero")
box2 = boxplot(["ESS" "GESS" "AGESS" "ARW"], ESS_2', title = "Effective Sample Size per Second (P = 10)", legend = false, yscale=:log10, yticks = [10, 100, 1000, 10000], ylims = [10, 10000], markerstrokewidth=0)
ylabel!("Effective Sample Size per Second")

ESS_2 = [median_ESS_per_second_ESS[2,:]'; median_ESS_per_second_GESS[2,:]'; median_ESS_per_second_AGESS[2,:]'; median_ESS_per_second_ARW[2,:]']
box2_single = boxplot(["ESS" "GESS" "AGESS" "ARW"], ESS_2', title = "Univariate Median Effective Sample Size (P = 10)", legend = false)
ylabel!("Effective Sample Size per Second")



ESS_2 = [ESS_per_second_ESS[3,:]'; ESS_per_second_GESS[3,:]'; ESS_per_second_AGESS[3,:]'; ESS_per_second_ARW[3,:]']
scatter3 = scatter(percent_zero[3,:], ESS_2', label=["ESS" "GESS" "AGESS" "ARW"], title = "Effective Sample Size per Second (P = 50)", yscale=:log10, yticks = [1, 10, 100, 1000], ylims = [1, 1000], markerstrokewidth=0)
ylabel!("Effective Sample Size per Second")
xlabel!("Proportion of μ that are zero")
box3 = boxplot(["ESS" "GESS" "AGESS" "ARW"], ESS_2', title = "Effective Sample Size per Second (P = 50)", legend = false, yscale=:log10, yticks = [1, 10, 100, 1000], ylims = [1, 1000], markerstrokewidth=0)
ylabel!("Effective Sample Size per Second")

ESS_2 = [median_ESS_per_second_ESS[3,:]'; median_ESS_per_second_GESS[3,:]'; median_ESS_per_second_AGESS[3,:]'; median_ESS_per_second_ARW[3,:]']
box3_single = boxplot(["ESS" "GESS" "AGESS" "ARW"], ESS_2', title = "Univariate Median Effective Sample Size (P = 50)", legend = false, yscale=:log10)
ylabel!("Effective Sample Size per Second")

plot(box1, scatter1, box2, scatter2, box3, scatter3, layout = (3,2),margin= 10Plots.mm)
plot!(size = (2000,1500))

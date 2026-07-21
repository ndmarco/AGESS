using LinearAlgebra, Random, Distributions, Printf, Statistics


function ESS_SingleStep(x::AbstractMatrix{Y}, z::AbstractVector{Y}, log_likelihood::Function, 
                        μ::AbstractVector{Y}, Σ_chol::LowerTriangular{Y, <:AbstractMatrix{Y}},
                        current_likelihood::Y, i::T) where {Y<:AbstractFloat, T<:Integer}
    P = size(x)[2]

    ## Propose new z from N(0, Σ)
    randn!(z)
    lmul!(Σ_chol, z)
    y = current_likelihood + log(rand())

    ## Propose Initial Angle
    θ = rand() * 2 * π
    θ_min = θ - 2 * π
    θ_max = θ

    ## Propose initial first move
    @views @. x[i,:] = ((x[i-1,:] - μ) * cos(θ) +  z * sin(θ)) + μ
    @views log_lik_prop = log_likelihood(x[i,:])
    current_likelihood = log_lik_prop
    ## Get number of likelihood evaluations
    num_evals = 1

    ## Check to make sure that posterior pdfs are computable
    if isnan(log_lik_prop)
        log_lik_prop = y - 1.0
    end
    if !isfinite(log_lik_prop)
        log_lik_prop = y - 1.0
    end

    while log_lik_prop <= y
        if θ < 0
            θ_min = θ
        else
            θ_max = θ
        end

        ## Propose new angle
        θ = θ_min + rand() * (θ_max - θ_min)
        @views @. x[i,:] = ((x[i-1,:] - μ) * cos(θ) +  z * sin(θ)) + μ
        @views log_lik_prop = log_likelihood(x[i,:])
        current_likelihood = log_lik_prop
        num_evals += 1

        ## Check to make sure that posterior pdfs are computable
        if isnan(log_lik_prop)
            log_lik_prop = y - 1.0
        end
        if !isfinite(log_lik_prop)
            log_lik_prop = y - 1.0
        end
    end
    
    return current_likelihood, num_evals
end


function ESS(x::AbstractMatrix{Y}, log_likelihood::Function, μ::AbstractVector{Y}, Σ::AbstractMatrix{Y}; burnin::Y = 0.5) where {Y<:AbstractFloat}
    P = size(x)[2]
    n_MCMC = size(x)[1]
    z = zeros(P)
    burnin_num = floor(Int64, burnin * n_MCMC)

    Σ_chol = cholesky(Σ)
    t1 = time()

    @views current_likelihood = log_likelihood(x[1,:])
    total_num_likevals = 1
    num_lik_iter = 0
    for i in 2:n_MCMC
        if i == burnin_num
            t1 = time()
            total_num_likevals = 0
        end

        current_likelihood, num_lik_iter = ESS_SingleStep(x, z, log_likelihood, μ, Σ_chol.L, current_likelihood, i)
        total_num_likevals += num_lik_iter

        ## Populate next value in Markov Chain
        if i < n_MCMC
            @views x[i+1,:] .= x[i,:]
        end

        ## Update User
        if (i % 1000) == 0
            println("MCMC iter: ", i)
            log_lik = @sprintf("%.2f", current_likelihood)
            println("Log Likelihood: ", log_lik)
        end
    end

    return time() - t1, total_num_likevals
end

function GESS_SingleStep(x::AbstractMatrix{Y}, z::AbstractVector{Y}, log_posterior::Function, ph::AbstractVector{Y}, 
                         μ::AbstractVector{Y}, Σ_chol::LowerTriangular{Y, <:AbstractMatrix{Y}}, i::T, 
                         current_posterior::Y, ν::Y) where {Y<:AbstractFloat, T<:Integer}
    P = size(x)[2]

    α = 0.5 * (ν + P)
    @views ph .= (x[i,:] .- μ)
    ldiv!(Σ_chol, ph)
    β = 0.5 * (ν + dot(ph,ph))
    s = rand(InverseGamma(α, β))


    current_posterior, num_evals = ESS_SingleStep(x, z, b -> (log_posterior(b) - dMvT(b, μ, Σ_chol, ph, ν, P)), μ, sqrt(s) * Σ_chol, current_posterior, i)

    return current_posterior, num_evals
end

function GESS(x::AbstractMatrix{Y}, log_posterior::Function, μ::AbstractVector{Y}, Σ::AbstractMatrix{Y};
              ν::Y = 6.0, burnin::Y = 0.5) where {Y<:AbstractFloat}
    n_MCMC = size(x)[1]
    P = size(x)[2]
    z = zeros(P)
    Σ_chol = cholesky(Σ)
    ph = zeros(P)
    burnin_num = floor(Int64, burnin * n_MCMC)

    @views current_posterior = log_posterior(x[1,:])
    total_num_likevals = 1
    num_lik_iter = 0

    t1 = time()
    for i in 2:n_MCMC
        if i == burnin_num
            t1 = time()
            total_num_likevals = 0
        end

        current_posterior, num_lik_iter = GESS_SingleStep(x, z, log_posterior, ph, μ, Σ_chol.L, i, current_posterior, ν)
        total_num_likevals += num_lik_iter

        ## Populate next value in Markov Chain
        if i < n_MCMC
            @views x[i+1,:] .= x[i,:]
        end

        ## Update User
        if (i % 1000) == 0
            println("MCMC iter: ", i)
            @views log_pos = @sprintf("%.2f", current_posterior)
            println("Log Posterior: ", log_pos)
        end
    end

    return time() - t1, total_num_likevals
end



function dMvT(x::AbstractVector{Y}, μ::AbstractVector{Y}, Σ_chol::LowerTriangular{Y, <:AbstractMatrix{Y}}, 
              ph::AbstractVector{Y}, ν::Y, P::T) where {Y<:AbstractFloat, T<:Integer}
    ph .= (x .- μ)
    ldiv!(Σ_chol, ph)
    pdf::Float64 = -(ν + P) * 0.5 * log1p((dot(ph, ph) / ν))
    return pdf
end

function dMvN(x::AbstractVector{Y}, μ::AbstractVector{Y}, Σ_chol::LowerTriangular{Y, <:AbstractMatrix{Y}}, 
              ph::AbstractVector{Y}) where {Y<:AbstractFloat}
    ph .= (x .- μ)
    ldiv!(Σ_chol, ph)
    pdf::Float64 = -0.5 * dot(ph, ph)
    return pdf
end

function cond_rMvT!(z::AbstractVector{Y}, x::AbstractVector{Y}, μ::AbstractVector{Y}, 
                    Σ_chol::LowerTriangular{Y, <:AbstractMatrix{Y}}, ν::Y, 
                    ph::AbstractVector{Y}, P::T) where {Y<:AbstractFloat, T<:Integer}
    randn!(z)
    lmul!(Σ_chol, z)
    ph .= (x .- μ)
    ldiv!(Σ_chol, ph)
    d = dot(ph, ph)
    ṽ = ν + P

    scale = sqrt((ν + d) / ṽ) / sqrt(rand(Gamma(ṽ / 2, 2 / ṽ)))
    @. z = z * scale + μ

    return nothing
end

function dMvT_1d(x::Y, μ::Y, σ::Y, ν::Y) where {Y<:AbstractFloat}
    pdf::Float64 = -(ν + 1) * 0.5 * log1p(((x - μ)^2 / (σ^2)) / ν)
    return pdf
end

function dMvN_1d(x::Y, μ::Y, σ::Y) where {Y<:AbstractFloat} 
    pdf::Float64 =  - 0.5 * ((x - μ)^2 / (σ^2))
    return pdf
end

function cond_rMvT_1d!(x::Y, μ::Y, σ::Y, ν::Y) where {Y<:AbstractFloat}
    z::Float64 = σ * randn() 
    d = (x - μ)^2 / (σ^2)
    ṽ = ν + 1
    z *= sqrt((ν + d) / ṽ)
    z *= 1 / sqrt(rand(Gamma(ṽ/2, 2/ ṽ)))
    z += μ

    return z
end
    

function AGESS_SingleStep(x::AbstractMatrix{Y}, z::AbstractVector{Y}, log_posterior::Function, 
                          ph::AbstractVector{Y}, t_dist::Bool, ν::Y, μ_adapt::AbstractVector{Y},
                          Σ_chol_adapt::LowerTriangular{Y, <:AbstractMatrix{Y}}, 
                          current_posterior::Y, i::T) where {Y<:AbstractFloat, T<:Integer}
    P = size(x)[2]
    y::Float64 = 0.0
    L_star::Float64 = 0.0

    ## Propose new z from N(μ, Σ)
    if t_dist == true
        @views cond_rMvT!(z, x[i,:], μ_adapt, Σ_chol_adapt, ν, ph, P)
    else
        randn!(z)
        lmul!(Σ_chol_adapt, z)
        z .+= μ_adapt
    end

    @views y = current_posterior + log(rand())
    if t_dist == true
        @views y -= dMvT(x[i,:], μ_adapt, Σ_chol_adapt, ph, ν, P)::Float64
    else
        @views y -= dMvN(x[i,:], μ_adapt, Σ_chol_adapt, ph)::Float64
    end

    ## Propose Initial Angle
    θ = rand() * 2 * π
    θ_min = θ - 2 * π
    θ_max = θ

    ## Propose initial first move
    @views @. x[i,:] = ((x[i-1,:] - μ_adapt) * cos(θ) +  (z - μ_adapt) * sin(θ)) + μ_adapt
    @views L_star = log_posterior(x[i,:])::Float64 
    current_posterior = L_star
    if t_dist == true
        @views L_star -= dMvT(x[i,:], μ_adapt, Σ_chol_adapt, ph, ν, P)::Float64
    else
        @views L_star -= dMvN(x[i,:], μ_adapt, Σ_chol_adapt, ph)::Float64
    end
    ## Get number of likelihood evaluations
    num_evals = 1

    ## Check to make sure that posterior pdfs are computable
    if isnan(L_star)
        L_star = y - 1.0
    end
    if !isfinite(L_star)
        L_star = y - 1.0
    end

    while L_star <= y
        if θ < 0
            θ_min = θ
        else
            θ_max = θ
        end

        ## Propose new angle
        θ = θ_min + rand() * (θ_max - θ_min)
        @views  @. x[i,:] = ((x[i-1,:] - μ_adapt) * cos(θ) +  (z - μ_adapt) * sin(θ)) + μ_adapt
        @views L_star = log_posterior(x[i,:])::Float64 
        current_posterior = L_star
        if t_dist == true
            @views L_star -= dMvT(x[i,:], μ_adapt, Σ_chol_adapt, ph, ν, P)::Float64
        else
            @views L_star -= dMvN(x[i,:], μ_adapt, Σ_chol_adapt, ph)::Float64
        end
        num_evals += 1

        ## Check to make sure that posterior pdfs are computable
        if isnan(L_star)
            L_star = y - 1.0
        end
        if !isfinite(L_star)
            L_star = y - 1.0
        end
    end

    return current_posterior, num_evals
end


function AGESS(x::AbstractMatrix{Y}, log_posterior::Function, 
                μ::AbstractVector{Y}, Σ::AbstractMatrix{Y},
                t_dist::Bool; ν::Y = 6.0, burnin::Y = 0.5, ϵ::Y = 0.1, 
                single_step_prop::Y = 0.05, β::Y = 0.5) where {Y<:AbstractFloat}
    P = size(x)[2]
    n_MCMC = size(x)[1]
    z = zeros(P)
    burnin_num = floor(Int64, burnin * n_MCMC)
    t1 = time()

    μ_adapt = copy(μ)
    μ_adapt_ph = copy(μ)
    ph = similar(μ_adapt)

    Σ_chol = cholesky(Σ)
    Σ_chol_adapt = deepcopy(Σ_chol)
    Σ_chol_adapt_ph = deepcopy(Σ_chol)

    μ_0 = zeros(P)
    ph_cholesky_update = ones(P)
    w_const = max(2/3, ((cbrt(P) - 1) / cbrt(P)))
    N_J = 2
    n_j = 2

    @views current_posterior = log_posterior(x[1,:])
    total_num_likevals = 1
    num_lik_iter = 0

    for i in 2:n_MCMC
        if i == burnin_num
            t1 = time()
            total_num_likevals = 0
        end

        if P >= 10
            if i < burnin_num * single_step_prop
                current_posterior, num_lik_iter = AGESS_SingleStep_1d(x, log_posterior, t_dist, ν, μ_adapt, 
                                                                      Σ_chol_adapt.L, current_posterior, i)
                total_num_likevals += num_lik_iter
            else
                if rand() > ϵ
                    current_posterior, num_lik_iter = AGESS_SingleStep(x, z, log_posterior, ph, t_dist, ν, μ_adapt,
                                                                       Σ_chol_adapt.L, current_posterior, i)
                    total_num_likevals += num_lik_iter
                elseif rand() > 0.5
                    current_posterior, num_lik_iter = AGESS_SingleStep_1d(x, log_posterior, t_dist, ν, μ_adapt, 
                                                                          Σ_chol_adapt.L, current_posterior, i)
                    total_num_likevals += num_lik_iter
                else
                    current_posterior, num_lik_iter = AGESS_SingleStep(x, z, log_posterior, ph, t_dist, ν, μ_0, 
                                                                       Σ_chol.L, current_posterior, i)
                    total_num_likevals += num_lik_iter
                end
            end
        else
            if rand() > ϵ
                current_posterior, num_lik_iter = AGESS_SingleStep(x, z, log_posterior, ph, t_dist, ν, μ_adapt,
                                                                   Σ_chol_adapt.L, current_posterior, i)
                total_num_likevals += num_lik_iter
            else
                current_posterior, num_lik_iter = AGESS_SingleStep(x, z, log_posterior, ph, t_dist, ν, μ_0, 
                                                                   Σ_chol.L, current_posterior, i)
                total_num_likevals += num_lik_iter
            end

        end
        
        w_i = i^(-w_const)
        Σ_chol_adapt_ph.U .= sqrt((1 - w_i)) .*  Σ_chol_adapt_ph.U
        @views ph_cholesky_update .= sqrt(w_i) .* (x[i,:] .- μ_adapt_ph)
        lowrankupdate!(Σ_chol_adapt_ph, ph_cholesky_update)
        @views μ_adapt_ph .= (1 - w_i) * μ_adapt_ph +  w_i * x[i,:]
        
        ## Adapt mean and covariance
        if i == N_J
            Σ_chol_adapt.U .= Σ_chol_adapt_ph.U
            μ_adapt .= μ_adapt_ph
            n_j += 1
            N_J += floor(Int64, n_j^β)
        end

        ## Populate next value in Markov Chain
        if i < n_MCMC
            @views x[i+1,:] .= x[i,:]
        end

        # Update User
        if P >= 10
            if i < burnin_num * single_step_prop
                if (i % 25) == 0
                    println("MCMC iter: ", i)
                    @views log_lik = @sprintf("%.2f", current_posterior)
                    println("Log Posterior: ", log_lik)
                end
            else
                if (i % 1000) == 0
                    println("MCMC iter: ", i)
                    @views log_lik = @sprintf("%.2f", current_posterior)
                    println("Log Posterior: ", log_lik)
                end
            end
        else
            if (i % 1000) == 0
                println("MCMC iter: ", i)
                @views log_lik = @sprintf("%.2f", current_posterior)
                println("Log Posterior: ", log_lik)
            end
        end
        
    end

    return time() - t1, Σ_chol_adapt.L * Σ_chol_adapt.U, μ_adapt, total_num_likevals
end

function AGESS_SingleStep_1d(x::AbstractMatrix{Y}, log_posterior::Function, 
                             t_dist::Bool, ν::Y, μ_adapt::AbstractVector{Y}, 
                             Σ_chol_adapt::LowerTriangular{Y, <:AbstractMatrix{Y}}, 
                             current_posterior::Y, i::T) where {Y<:AbstractFloat, T<:Integer}

    P = size(x)[2]
    z::Float64 = 0.0
    y::Float64 = 0.0
    L_star::Float64 = 0.0
    ## Get number of likelihood evaluations
    num_evals = 0
    for j in randperm(P)
        
        ## Propose new z from N(0, Σ)
        if t_dist == true
            z = cond_rMvT_1d!(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j], ν)
        else
            z = Σ_chol_adapt[j,j] * randn() + μ_adapt[j]
        end

        @views y = current_posterior + log(rand())
        if t_dist == true
            y -= dMvT_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j], ν)
        else
            y -= dMvN_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j])
        end

        ## Propose Initial Angle
        θ = rand() * 2 * π
        θ_min = θ - 2 * π
        θ_max = θ

        ## Propose initial first move
        x[i,j] = ((x[i-1,j] - μ_adapt[j]) * cos(θ) +  (z - μ_adapt[j]) * sin(θ)) + μ_adapt[j]
        @views L_star = log_posterior(x[i,:])
        current_posterior = L_star
        if t_dist == true
            L_star -= dMvT_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j], ν)
        else
            L_star -= dMvN_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j])
        end

        num_evals += 1

        ## Check to make sure that posterior pdfs are computable
        if isnan(L_star)
            L_star = y - 1.0
        end
        if !isfinite(L_star)
            L_star = y - 1.0
        end

        while L_star <= y
            if θ < 0
                θ_min = θ
            else
                θ_max = θ
            end

            ## Propose new angle
            θ = θ_min + rand() * (θ_max - θ_min)
            x[i,j] = ((x[i-1,j] - μ_adapt[j]) * cos(θ) +  (z - μ_adapt[j]) * sin(θ)) + μ_adapt[j]
            @views L_star = log_posterior(x[i,:])
            current_posterior = L_star
            if t_dist == true
                L_star -= dMvT_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j], ν)
            else
                L_star -= dMvN_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j])
            end
            num_evals += 1

            ## Check to make sure that posterior pdfs are computable
            if isnan(L_star)
                L_star = y - 1.0
            end   
            if !isfinite(L_star)
                L_star = y - 1.0
            end
        end
    end

    return current_posterior, num_evals
end


function ARW(x::AbstractMatrix{Y}, log_likelihood::Function, log_prior::Function, block1::T,
             init_σ::Y, μ::AbstractVector{Y}, Σ::AbstractMatrix{Y}; tuning_step = 25, burnin::Y=0.5) where {Y<:AbstractFloat, T<:Integer}
    P = size(x)[2]
    n_MCMC = size(x)[1]
    z = zeros(P)
    σ_rw = ones(P) * init_σ
    acceptance = zeros(P)
    burnin_num = floor(Int64, burnin * n_MCMC)
    Σ_chol = cholesky(Σ)
    ph_cholesky_update = ones(P)
    @views current_logposterior = log_likelihood(x[1,:]) + log_prior(x[1,:])
    total_num_likevals = 1
    prop_logposterior = 0.0
    for i in 2:block1
        # Perform random walk proposal
        for j in 1:P
            @views z .= x[i,:]
            z[j] = x[i,j] + randn() * σ_rw[j]
            prop_logposterior = log_likelihood(z) + log_prior(z)
            total_num_likevals += 1
            @views accept_prob = (prop_logposterior) - (current_logposterior)
            if isfinite(accept_prob)
                if log(rand()) < accept_prob
                    x[i,j] = z[j]
                    acceptance[j] += 1
                    current_logposterior = prop_logposterior
                end
            end
        end

        ## Tune Acceptance
        if (i % tuning_step) == 0
            println("MCMC iter: ", i)
            println("Acceptance: ", round(mean(acceptance) / tuning_step, digits=3))
            @views log_lik = @sprintf("%.2f", current_logposterior)
            println("Log Posterior: ", log_lik)
            for j in 1:P
                σ_rw[j] = exp(log(σ_rw[j]) + ((acceptance[j] / tuning_step) - 0.44))
                acceptance[j] = 0
            end
        end

        ## Update Markov Chain
        @views x[i+1,:] .= x[i,:]
    end

    half_warm_up_block = floor(Int64, block1/2)
    @views Σ_chol_adapt = cholesky(cov(x[half_warm_up_block:block1,:]))
    @views x_mean = mean(x[half_warm_up_block:block1,:], dims=1)[1,:]
    Σ_rw = Σ_chol_adapt.L .* (2.38 / sqrt(P))
    scaling_Σ_I = 1.0

    t1 = time()

    for i in (block1 + 1):n_MCMC
        if i == burnin_num
            t1 = time()
            total_num_likevals = 0
        end

        # Block RW update
        randn!(z)
        lmul!(Σ_rw, z)
        @views z .+= x[i,:]
        prop_logposterior = log_likelihood(z) + log_prior(z)
        total_num_likevals += 1
        @views accept_prob = (prop_logposterior) - (current_logposterior)
        if isfinite(accept_prob)
            if log(rand()) < accept_prob
                @views x[i,:] .= z
                acceptance[1] += 1
                current_logposterior = prop_logposterior
            end
        end

        # Adapt the covariance structure of proposal
        w_i = 1 / (i - half_warm_up_block)
        @views x_mean = (1 - w_i) * x_mean + w_i * x[i,:]
        Σ_chol_adapt.U .= sqrt((1 - w_i)) .*  Σ_chol_adapt.U
        @views ph_cholesky_update .= sqrt(w_i) .* (x[i,:] .- x_mean)
        lowrankupdate!(Σ_chol_adapt, ph_cholesky_update)
        Σ_rw .= (Σ_chol_adapt.L) .* scaling_Σ_I .* (2.38 / sqrt(P))

        if (i % tuning_step) == 0
            ## Diminishing Adaptation to 25% acceptance rate
            scaling_Σ_I =  exp(log(scaling_Σ_I) + (((acceptance[1] / tuning_step) - 0.25) / cbrt(i - block1)))
            Σ_rw .= (Σ_chol_adapt.L) .* scaling_Σ_I .* (2.38 / sqrt(P))

            ## Update User
            println("MCMC iter: ", i)
            println("Acceptance: ", round(acceptance[1] / tuning_step, digits=3))
            @views log_lik = @sprintf("%.2f", current_logposterior)
            println("Log Posterior: ", log_lik)
            acceptance[1] = 0
        end

        if i < n_MCMC
            @views x[i+1,:] .= x[i,:]
        end
    end

    return time() - t1, total_num_likevals
end

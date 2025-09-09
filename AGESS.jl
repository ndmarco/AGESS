using LinearAlgebra, Random, Distributions, Printf


function ESS_SingleStep(x::AbstractMatrix{Y}, z::AbstractVector{Y}, log_likelihood::Function, 
                        μ::AbstractVector{Y}, Σ_chol::LowerTriangular{Y, <:AbstractMatrix{Y}}, i::T) where {Y<:AbstractFloat, T<:Integer}
    P = size(x)[2]

    ## Propose new z from N(0, Σ)
    z .= Σ_chol * randn(P)
    @views y = log_likelihood(x[i,:]) + log(rand(1)[1])

    ## Propose Initial Angle
    θ = rand() * 2 * π
    θ_min = θ - 2 * π
    θ_max = θ

    ## Propose initial first move
    @views x[i,:] .= ((x[i-1,:] - μ) .* cos(θ) .+  z .* sin(θ)) .+ μ
    @views log_lik_prop = log_likelihood(x[i,:])

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
        @views x[i,:] .= ((x[i-1,:] - μ) .* cos(θ) .+  z .* sin(θ)) .+ μ
        @views log_lik_prop = log_likelihood(x[i,:])

        ## Check to make sure that posterior pdfs are computable
        if isnan(log_lik_prop)
            log_lik_prop = y - 1.0
        end
        if !isfinite(log_lik_prop)
            log_lik_prop = y - 1.0
        end
    end
    
    return nothing
end


function ESS(x::AbstractMatrix{Y}, log_likelihood::Function, μ::AbstractVector{Y}, Σ::AbstractMatrix{Y}; burnin::Y = 0.5) where {Y<:AbstractFloat}
    P = size(x)[2]
    n_MCMC = size(x)[1]
    z = zeros(P)
    burnin_num = floor(Int64, burnin * n_MCMC)

    Σ_chol = cholesky(Σ)
    t1 = time()

    for i in 2:n_MCMC
        if i == burnin_num
            t1 = time()
        end

        ESS_SingleStep(x, z, log_likelihood, μ, Σ_chol.L, i)

        ## Populate next value in Markov Chain
        if i < n_MCMC
            @views x[i+1,:] .= x[i,:]
        end

        ## Update User
        if (i % 25) == 0
            println("MCMC iter: ", i)
            log_lik = @sprintf("%.2f", log_likelihood(x[i,:]))
            println("Log Likelihood: ", log_lik)
        end
    end

    return time() - t1
end

function GESS_SingleStep(x::AbstractMatrix{Y}, z::AbstractVector{Y}, log_posterior::Function, ph::AbstractVector{Y}, 
                         μ::AbstractVector{Y}, Σ_chol::LowerTriangular{Y, <:AbstractMatrix{Y}}, i::T, ν::Y) where {Y<:AbstractFloat, T<:Integer}
    P = size(x)[2]

    α = 0.5 * (ν + P)
    @views ph .= (x[i,:] .- μ)
    ph .= ((Σ_chol) \ ph)
    β = 0.5 * (ν + dot(ph,ph))
    s = rand(InverseGamma(α, β))


    ESS_SingleStep(x, z, b -> (log_posterior(b) - dMvT(b, μ, Σ_chol, ph, ν, P)), μ, sqrt(s) * Σ_chol, i)

    return nothing
end

function GESS(x::AbstractMatrix{Y}, log_posterior::Function, μ::AbstractVector{Y}, Σ::AbstractMatrix{Y};
              ν::Y = 6.0, burnin::Y = 0.5) where {Y<:AbstractFloat}
    n_MCMC = size(x)[1]
    P = size(x)[2]
    z = zeros(P)
    Σ_chol = cholesky(Σ)
    ph = zeros(P)
    burnin_num = floor(Int64, burnin * n_MCMC)

    t1 = time()
    for i in 2:n_MCMC
        if i == burnin_num
            t1 = time()
        end

        GESS_SingleStep(x, z, log_posterior, ph, μ, Σ_chol.L, i, ν)

        ## Populate next value in Markov Chain
        if i < n_MCMC
            @views x[i+1,:] .= x[i,:]
        end

        ## Update User
        if (i % 25) == 0
            println("MCMC iter: ", i)
            @views log_pos = @sprintf("%.2f", log_posterior(x[i,:]))
            println("Log Posterior: ", log_pos)
        end
    end

    return time() - t1
end



function dMvT(x::AbstractVector{Y}, μ::AbstractVector{Y}, Σ_chol::LowerTriangular{Y, <:AbstractMatrix{Y}}, 
              ph::AbstractVector{Y}, ν::Y, P::T) where {Y<:AbstractFloat, T<:Integer}
    ph .= (x .- μ)
    ph .= ((Σ_chol) \ ph)
    pdf::Float64 = -(ν + P) * 0.5 * log1p((dot(ph, ph) / ν))
    return pdf
end

function dMvN(x::AbstractVector{Y}, μ::AbstractVector{Y}, Σ_chol::LowerTriangular{Y, <:AbstractMatrix{Y}}, 
              ph::AbstractVector{Y}) where {Y<:AbstractFloat}
    ph .= (x .- μ)
    ph .= ((Σ_chol) \ ph)
    pdf::Float64 = dot(ph, ph)
    return pdf
end

function cond_rMvT!(z::AbstractVector{Y}, x::AbstractVector{Y}, μ::AbstractVector{Y}, 
                    Σ_chol::LowerTriangular{Y, <:AbstractMatrix{Y}}, ν::Y, ph::AbstractVector{Y}, P::T) where {Y<:AbstractFloat, T<:Integer}
    z .= Σ_chol * randn(P) 
    ph .= (x .- μ)
    ph .= ((Σ_chol) \ ph)
    d = dot(ph, ph)
    ν1 = ν + P
    z .*= sqrt((ν + d) / ν1)
    z .*=  1 / sqrt(rand(Gamma(ν1/2, 2/ν1)))
    z .+= μ

    return nothing
end

function cond_dMvT!(z::AbstractVector{Y}, x::AbstractVector{Y}, μ::AbstractVector{Y}, 
                    Σ_chol::LowerTriangular{Y, <:AbstractMatrix{Y}}, ν::Y, ph::AbstractVector{Y}, 
                    Σ_chol_ph::LowerTriangular{Y, <:AbstractMatrix{Y}}, P::T) where {Y<:AbstractFloat, T<:Integer}
    ph .= (x .- μ)
    ph .= ((Σ_chol) \ ph)
    d = dot(ph, ph)
    ν1 = ν + P
    Σ_chol_ph .= sqrt((ν + d) / ν1) * Σ_chol
    pdf::Float64 = dMvT(z, μ, Σ_chol_ph, ph, ν1, P)

    return pdf
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
    ν1 = ν + 1
    z *= sqrt((ν + d) / ν1)
    z *= 1 / sqrt(rand(Gamma(ν1/2, 2/ ν1)))
    z += μ

    return z
end

function cond_dMvT_1d!(z::Y, x::Y, μ::Y, σ::Y, ν::Y) where {Y<:AbstractFloat}
    d = ((x - μ) / σ)^2
    ν1 = ν + 1
    σ_ph = sqrt((ν + d) / ν1) * σ
    pdf::Float64 = dMvT_1d(z, μ, σ_ph, ν1)

    return pdf
end

function AGESS_SingleStep(x::AbstractMatrix{Y}, z::AbstractVector{Y}, log_posterior::Function, 
                          ph::AbstractVector{Y}, t_dist::Bool, ν::Y, μ_adapt::AbstractVector{Y},
                          Σ_chol_adapt::LowerTriangular{Y, <:AbstractMatrix{Y}}, 
                          Σ_chol_ph::LowerTriangular{Y, <:AbstractMatrix{Y}}, i::T) where {Y<:AbstractFloat, T<:Integer}
    P = size(x)[2]
    y::Float64 = 0.0
    L_star::Float64 = 0.0
    ## Propose new z from N(μ, Σ)
    if t_dist == true
        @views cond_rMvT!(z, x[i,:], μ_adapt, Σ_chol_adapt, ν, ph, P)
    else
        z .= Σ_chol_adapt * randn(P) .+ μ_adapt
    end

    @views y = log_posterior(x[i,:])::Float64 + log(rand())
    if t_dist == true
        @views y -= cond_dMvT!(x[i,:], z, μ_adapt, Σ_chol_adapt, ph, ν, Σ_chol_ph, P)::Float64
    else
        @views y -= dMvN(x[i,:], μ_adapt, Σ_chol_adapt, ph)::Float64
    end

    ## Propose Initial Angle
    θ = rand() * 2 * π
    θ_min = θ - 2 * π
    θ_max = θ

    ## Propose initial first move
    @views x[i,:] .= ((x[i-1,:] - μ_adapt) .* cos(θ) .+  (z - μ_adapt) .* sin(θ)) .+ μ_adapt
    @views L_star = log_posterior(x[i,:])::Float64 
    if t_dist == true
        ### Add conditioning
        @views L_star -= dcond_dMvT!(x[i,:], z, μ_adapt, Σ_chol_adapt, ph, ν, Σ_chol_ph, P)::Float64
    else
        @views L_star -= dMvN(x[i,:], μ_adapt, Σ_chol_adapt, ph)::Float64
    end

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
        @views x[i,:] .= ((x[i-1,:] - μ_adapt) .* cos(θ) .+  (z - μ_adapt) .* sin(θ)) .+ μ_adapt
        @views L_star = log_posterior(x[i,:])::Float64 
        if t_dist == true
            @views L_star -= cond_dMvT!(x[i,:], z, μ_adapt, Σ_chol_adapt, ph, ν, Σ_chol_ph, P)::Float64
        else
            @views L_star -= dMvN(x[i,:], μ_adapt, Σ_chol_adapt, ph)::Float64
        end

        ## Check to make sure that posterior pdfs are computable
        if isnan(L_star)
            L_star = y - 1.0
        end
        if !isfinite(L_star)
            L_star = y - 1.0
        end
    end

    return nothing
end

function AGESS(x::AbstractMatrix{Y}, log_posterior::Function, 
               μ::AbstractVector{Y}, Σ::AbstractMatrix{Y},
               t_dist::Bool; ν::Y = 6.0, burnin::Y = 0.5, ϵ::Y = 0.1, single_step_prop::Y = 0.05) where {Y<:AbstractFloat}
    P = size(x)[2]
    n_MCMC = size(x)[1]
    z = zeros(P)
    burnin_num = floor(Int64, burnin * n_MCMC)
    t1 = time()

    μ_adapt = copy(μ)
    ph = similar(μ_adapt)

    Σ_chol = cholesky(Σ)
    Σ_chol_adapt = deepcopy(Σ_chol)

    Σ_ph =  LowerTriangular(diagm(ones(P)))
    Σ_ph1 =  LowerTriangular(diagm(ones(P)))
    μ_0 = zeros(P)
    ph_cholesky_update = ones(P)

    for i in 2:n_MCMC
        if i == burnin_num
            t1 = time()
        end

        if P >= 10
            if rand() > ϵ
                AGESS_SingleStep(x, z, log_posterior, ph, t_dist, ν, μ_adapt,
                                    Σ_chol_adapt.L, Σ_ph1, i)
            elseif rand() > 0.5
                AGESS_SingleStep_1d(x, log_posterior, t_dist, ν, μ_adapt, Σ_chol_adapt.L, i)
            else
                AGESS_SingleStep(x, z, log_posterior, ph, t_dist, ν, μ_0, Σ_ph, Σ_ph1, i)
            end
        else
            if rand() > ϵ
                AGESS_SingleStep(x, z, log_posterior, ph, t_dist, ν, μ_adapt,
                                        Σ_chol_adapt.L, Σ_ph1, i)
            else
                AGESS_SingleStep(x, z, log_posterior, ph, t_dist, ν, μ_0, Σ_ph, Σ_ph1, i)
            end

        end
        
        
        ## Adapt mean and covariance
        w_i = i^(-1)
        Σ_chol_adapt.U .*= sqrt(1 - w_i)
        @views ph_cholesky_update .= sqrt(w_i) * (x[i,:] - μ_adapt)
        lowrankupdate!(Σ_chol_adapt, ph_cholesky_update)
        @views μ_adapt .= (1 - w_i) * μ_adapt +  w_i * x[i,:]

        ## Populate next value in Markov Chain
        if i < n_MCMC
            @views x[i+1,:] .= x[i,:]
        end

        # Update User
        if P >= 10
            if i < burnin_num * single_step_prop
                if (i % 25) == 0
                    println("MCMC iter: ", i)
                    @views log_lik = @sprintf("%.2f", log_posterior(x[i,:]))
                    println("Log Posterior: ", log_lik)
                end
            else
                if (i % 1000) == 0
                    println("MCMC iter: ", i)
                    @views log_lik = @sprintf("%.2f", log_posterior(x[i,:]))
                    println("Log Posterior: ", log_lik)
                end
            end
        else
            if (i % 1000) == 0
                println("MCMC iter: ", i)
                @views log_lik = @sprintf("%.2f", log_posterior(x[i,:]))
                println("Log Posterior: ", log_lik)
            end
        end
        
    end

    return time() - t1, Σ_chol_adapt.L * Σ_chol_adapt.U
end

function AGESS_SingleStep_1d(x::AbstractMatrix{Y}, log_posterior::Function, 
                             t_dist::Bool, ν::Y, μ_adapt::AbstractVector{Y}, Σ_chol_adapt::LowerTriangular{Y, <:AbstractMatrix{Y}}, i::T) where {Y<:AbstractFloat, T<:Integer}

    P = size(x)[2]
    z::Float64 = 0.0
    y::Float64 = 0.0
    L_star::Float64 = 0.0
    for j in randperm(P)
        
        ## Propose new z from N(0, Σ)
        if t_dist == true
            z = cond_rMvT_1d!(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j], ν)
        else
            z = Σ_chol_adapt[j,j] * randn() + μ_adapt[j]
        end

        @views y = log_posterior(x[i,:]) + log(rand())
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
        if t_dist == true
            L_star -= dMvT_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j], ν)
        else
            L_star -= dMvN_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j])
        end

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
            if t_dist == true
                L_star -= dMvT_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j], ν)
            else
                L_star -= dMvN_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j])
            end

            ## Check to make sure that posterior pdfs are computable
            if isnan(L_star)
                L_star = y - 1.0
            end   
            if !isfinite(L_star)
                L_star = y - 1.0
            end
        end
    end

    return nothing
end

function AGESS_SingleStep_1d_single(x::AbstractMatrix{Y}, log_posterior::Function, 
                                    t_dist::Bool, ν::Y, μ_adapt::AbstractVector{Y}, 
                                    Σ_chol_adapt::LowerTriangular{Y, <:AbstractMatrix{Y}}, i::T, j::T) where {Y<:AbstractFloat, T<:Integer}

    z = 0.0
    ## Propose new z from N(0, Σ)
    if t_dist == true
        z = cond_rMvT_1d!(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j], ν)
    else
        z = Σ_chol_adapt[j,j] * randn() + μ_adapt[j]
    end

    @views y = log_posterior(x[i,:]) + log(rand(1)[1])
    if t_dist == true
        y -= cond_dMvT_1d!(x[i,j], z, μ_adapt[j], Σ_chol_adapt[j,j], ν)
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
    if t_dist == true
        L_star -= cond_dMvT_1d!(x[i,j], z, μ_adapt[j], Σ_chol_adapt[j,j], ν)
    else
        L_star -= dMvN_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j])
    end

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
        if t_dist == true
            L_star -= cond_dMvT_1d!(x[i,j], z, μ_adapt[j], Σ_chol_adapt[j,j], ν)
        else
            L_star -= dMvN_1d(x[i,j], μ_adapt[j], Σ_chol_adapt[j,j])
        end

        ## Check to make sure that posterior pdfs are computable
        if isnan(L_star)
            L_star = y - 1.0
        end   
        if !isfinite(L_star)
            L_star = y - 1.0
        end
    end

    return nothing
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

    for i in 2:block1
        # Perform random walk proposal
        for j in 1:P
            @views z .= x[i,:]
            z[j] = x[i,j] + randn() * σ_rw[j]
            @views accept_prob = (log_likelihood(z) + log_prior(z)) - (log_likelihood(x[i,:]) + log_prior(x[i,:]))
            if isfinite(accept_prob)
                if log(rand()) < accept_prob
                    x[i,j] = z[j]
                    acceptance[j] += 1
                end
            end
        end

        ## Tune Acceptance
        if (i % tuning_step) == 0
            println("MCMC iter: ", i)
            println("Acceptance: ", round(mean(acceptance) / tuning_step, digits=3))
            @views log_lik = @sprintf("%.2f", log_likelihood(x[i,:]))
            println("Log Likelihood: ", log_lik)
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
        end

        # Block RW update
        @views z = x[i,:] .+ Σ_rw * randn(P)
        @views accept_prob = (log_likelihood(z) + log_prior(z)) - (log_likelihood(x[i,:]) + log_prior(x[i,:]))
        if isfinite(accept_prob)
            if log(rand()) < accept_prob
                @views x[i,:] .= z
                acceptance[1] += 1
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
            @views log_lik = @sprintf("%.2f", log_likelihood(x[i,:]))
            println("Log Likelihood: ", log_lik)
            acceptance[1] = 0
        end

        if i < n_MCMC
            @views x[i+1,:] .= x[i,:]
        end
    end

    return time() - t1
end

function ARW_SingleStep_1d(x::AbstractVector{Y}, log_likelihood::Function, log_prior::Function,
                           σ_rw::Y, i::T, acceptance::T) where {Y<:AbstractFloat, T<:Integer}
    z = x[i] + randn() * σ_rw
    accept_prob = (log_likelihood(z) + log_prior(z)) - (log_likelihood(x[i]) + log_prior(x[i]))
    if isfinite(accept_prob)
        if log(rand()) < accept_prob
            x[i] = z
            acceptance += 1
        end
    end

    return nothing
end
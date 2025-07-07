using LinearAlgebra, Random, Distributions, Printf

function ESS(x::AbstractMatrix{Y}, log_likelihood::Function, Σ::AbstractMatrix{Y}; burnin::Y = 0.5) where {Y<:AbstractFloat}
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
        ## Propose new z from N(0, Σ)
        z .= Σ_chol.L * randn(P)
        @views y = log_likelihood(x[i,:]) + log(rand(1)[1])

        ## Propose Initial Angle
        θ = rand(1)[1] * 2 * π
        θ_min = θ - 2 * π
        θ_max = θ

        ## Propose initial first move
        x[i,:] .= x[i-1,:] .* cos(θ) .+  z .* sin(θ)
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
            θ = θ_min + rand(1)[1] * (θ_max - θ_min)
            x[i,:] .= x[i-1,:] .* cos(θ) .+  z .* sin(θ)
            @views log_lik_prop = log_likelihood(x[i,:])

            ## Check to make sure that posterior pdfs are computable
            if isnan(log_lik_prop)
                log_lik_prop = y - 1.0
            end
            if !isfinite(log_lik_prop)
                log_lik_prop = y - 1.0
            end
        end
        ## Populate next value in Markov Chain
        if i < n_MCMC
            x[i+1,:] .= x[i,:]
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

function ESS_SingleStep(x::AbstractMatrix{Y}, z::AbstractVector{Y}, log_likelihood::Function, 
                        Σ_chol::LowerTriangular{Y, Matrix{Y}}, i::T) where {Y<:AbstractFloat, T<:Integer}
    P = size(x)[2]

    ## Propose new z from N(0, Σ)
    z .= Σ_chol * randn(P)
    @views y = log_likelihood(x[i,:]) + log(rand(1)[1])

    ## Propose Initial Angle
    θ = rand(1)[1] * 2 * π
    θ_min = θ - 2 * π
    θ_max = θ

    ## Propose initial first move
    x[i,:] .= x[i-1,:] .* cos(θ) .+  z .* sin(θ)
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
        θ = θ_min + rand(1)[1] * (θ_max - θ_min)
        x[i,:] .= x[i-1,:] .* cos(θ) .+  z .* sin(θ)
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



## Need to do based on conditional distribution??
function dMvT(x::AbstractVector{Y}, μ::AbstractVector{Y}, Σ_chol::LowerTriangular{Y, Matrix{Y}}, 
              ph::AbstractVector{Y}, ν::Y, P::T) where {Y<:AbstractFloat, T<:Integer}
    ph .= ((Σ_chol) \ (x .- μ))
    pdf = -(ν + P) * 0.5 * log1p((dot(ph, ph) / ν))
    return pdf
end

function dMvN(x::AbstractVector{Y}, μ::AbstractVector{Y}, Σ_chol::LowerTriangular{Y, Matrix{Y}}, 
              ph::AbstractVector{Y}) where {Y<:AbstractFloat}
    ph .= ((Σ_chol) \ (x .- μ))
    pdf = dot(ph, ph)
    return pdf
end

function cond_rMvT!(z::AbstractVector{Y}, x::AbstractVector{Y}, μ::AbstractVector{Y}, 
                    Σ_chol::LowerTriangular{Y, Matrix{Y}}, ν::Y, ph::AbstractVector{Y}, P::T) where {Y<:AbstractFloat, T<:Integer}
    z .= Σ_chol * randn(P) 
    ph .= ((Σ_chol) \ (x .- μ))
    d = dot(ph, ph)
    ν̃ = ν + P
    z .*= sqrt((ν + d) / ν̃)
    z .*= sqrt(rand(Gamma(ν̃/2, 2/ ν̃)))
    z .+= μ

    return nothing
end
    


function AGESS(x::AbstractMatrix{Y}, log_likelihood::Function, log_prior::Function, 
               μ::AbstractVector{Y}, Σ::AbstractMatrix{Y}, t_dist::Bool, ν::Y = -1.0;
               burnin::Y = 0.5) where {Y<:AbstractFloat}
    P = size(x)[2]
    n_MCMC = size(x)[1]
    z = zeros(P)
    burnin_num = floor(Int64, burnin * n_MCMC)
    t1 = time()

    if ν <= 0.0
        ν = float(P)
    end

    μ_adapt = copy(μ)
    ph = similar(μ_adapt)

    Σ_chol = cholesky(Σ)
    Σ_chol_adapt = deepcopy(Σ_chol)


    for i in 2:n_MCMC
        if i == burnin_num
            t1 = time()
        end
        ## Propose new z from N(0, Σ)
        if t_dist == true
            cond_rMvT!(z, x[i,:], μ_adapt, Σ_chol_adapt.L, ν, ph, P)
        else
            z .= Σ_chol_adapt.L * randn(P) .+ μ_adapt
        end
        
        @views y = log_likelihood(x[i,:]) + log_prior(x[i,:], μ, Σ_chol.L) + log(rand(1)[1])
        if t_dist == true
            @views y -= dMvT(x[i,:], μ_adapt, Σ_chol_adapt.L, ph, ν, P)
        else
            @views y -= dMvN(x[i,:], μ_adapt, Σ_chol_adapt.L, ph)
        end

        ## Propose Initial Angle
        θ = rand(1)[1] * 2 * π
        θ_min = θ - 2 * π
        θ_max = θ

        ## Propose initial first move
        x[i,:] .= x[i-1,:] .* cos(θ) .+  z .* sin(θ)
        @views L_star = log_likelihood(x[i,:]) + log_prior(x[i,:], μ, Σ_chol.L) 
        if t_dist == true
            @views L_star -= dMvT(x[i,:], μ_adapt, Σ_chol_adapt.L, ph, ν, P)
        else
            @views L_star -= dMvN(x[i,:], μ_adapt, Σ_chol_adapt.L, ph)
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
            θ = θ_min + rand(1)[1] * (θ_max - θ_min)
            x[i,:] .= x[i-1,:] .* cos(θ) .+  z .* sin(θ)
            @views L_star = log_likelihood(x[i,:]) + log_prior(x[i,:], μ, Σ_chol.L) 
            if t_dist == true
                @views L_star -= dMvT(x[i,:], μ_adapt, Σ_chol_adapt.L, ph, ν, P)
            else
                @views L_star -= dMvN(x[i,:], μ_adapt, Σ_chol_adapt.L, ph)
            end

            ## Check to make sure that posterior pdfs are computable
            if isnan(L_star)
                L_star = y - 1.0
            end
            if !isfinite(L_star)
                L_star = y - 1.0
            end
        end

        ## Adapt mean and covariance
        w_i = min(1.0/(10 * P), 1.0/i)
        μ_adapt .= (1 - w_i) * μ_adapt +  w_i * x[i,:]
        Σ_chol_adapt.U .= sqrt((1 - w_i)) .*  Σ_chol_adapt.U
        lowrankupdate!(Σ_chol_adapt, sqrt(w_i) .* (x[i,:] .- μ_adapt))

        ## Populate next value in Markov Chain
        if i < n_MCMC
            x[i+1,:] .= x[i,:]
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

function AGESS_SingleStep(x::AbstractMatrix{Y}, log_likelihood::Function, log_prior::Function, 
                          ph::AbstractVector{Y}, t_dist::Bool, ν::Y, μ_adapt::AbstractVector{Y},
                          Σ_chol_adapt::LowerTriangular{Y, Matrix{Y}}, i::T) where {Y<:AbstractFloat, T<:Integer}
    P = size(x)[2]

    z = zeros(P)
    ## Propose new z from N(0, Σ)
    if t_dist == true
        cond_rMvT!(z, x[i,:], μ_adapt, Σ_chol_adapt, ν, ph, P)
    else
        z .= Σ_chol_adapt * randn(P) .+ μ_adapt
    end

    @views y = log_likelihood(x[i,:]) + log_prior(x[i,:]) + log(rand(1)[1])
    if t_dist == true
        @views y -= dMvT(x[i,:], μ_adapt, Σ_chol_adapt, ph, ν, P)
    else
        @views y -= dMvN(x[i,:], μ_adapt, Σ_chol_adapt, ph)
    end

    ## Propose Initial Angle
    θ = rand(1)[1] * 2 * π
    θ_min = θ - 2 * π
    θ_max = θ

    ## Propose initial first move
    x[i,:] .= x[i-1,:] .* cos(θ) .+  z .* sin(θ)
    @views L_star = log_likelihood(x[i,:]) + log_prior(x[i,:]) 
    if t_dist == true
        @views L_star -= dMvT(x[i,:], μ_adapt, Σ_chol_adapt, ph, ν, P)
    else
        @views L_star -= dMvN(x[i,:], μ_adapt, Σ_chol_adapt, ph)
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
        θ = θ_min + rand(1)[1] * (θ_max - θ_min)
        x[i,:] .= x[i-1,:] .* cos(θ) .+  z .* sin(θ)
        @views L_star = log_likelihood(x[i,:]) + log_prior(x[i,:]) 
        if t_dist == true
            @views L_star -= dMvT(x[i,:], μ_adapt, Σ_chol_adapt, ph, ν, P)
        else
            @views L_star -= dMvN(x[i,:], μ_adapt, Σ_chol_adapt, ph)
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

    for i in 2:block1
        # Perform random walk proposal
        for j in 1:P
            z .= x[i,:]
            z[j] = x[i,j] + randn() * σ_rw[j]
            @views accept_prob = (log_likelihood(z) + log_prior(z, μ, Σ_chol.L)) - (log_likelihood(x[i,:]) + log_prior(x[i,:], μ, Σ_chol.L))
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
            log_lik = @sprintf("%.2f", log_likelihood(x[i,:]))
            println("Log Likelihood: ", log_lik)
            for j in 1:P
                σ_rw[j] = exp(log(σ_rw[j]) + ((acceptance[j] / tuning_step) - 0.44))
                acceptance[j] = 0
            end
        end

        ## Update Markov Chain
        x[i+1,:] .= x[i,:]
    end

    half_warm_up_block = floor(Int64, block1/2)
    Σ_chol_adapt = cholesky(cov(x[half_warm_up_block:block1,:]))
    x_mean = mean(x[half_warm_up_block:block1,:], dims=1)[1,:]
    Σ_rw = Σ_chol_adapt.L .* (2.38 / sqrt(P))
    scaling_Σ_I = 1.0

    t1 = time()

    for i in (block1 + 1):n_MCMC
        if i == burnin_num
            t1 = time()
        end

        # Block RW update
        z = x[i,:] .+ Σ_rw * randn(P)
        @views accept_prob = (log_likelihood(z) + log_prior(z, μ, Σ_chol.L)) - (log_likelihood(x[i,:]) + log_prior(x[i,:], μ, Σ_chol.L))
        if isfinite(accept_prob)
            if log(rand()) < accept_prob
                x[i,:] .= z
                acceptance[1] += 1
            end
        end

        # Adapt the covariance structure of proposal
        w_i = 1 / (i - half_warm_up_block)
        x_mean = (1 - w_i) * x_mean + w_i * x[i,:]
        Σ_chol_adapt.U .= sqrt((1 - w_i)) .*  Σ_chol_adapt.U
        @views lowrankupdate!(Σ_chol_adapt, sqrt(w_i) .* (x[i,:] .-  x_mean))
        Σ_rw .= (Σ_chol_adapt.L) .* scaling_Σ_I .* (2.38 / sqrt(P))

        if (i % tuning_step) == 0
            ## Diminishing Adaptation to 25% acceptance rate
            scaling_Σ_I =  exp(log(scaling_Σ_I) + (((acceptance[1] / tuning_step) - 0.25) / cbrt(i - block1)))
            Σ_rw .= (Σ_chol_adapt.L) .* scaling_Σ_I .* (2.38 / sqrt(P))

            ## Update User
            println("MCMC iter: ", i)
            println("Acceptance: ", round(acceptance[1] / tuning_step, digits=3))
            log_lik = @sprintf("%.2f", log_likelihood(x[i,:]))
            println("Log Likelihood: ", log_lik)
            acceptance[1] = 0
        end

        if i < n_MCMC
            x[i+1,:] .= x[i,:]
        end
    end

    return time() - t1
end

function ARW_SingleStep_1d(x::AbstractVector{Y}, log_likelihood::Function, log_prior::Function,
                           σ_rw::Y, i::T, acceptance::T) where {Y<:AbstractFloat, T<:Integer}
    z = x[i] + randn() * σ_rw
    accept_prob = (log_likelihood(z) + log_prior(x)) - (log_likelihood(x[i]) + log_prior(x[i]))
    if isfinite(accept_prob)
        if log(rand()) < accept_prob
            x[i] = z
            acceptance += 1
        end
    end

    return nothing
end
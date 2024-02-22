# Author: Yen-Huan Li
# Email: yenhuan.li@csie.ntu.edu.tw
# computing the value of a matrix game by Nesterov's excessive gap technique

using LinearAlgebra
using Random 
using Distributions
using BenchmarkTools

function residual(x::Vector{Float64}, 
    y::Vector{Float64}, 
    A::Array{Float64, 2})
    return maximum(A * x) - minimum(A' * y)
end

function entropy_projection(v::Vector{Float64})
    return v ./ sum(v)
end

function robust_exp(v::Vector{Float64})
    m = maximum(abs.(v))
    if m < 100.0
        return exp.(v)
    else
        c = m / 100.0
        y = (exp.(v ./ c)) .^ c
        return y
    end
end

function prox(ξ::Vector{Float64}, 
    x::Vector{Float64})

    return entropy_projection(x .* robust_exp(-ξ))
end

# Yu. Nesterov. Excessive gap technique in nonsmooth convex minimization. SIAM J. Optim. 2005. 
# C. Kroer et al. Faster algorithms for extensive-form game solving via improved smoothing functions. Math. Program. 2020. 
# function EGT(#μ_1::Float64, 
#     μ_2::Float64, 
#     ε::Float64, 
#     A::Array{Float64, 2})

#     function gap(x::Vector{Float64}, y::Vector{Float64})
#         return residual(x, y, A)
#     end

#     function x_μ(y::Vector{Float64}, 
#         μ::Float64)

#         return entropy_projection(robust_exp(- (A' * y) ./ μ)) 
#     end
    
#     function y_μ(x::Vector{Float64}, 
#         μ::Float64)
    
#         return entropy_projection(robust_exp((A * x) ./ μ))
#     end
    
#     function grad_φ_x(x::Vector{Float64}, 
#         μ::Float64)
    
#         return A' * y_μ(x, μ)
#     end
    
#     function grad_φ_y(y::Vector{Float64}, 
#         μ::Float64)
    
#         return A * x_μ(y, μ)
#     end
    
#     function step(μ_1::Float64,
#         μ_2::Float64,
#         x::Vector{Float64}, 
#         y::Vector{Float64}, 
#         τ::Float64,
#         even_round::Bool)
        
#         if even_round
#             xμy      = x_μ(y, μ_1)
#             x_hat    = (1.0 - τ) * x + τ * xμy
#             y_plus   = (1.0 - τ) * y + τ * y_μ(x_hat, μ_2)
#             x_tilde  = prox((τ / (1.0 - τ) / μ_1) .* grad_φ_x(x_hat, μ_2), xμy)
#             x_plus   = (1.0 - τ) * x + τ * x_tilde
#             μ_1_plus = (1.0 - τ) * μ_1 
#         else
#             yμx      = y_μ(x, μ_1)
#             y_hat    = (1.0 - τ) * y + τ * yμx
#             x_plus   = (1.0 - τ) * x + τ * x_μ(y_hat, μ_2)
#             y_tilde  = prox((τ / (1.0 - τ) / μ_1) .* grad_φ_y(y_hat, μ_2), yμx)
#             y_plus   = (1.0 - τ) * y + τ * y_tilde
#             μ_1_plus = (1.0 - τ) * μ_1 
#         end
    
#         return μ_1_plus, x_plus, y_plus
    
#     end

#     dim_x = size(A, 2)
#     x_ω   = ones(dim_x) / dim_x

#     t      = 0
#     A_norm = maximum(abs.(A))
#     μ_2_t  = μ_2
#     μ_1_t  = (A_norm ^ 2.0) / μ_2 
#     x_t    = prox((1.0 / μ_1_t) * grad_φ_x(x_ω, μ_2), x_ω)
#     y_t    = y_μ(x_ω, μ_2)

#     while gap(x_t, y_t) > ε
#         τ = 2.0 / (t + 3.0)
#         μ_1_t, x_t, y_t = step(μ_1_t, μ_2_t, x_t, y_t, τ, Bool(t % 2))
#         println(t, ": ", gap(x_t, y_t))
#         t = t + 1
#     end

#     return x_t, y_t

# end

# function testEGT(μ_2::Float64, A::Array{Float64, 2})
#     ε = .01

#     return EGT(μ_2, ε, A)

# end

function ensure_probability(p::Vector{Float64})
    p = max.(p, 0.0)
    p = p / sum(p)
end

# Y. Chen et al. Optimal primal-dual methods for a class of saddle point problems. SIAM J. Optim. 2014. 
# The stochastic version is very slow empirically. 
function APD(ε::Float64, 
    A::Array{Float64, 2}, 
    type::String)

    gap(x, y) = residual(x, y, A)
    Ω(dim, ν) = sqrt((1.0 + ν / dim) * log(1.0 + dim / ν))

    function g_x(y::Vector{Float64})
        if type == "deterministic" || type == "d"
            return A' * y 
        elseif type == "stochastic" || type == "s"
            # Section 5.3
            y = ensure_probability(y)
            return vec(A[rand(Categorical(y)), :])
        end
    end

    function g_y(x::Vector{Float64})
        if type == "deterministic" || type == "d"
            return - A * x
        elseif type == "stochastic" || type == "s"
            # Section 5.3 
            x = ensure_probability(x)
            return - A[:, rand(Categorical(x))]
        end
    end

    # Section 5.3
    ν = 1e-16

    # Section 5.2 
    n_y, n_x = size(A)
    D_x      = sqrt(2) * Ω(n_x, ν)
    D_y      = sqrt(2) * Ω(n_y, ν)
    L_K      = maximum(abs.(A)) 
    α_x = α_y = 1.0 + ν

    x_bar = x = x_ag = ones(n_x) / Float64(n_x)
    y         = y_ag = ones(n_y) / Float64(n_y)

    # Corollary 2.2
    η = α_x * D_x / L_K / D_y 
    τ = α_y * D_y / L_K / D_x 
    # Section 5.3
    σ_x = σ_y = 2.0 * L_K 

    t = 1
    while gap(x_ag, y_ag) > ε
        # Corollary 2.2
        β_inv = 2.0 / (t + 1.0)
        θ     = (t - 1.0) / t

        #Corollary 3.2
        if type == "stochastic"
            η = 2.0 * α_x * D_x / (3.0 * L_K * D_y + 3.0 * σ_x * sqrt(t))
            τ = 2.0 * α_y * D_y / (3.0 * L_K * D_x + 3.0 * σ_y * sqrt(t))
        end

        # "Numerically speaking", one can just ignore \nu and do entropic mirror descent, as suggested in: 
        # A. Ben-Tal & A. Nemirovski. Non-euclidean restricted memory level method for large-scale convex optimization. Math. Program., Ser. A. 2005. 
        y      = prox(τ .* g_y(x_bar), y) 
        x_prev = x 
        x      = prox(η .* g_x(y), x)
        x_ag   = (1.0 - β_inv) .* x_ag + β_inv .* x
        y_ag   = (1.0 - β_inv) .* y_ag + β_inv .* y
        x_bar  = θ .* (x - x_prev) + x 

        # println(gap(x_ag, y_ag))

        t = t + 1
    end

    x_ag = ensure_probability(x_ag)
    y_ag = ensure_probability(y_ag)
    return x_ag, y_ag, gap(x_ag, y_ag)

end

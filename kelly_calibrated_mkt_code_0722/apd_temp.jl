# Author: Yen-Huan Li
# Email: yenhuan.li@csie.ntu.edu.tw
# computing the value of a matrix game by Nesterov's excessive gap technique

using LinearAlgebra
using Random 
using Distributions
using BenchmarkTools

function residual(x::Vector{Float64},
    y::Vector{Float64},
    A)
    return maximum(A' * x) - minimum(A * y)
end

#proj vector onto simplex, ensuring prob distri
function entropy_projection(v::Vector{Float64})
    return v ./ sum(v)
end

#compute element-wise exp func w robust handling for large vals
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

function ensure_probability(p::Vector{Float64})
    p = max.(p, 0.0)
    p = p / sum(p)
end

# Y. Chen et al. Optimal primal-dual methods for a class of saddle point problems. SIAM J. Optim. 2014. 
# The stochastic version is very slow empirically. 
function APD(ε::Float64, 
    A,
    type::String)

    # calculates primal dual gap (minmax = maxmin)
    # p. 6(1810) gap function 
    gap(x, y) = residual(x, y, A)

    #p. 1809
    #Ω_x^2, Ω_y^2
    Ω(dim, ν) = sqrt((1.0 + ν / dim) * log(1.0 + dim / ν))

    #p. 1810
    #grad of x, <x, Ay>: Ay / grad of x, <Kx, y>: K'y
    function g_x(y)
        if type == "deterministic" || type == "d"
        #    return A' * y
            return A * y
        elseif type == "stochastic" || type == "s"
            # Section 5.3
            y = ensure_probability(y) #within 0-1
            return vec(A[rand(Categorical(y)), :]) #sampling 1
        end
    end

    #grad of y, <x, Ay>: -A'x / grad of y, <Kx, y>: -Kx
    # "-": ensures grad aligned with the direction of maximizing the dual objective func
    function g_y(x)
        if type == "deterministic" || type == "d"
            return - A' * x
        elseif type == "stochastic" || type == "s"
            # Section 5.3 
            x = ensure_probability(x)
            return - A[:, rand(Categorical(x))]
        end
    end

    # Section 5.2, p. 1809
    ν = 1e-16

    # Section 5.2, p. 1809
    n_x, n_y = size(A) #x = 101, y = 2

    D_x      = sqrt(2) * Ω(n_x, ν) #why not sqrt(2/α_x), sqrt(2/α_y)?
    D_y      = sqrt(2) * Ω(n_y, ν)
    L_K      = maximum(abs.(A)) #K = matrix A
    α_x = α_y = 1.0 + ν #strog convexity parameters, 1 under Euclidean setting
    #D_x      = sqrt(2/α_x) * Ω(n_x, ν)
    #D_y      = sqrt(2/α_y) * Ω(n_y, ν)

    #p 1784 alg 1
    x_bar = x = x_ag = ones(n_x) / Float64(n_x) #uniform vector
    y         = y_ag = ones(n_y) / Float64(n_y)

    # Corollary 2.2, p. 1786
    # stepsizes η (for x), τ (for y)
    η = α_x * D_x / L_K / D_y #diff than the paper η_t = α_x * t / (2 * L_G + t * L_k * D_Y / D_x)
    τ = α_y * D_y / L_K / D_x 
    # Section 5.3, p. 1811 
    σ_x = σ_y = 2.0 * L_K 

    t = 1
    while gap(x_ag, y_ag) > ε #termination condition based on primal-dual gap
        # Corollary 2.2, p. 1786
        β_inv = 2.0 / (t + 1.0)
        θ     = (t - 1.0) / t

        #Corollary 3.2, p. 1790
        if type == "stochastic"
            η = 2.0 * α_x * D_x / (3.0 * L_K * D_y + 3.0 * σ_x * sqrt(t)) #why not 6 * L_G * D_x / t?
            τ = 2.0 * α_y * D_y / (3.0 * L_K * D_x + 3.0 * σ_y * sqrt(t))
        end

        # "Numerically speaking", one can just ignore \nu and do entropic mirror descent, as suggested in: 
        # A. Ben-Tal & A. Nemirovski. Non-euclidean restricted memory level method for large-scale convex optimization. Math. Program., Ser. A. 2005. 

        #entropic md?
        #p. 1784 alg 1, p. 1785 alg 2
        y      = prox(τ .* g_y(x_bar), y) #τ
        x_prev = x 
        x      = prox(η .* g_x(y), x) #η
        x_ag   = (1.0 - β_inv) .* x_ag + β_inv .* x #convex combination of previous x and aggregated x
        y_ag   = (1.0 - β_inv) .* y_ag + β_inv .* y #convex combination of y and y_ag
        x_bar  = θ .* (x - x_prev) + x 

        # println(gap(x_ag, y_ag))

        t = t + 1
    end

    #ag: aggregated
    x_ag = ensure_probability(x_ag)
    y_ag = ensure_probability(y_ag)
    return x_ag, y_ag, gap(x_ag, y_ag)

end
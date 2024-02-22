using LinearAlgebra
using Statistics

#objective function
#returns: vector/matrix of price relatives
function f_kelly(b::Vector{Float64}, returns, x_t::Vector{Float64}) #b: portfolio, x_t: estimated market distribution at t
    return sum(x_t[i] * (- log(b ⋅ returns[i])) for i in 1:length(returns))
end

function ∇f_kelly(b::Vector{Float64}, returns, x_t::Vector{Float64})
    subgrad = zeros(length(b))
    for i in 1:length(returns)
        subgrad .+= (- x_t[i] * returns[i] / (b ⋅ returns[i]))
    end
    return subgrad
end

function f_bcrp(b, x_sequence) #best constant rebalanced portfolio based on past(1~t-1) Market return vector
    return (sum(- log(b ⋅ x_sequence[t]) for t in 1:length(x_sequence)))
end

function ∇f_bcrp(b, x_sequence) #b: portfolio, x_sequence: Market vector from t=1 to T
    subgrad = zeros(length(b))
    for i in 1:length(x_sequence)
        subgrad .+= (- x_sequence[i] / (b ⋅ x_sequence[i]))
    end
    return subgrad
end

function f(x_sequence_t, b)
    return -log(x_sequence_t ⋅ b)
end

function sumf(x_sequence, b)
    return -sum(log.(x_sequence ⋅ b))
end

function ∇f(x_sequence_t, b)
    return - x_sequence_t / (x_sequence_t ⋅ b)
end

function Ef(x_sequence, b) #A = x_sequence?
    return -sum(log.(x_sequence ⋅ b)) / size(x_sequence, 2) #A?
end

function ∇Ef(x_sequence, b)
    grads = - x_sequence ./ (x_sequence ⋅ b)'
    return reshape(sum(grads, dims=2) / size(x_sequence, 2), (size(x_sequence, 1), 1))
end

# Asymptotic Optimality and Asymptotic Equipartition Properties of Log-optimum Investment
# By Paul H. Algoet and Thomas M. Cover 
function kelly(x_t::Vector{Float64}, returns, t, T, b)
    m = length(returns[1]) # # of targets

    r = 0.9
    τ = 0.1
    #α_bar = r * sqrt(log(m)/t) # r*sqrt(log(n)/t) n: number of targets r: returns uniformly lower bound
    α_bar = 0.1 #step size upper bound
    
    #optimization- eg_armijo, polyak step size
    #b = ones(m) / m
    b_update = eg_armijo(b -> f_kelly(b, returns, x_t), ∇f_kelly(b, returns, x_t), b, α_bar, r, τ)
    while abs(f_kelly(b_update, returns, x_t) - f_kelly(b, returns, x_t)) > 1e-5 #function value yield from kelly_portfolio
        b = b_update
        b_update = eg_armijo(b -> f_kelly(b, returns, x_t), ∇f_kelly(b, returns, x_t), b, α_bar, r, τ)
    end
    b = b_update
    
    #max_iter = 1000
    #η = 0.1
    #δ1 = 0.125
    #B = 1.0 
    #c = 0.6
    #b_thm2 = polyak(b -> f(b, returns, x_t), b -> ∇f(b, returns, x_t), T_polyak, constraint_set, b, δ1, B, c, max_iter)

    return b #investor portfolio
end

#T. Cover. Universal Portfolios, 1991.
#http://www-isl.stanford.edu/~cover/papers/paper93.pdf
function bcrp(x_sequence, t)
    n = length(x_sequence) #number of time periods
    m = length(x_sequence[1]) #number of targets

    b = ones(m) / m #uniform
    r = 0.9
    τ = 0.1
    α_bar = r * sqrt(log(m)/t)
    b_bcrp = eg_armijo(b -> f_bcrp(b, x_sequence), ∇f_bcrp(b, x_sequence), b, α_bar, r, τ)

    return b_bcrp
end

# benchmarks - online
function soft_bayes(x_sequence_t, T, b)
    m = length(x_sequence_t) #number of targets
    #b = ones(m) / m
    eta = (log(m) / (m * T))^0.5

    #print("gradf(x_sequence_t, b): ", ∇f(x_sequence_t, b),"\n")
    b = b .* (1.0 .- eta .- eta * ∇f(x_sequence_t, b))

    return b
end

function lb_ftrl(x_sequence, T, b)
    m = length(x_sequence_t) #number of targets
    #b = ones(m) / m
    eta = (m * log(T) / (2 * T))^0.5
    cumu_grad = zeros(m)
    #update
    #Newton's method
    tol = 1e-5
    for t in 1:T
        cumu_grad += ∇f(x_sequence[t], b)

        lamb = 1.0 + eta * maximum(-cumu_grad)
        s = lamb .+ eta .* cumu_grad
        dpsi = 1.0 - sum(1.0 ./ s)
        ddpsi = sum(1.0 ./ (s .* s))
        while dpsi^2 > tol * ddpsi
            lamb -= dpsi / ddpsi
            s = lamb .+ eta .* cumu_grad
            dpsi = 1.0 - sum(1.0 ./ s)
            ddpsi = sum(1.0 ./ (s .* s))
        end
        b = 1.0 ./ s
    end
    return b
end

#Chungen's algo 1
function ada_lb_ftrl(x_sequence_t, T, b)
    m = length(x_sequence_t) #number of targets
    #b = ones(m) / m
    cumu_grad = zeros(m)
    cumu_grad_norm2 = 0.0
    eta = 0.5
    tol = 1e-5
    grad = ∇f(x_sequence_t, b)
    cumu_grad .+= grad

    #learning rate
    alpha = sum(x_sequence_t .* (b .* b)) / (x_sequence_t ⋅ b) / (b ⋅ b)
    v = b .* (grad .+ alpha)
    cumu_grad_norm2 += (v ⋅ v)
    eta = m^0.5 / (2 * m^0.5 + cumu_grad_norm2^0.5)

    #newton's method
    lamb = 1.0 + eta * maximum(- cumu_grad)
    s = lamb .+ eta .* cumu_grad
    dpsi = 1.0 - sum(1.0 ./ s)
    ddpsi = sum( 1.0 ./ (s .* s))
    while dpsi^2 > tol * ddpsi
        lamb -= dpsi / ddpsi
        s = lamb .+ eta .* cumu_grad
        dpsi = 1.0 - sum(1.0 ./ s)
        ddpsi = sum(1.0 ./ (s .* s))
    end
    b .= (1.0 ./ s)
    return b
end

#Chungen algo 2
function last_grad_opt_lb_ftrl(x_sequence_t, T, b)
    m = length(x_sequence_t) #number of targets
    #b = ones(m) / m
    cumu_grad = zeros(m)
    eta = 1.0 / (16.0 * 2^0.5)

    prev_b = ones(m) / m
    prev_x = nothing
    variation = 0.0
    tol = 1e-5

    grad = ∇f(x_sequence_t, b)
    cumu_grad .+= grad
    v = b .* grad

    #learning rate
    if prev_x !== nothing
        temp = prev_x .* (∇f(prev_b, prev_x) - ∇f(x_sequence_t, prev_b))
        variation += (temp ⋅ temp)
        eta = m^0.5 / (16 * (2 * m)^0.5 + variation^0.5)
    end

    #newton's method
    lamb = 1.0 + eta * maximum(- cumu_grad)
    s = lamb .+ eta .* cumu_grad
    dpsi = 1.0 .- sum((1.0 .- eta * v) ./ s)
    ddpsi = sum((1.0 .- eta * v) ./ (s .* s))
    while dpsi^2 > tol * ddpsi
        lamb -= dpsi / ddpsi
        s = lamb .+ eta .* cumu_grad
        dpsi = 1.0 - sum((1.0 .- eta * v) ./ s)
        ddpsi = sum((1.0 .- eta * v) ./ (s .* s))
    end
    prev_b = b
    prev_x = x_sequence_t
    b .= (1.0 .- eta * v) ./ s
    return b
end

# benchmarks- offline
# Cover's method (slow) 1984
# T. Cover. An algorithm for maximizing expected log investment return, 1984.
function cover(x_sequence) #x_sequence: Market history
    m = length(x_sequence[1]) # # of targets

    #b = ones(m) / m #portfolio init
    for t in 1:length(x_sequence) #number of time periods
        b = ones(m) / m #portfolio init
        #update
        grad = ∇Ef(A, b) #A = x_sequence[t]?
        b = multiply(b, -grad)
        #regrets[alg][t] = (regrets[alg][t-1] + f(x_sequence[t], x)) if t >= 1 else f(x_sequence[t], x)
    end
    b ./= sum(b)

    #print("cover's method portfolio: ", b, "\n")

    return b
end

#optimization methods 
#entropic mirror descent
function eg_armijo(f, ∇f, x, α_bar, r, τ)
    α = copy(α_bar) #stepsize
    j = 1
    x_new = exp.(log.(x) - α * ∇f)
    x_new /= sum(x_new) #C_k^{-1}; normalization
    while f(x_new) > f(x) + τ * ( ∇f ⋅ (x_new - x))
        α = α_bar * r^j
        j += 1
        x_new = exp.(log.(x) - α * ∇f)
        x_new /= sum(x_new)
    end
    #print("x_new: ", x_new, "\n")
    return x_new
end

function polyak(f, ∇f, T, X, b, δ1, B, c, max_iter) #T: update rule,X: constraint set
    x = copy(b)
    δ = δ1
    σ = 0
    l = 1
    k_l = 1
    k = 1
    η = 0.1

    for iter in 1:max_iter #for all k \in N do
        if norm(∇f(x)) == 0
            return x 
        end

        f_rec = Inf #f_rec: the min value of f observed so far
        for κ in 1:k #k: current iteration
            f_rec = min(f_rec, f(x))
        end

        if f(x) ≤ f_rec - 0.5 * δ
            k_l = k
            σ = 0
            δ = δ
            l += 1
        elseif σ > B
            k_l = k
            σ = 0
            δ = 0.5 * δ
            l += 1
        end

        f̂_k = f_rec - δ
        η_k = (f(x) - f̂_k) / (c * norm(∇f(x))^2)

        #update x
        x = T(x, η_k, ∇f(x))
        σ += c * η_k * norm(∇f(x))
        k += 1
    end
    return x
end

#update rule T(x, η)
function T_polyak(x, η, ∇f) #x: current iterate
    x -=  η * ∇f
    x .= max.(x, 0.0)
    x ./= sum(x)
    return x
end

function sub_gd(returns, x_t, α, max_iter)
    m = length(returns[1])
    b = ones(m) / m

    for k in 1:max_iter
        subgrad = ∇f(b, returns, x_t) 
        b -= α * subgrad  # Update current point
        b .= max.(b, 0.0) #non-negative
        b ./= sum(b) #normalization
    end
    return b
end

function constraint_set(b)
    return all(b .>= 0.0) && sum(b) == 1.0
end
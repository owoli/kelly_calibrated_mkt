using LinearAlgebra
using Statistics

#goal: lin inf log S_T^*/S_T >= 0 as T -> +Inf
# objective function
# returns: vector/matrix of price relatives

function f_kelly(b::Vector{Float64}, returns, x_t::Vector{Float64}) #b: portfolio, x_t: calibrated mkt distribution estimation
    return - sum(x_t[i] * (log(b ⋅ returns[i])) for i in 1:length(returns))
end

function ∇f_kelly(b::Vector{Float64}, returns, x_t::Vector{Float64})
    subgrad = zeros(length(b))
    for i in 1:length(returns)
        subgrad .+= (- x_t[i] * returns[i] / (b ⋅ returns[i]))
    end
    return subgrad
end

function f_bcrp(b, x_sequence) #best constant rebalanced portfolio based on past(1~t-1) Market return vector
    return - sum(log(x_sequence[t] ⋅ b) for t in 1:length(x_sequence))
end

function ∇f_bcrp(b, x_sequence) #b: portfolio, x_sequence: Market vector from t=1 to T
    subgrad = zeros(length(b))
    for t in 1:length(x_sequence)
        subgrad .+= (- x_sequence[t] / (b ⋅ x_sequence[t]))
    end
    subgrad ./= length(x_sequence) #./= length(x_sequence) added
    return subgrad
end

function sumf(x_sequence, b)
    return - sum(log(x_sequence[t] ⋅ b) for t in 1:length(x_sequence))
end

function ∇Ef(x_sequence, b)
    grad = zeros(length(b))
    for t in 1:length(x_sequence)
        grad .+= (- x_sequence[t] / (b ⋅ x_sequence[t]))
    end
    grad ./= length(x_sequence)
    return grad #average gradient for each target?
end

function f(mkt_sequence_t, b) #a: market sequence at time t, x: portfolio
    return -log(mkt_sequence_t ⋅ b)
end

function ∇f(mkt_sequence_t, b)
    return - mkt_sequence_t / (mkt_sequence_t ⋅ b)
end

function Ef(x_sequence, b)
    return - sum(log(x_sequence[t] ⋅ b) for t in 1:length(x_sequence)) / length(x_sequence)
end

# T. Cover. Universal Portfolios, 1991.
# http://www-isl.stanford.edu/~cover/papers/paper93.pdf
function bcrp(x_sequence, t)
    n = length(x_sequence) # # of time periods
    m = length(x_sequence[1]) # # of targets

    r = 0.9
    τ = 0.1
    α_bar = r * sqrt(log(m)/t)
    #α_bar = 0.1
    b = ones(m) / m
    b_update = eg_armijo(b -> f_bcrp(b, x_sequence), ∇f_bcrp(b, x_sequence), b, α_bar, r, τ)
    while abs(f_bcrp(b_update, x_sequence) - f_bcrp(b, x_sequence)) > 1e-5
        b = b_update
        b_update = eg_armijo(b -> f_bcrp(b, x_sequence), ∇f_bcrp(b, x_sequence), b, α_bar, r, τ)
    end

    return b_update
end

#benchmark- offline / batch
#Cover's method (slow) 1984
#T. Cover. An algorithm for maximizing expected log investment return, 1984.
function cover(mkt_sequence)
    d = length(mkt_sequence[1]) # # of targets; dimension
    b = ones(d) / d # portfolio

    for t in 1:length(mkt_sequence)
        b = ones(d) / d #portfolio init
        for _ in 1:Int(ceil(log(d) * (t+1) / 10))
            grad = ∇Ef(mkt_sequence, b)
            b = b .* (- grad)
        end
    end
    #b ./= sum(b)

    return b
end

# Asymptotic Optimality and Asymptotic Equipartition Properties of Log-optimum Investment
# By Paul H. Algoet and Thomas M. Cover 
function kelly(x_t::Vector{Float64}, returns, t, T, b_last)
    m = length(returns[1]) # # of targets

    r = 0.9
    τ = 0.1
    α_bar = r * sqrt(log(m)/t)
    #α_bar = 0.1 #step size upper bound
    #tol = 1e-5
    if returns == ([1.0, 0.0], [0.0, 1.0])
        tol = 1e-20
    else
        tol = 1e-5
    end
    
    # optimization- eg_armijo, polyak step size
    #b = copy(b_last)
    b = ones(m) / m
    b_update = eg_armijo(b -> f_kelly(b, returns, x_t), ∇f_kelly(b, returns, x_t), b, α_bar, r, τ)
    
    while abs(f_kelly(b_update, returns, x_t) - f_kelly(b, returns, x_t)) > tol #function value yield from kelly_portfolio
        b = b_update
        b_update = eg_armijo(b -> f_kelly(b, returns, x_t), ∇f_kelly(b, returns, x_t), b, α_bar, r, τ)
    end
    b = b_update

    return b
end

# benchmarks - online

#加round 0 (t+1), uniform distributed on round 0 (counts + 1/m)
function kt_mix(mkt_sequence_t, t, counts, prior)
    m = length(mkt_sequence_t)
    prior = ones(Float64, m) / m
    counts .+= mkt_sequence_t
    posterior = (counts .+ prior) / (t + 1)
    return posterior
end

function soft_bayes(mkt_sequence_t, T, b_last)

    # init
    m = length(mkt_sequence_t) # # of targets; dimension
    eta = (log(m) / (m * T))^0.5 #ok
    b = copy(b_last) #ok

    # update
    b = b .* (1.0 .- eta .- eta * ∇f(mkt_sequence_t, b))

    return b
end

function lb_ftrl(mkt_sequence, T, b_last)

    # init
    d = length(mkt_sequence[t]) #number of targets; dimension
    b = copy(b_last)
    η = (d * log(T) / (2 * T))^0.5
    cumu_grad = zeros(d)

    # update
    # Newton's method
    tol = 1e-5
    for t in 1:T
        cumu_grad += ∇f(mkt_sequence[t], b)
        lamb = 1.0 + η * maximum(-cumu_grad) #np.amax = maximum? 
        s = lamb .+ η .* cumu_grad 
        dpsi = 1.0 - sum(1.0 ./ s) #psi differentiation
        ddpsi = sum(1.0 ./ (s .* s))
        while dpsi^2 > tol * ddpsi
            lamb -= dpsi / ddpsi
            s = lamb .+ η .* cumu_grad
            dpsi = 1.0 - sum(1.0 ./ s)
            ddpsi = sum(1.0 ./ (s .* s))
        end
        b = (1.0 ./ s)
    end
    return b
end

#Chung-En Tsai et al, Data-Dependent Bounds for Online Portfolio Selection Without Lipschitzness 
#and Smoothness, 2023
# algorithm 3
function ada_lb_ftrl(mkt_sequence_t, b_last, ada_cumu_grad, ada_cumu_grad_norm2)

    # init
    d = length(mkt_sequence_t) # number of investment alternatives
    b = copy(b_last)
    ada_η = 0.5
    tol = 1e-5

    # update
    grad = ∇f(mkt_sequence_t, b)
    global ada_cumu_grad .+= grad

    #    learning rate
    alpha = sum(mkt_sequence_t .* b .* b) / (mkt_sequence_t ⋅ b) / (b ⋅ b)
    v = b .* (grad .+ alpha)
    global ada_cumu_grad_norm2 += (v ⋅ v)
    global ada_η = d^0.5 / (2 * d^0.5 + ada_cumu_grad_norm2^0.5)

    #    Newton's method
    lamb = 1.0 + ada_η * maximum(- ada_cumu_grad)
    s = lamb .+ ada_η .* ada_cumu_grad #*
    dpsi = 1.0 - sum(1.0 ./ s)
    ddpsi = sum(1.0 ./ (s .* s))
    while dpsi^2 > tol * ddpsi
        lamb -= dpsi / ddpsi
        s = lamb .+ ada_η .* ada_cumu_grad #*
        dpsi = 1.0 - sum(1.0 ./ s)
        ddpsi = sum(1.0 ./ (s .* s))
    end
    b .= (1.0 ./ s)
    return b
end

# algorithm 4
function last_grad_opt_lb_ftrl(x_sequence_t, b_last, lg_cumu_grad, lg_prev_b, lg_prev_x, lg_variation, lg_η)

    # init
    m = length(x_sequence_t) #number of targets; dimensions
    tol = 1e-5
    b = copy(b_last)
    #η = 1.0 / (16.0 * 2^0.5)

    #update
    grad = ∇f(x_sequence_t, b)
    global lg_cumu_grad .+= grad
    v = b .* grad

    #   learning rate
    if lg_prev_x !== nothing
        temp = lg_prev_b .* (∇f(lg_prev_x, lg_prev_b) - ∇f(x_sequence_t, lg_prev_b))
        global lg_variation += (temp ⋅ temp)
        global lg_η = m^0.5 / (16 * (2 * m)^0.5 + lg_variation^0.5)
    end

    #   Newton's method
    lamb = 1.0 + lg_η * maximum(- lg_cumu_grad)
    s = lamb .+ lg_η .* lg_cumu_grad
    dpsi = 1.0 .- sum((1.0 .- lg_η * v) ./ s)
    ddpsi = sum((1.0 .- lg_η * v) ./ (s .* s))
    while dpsi^2 > tol * ddpsi
        lamb -= dpsi / ddpsi
        s = lamb .+ lg_η .* lg_cumu_grad
        dpsi = 1.0 - sum((1.0 .- lg_η * v) ./ s)
        ddpsi = sum((1.0 .- lg_η * v) ./ (s .* s))
    end
    global lg_prev_x = x_sequence_t
    global lg_prev_b = b
    b .= ((1.0 .- lg_η * v) ./ s)
    return b
end

# optimization methods 
# entropic mirror descent
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

# update rule T(x, η)
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
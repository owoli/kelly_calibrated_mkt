using LinearAlgebra
using Combinatorics

#oco algorithm
#loss function: f_t(w) = - d(r, S) = - (w ⋅ r_t - h_s(w))
#-: concave -> convex
#h_s(w) := sup_{s in S} <w,s>; i.e. the dual norm of w; S (1-norm ε ball)
function shimkin_step(t, w::Vector{Vector{Float64}}, # current iterate
    r_t::Vector{Vector{Float64}}, # reward vector
    η::Float64,              # learning rate 
    R::Float64               # radius of approach set S (ε_1-norm ball)  
    )::Vector{Vector{Float64}}

    m = length(r_t[1]) # # of price relative distributions
    n = length(r_t) # # of Forecaster strategies

    ∇f_t = - r_t #+ ∇h(w, R)
    w_vector = ogd(η, reduce(vcat, ∇f_t), reduce(vcat, w), reduce(vcat, r_t), t, R)
    #w_vector = EG_pm_vector(t, reduce(vcat, loss), reduce(vcat, w), reduce(vcat, grad), η, R)
    w = [ w_vector[i:i+m-1] for i in 1:m:length(w_vector) ]
    
    return w
end

h(w, R) = R * norm(w, Inf) #the inf norm of w scaled by R
function ∇h(w::Vector{Vector{Float64}}, R::Float64 # radius of approach set S (ε_1-norm ball)
    )   
    g = Vector{Vector{Float64}}(undef, length(w))
    for i in 1:length(w)
        g[i] = zeros(size(w[i]))
        ~, idx = findmax(abs.(w[i])) # index of the maximum abs value of w
        g[i][idx] = sign(w[i][idx]) * R
    end
    return g 
end

#W = B_2, 2-norm unit ball
#Bregman divergnece
function ogd(η, ∇f_t, w, r_t, t, R)
    η /= sqrt(t) #Zinkevich 2003

    #w -= η * ∇f_t .* r_t #(.* r_t): 僅更動單一 w entry值
    w += η * ∇f_t
    
    #project back to 2-norm ball
    w_proj = w ./ max(1,norm(w, 2))

    return w_proj
end

# input s =  w_{t-1}
# y_hat = (w_t_plus - w_t_minus) ⋅ loss_t
function EG_pm_vector(t, r_t, w::Vector{Float64}, # current iterate 
    ∇f_t::Vector{Float64}, # gradient 
    η::Float64, # learning rate
    R::Float64 # radius of the \ell_1-norm ball 
    )::Vector{Float64}
    #doubling the dimensions; 原位於1-norm ball之點w 拆成w_plus與w_minus後接起來的vector會位於prob. simplex上
    s_plus = Vector{Float64}()
    for i in 1:length(w)
        push!(s_plus, w[i] / 2)
    end
    s_minus = copy(s_plus)
    U = norm([s_plus; s_minus], 1)
    #U = sum([s_plus; s_minus])
    w_plus = U * s_plus
    w_minus = U * s_minus

    #X: estimated upper bound for the maximum L_{inf} norm of the instances
    X = norm(w, Inf) * R
    η = 1/(3 * U^2 * X^2)
    r_plus  = exp.(- η * ∇f_t * U .* r_t)
    r_minus = 1 ./ r_plus
    w_plus  .*= r_plus
    w_minus .*= r_minus
    s = sum([w_plus; w_minus])
    #s = norm([w_plus; w_minus], 1)
    #s = sum(w_plus)
    w_plus ./= s
    w_minus ./= s
    w_plus *= U  #scaling 考慮的問題非unit one-norm ball
    w_minus *= U

    return w_plus - w_minus
end

#=
function EG_pm(w::Vector{Float64}, # current iterate 
    ∇f_t::Vector{Float64}, # gradient 
    η::Float64, # learning rate
    R::Float64, r_t # radius of the \ell_1-norm ball 
    )::Vector{Float64}

    r_plus  = exp.(- η * R * ∇f_t)
    r_minus = 1 ./ r_plus
    w_plus  = w .* r_plus 
    w_minus = w .* r_minus 
    U       = sum([w_plus; w_minus])
    w_plus ./= U
    w_minus ./= U
    w_plus  = w_plus * R
    w_minus = w_minus * R 

    return w_plus - w_minus
end

function EG_pm(loss, w::Vector{Vector{Float64}}, # current iterate
    ∇f_t::Vector{Vector{Float64}}, # gradient
    η::Float64, # learning rate
    R::Float64 # radius of the \ell_1-norm ball 
    )::Vector{Vector{Float64}}

    weight_update = Vector{Vector{Float64}}(undef, length(w))
    r_plus = Vector{Vector{Float64}}(undef, length(w)) #exp(-η*∇L_{weight_last iterate})
    r_minus = Vector{Vector{Float64}}(undef, length(w))
    w_plus = Vector{Vector{Float64}}() #undef, length(w)
    w_minus = Vector{Vector{Float64}}() #undef, length(w)
    s = Vector{Float64}(undef, length(w))

    for i in 1:length(w) 
        push!(w_plus, R * 1/(2*length(w)*length(w[1]))* ones(length(w[1])))
        push!(w_minus, R * 1/(2*length(w)*length(w[1]))* ones(length(w[1])))
    end

    for i in 1:length(w) #每個w[i]為一2維向量, 共i=101個, w[i][1], w[i][2] 方為單一元素
        r_plus[i] = exp.(-r_t[i]) #-η * R * grad[i]
        r_minus[i] = 1 ./ r_plus[i]
        w_plus[i] = w[i] .* r_plus[i]  #element-wise multiplication of w with exponentiated negative and positive η * R * grad 
        w_minus[i] = w[i] .* r_minus[i]
        s[i] = sum([w_plus[i]; w_minus[i]]) #normalization const
    end
    U = 0 #total weight the weight matrices 2*101 elements
    for i in 1:length(s)
        U += s[i]
    end
    for i in 1:length(w)
        w_plus[i] ./= U
        w_minus[i] ./= U
    end

    return w_plus - w_minus
end
=#
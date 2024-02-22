# Author: Yen-Huan Li
# Email: yenhuan.li@csie.ntu.edu.tw 

include("egt.jl")

function blackwell_approach(ε::Float64, 
    f_bar::Vector{Float64}, 
    payoff::Function, 
    projection::Function, 
    n_x::Int, 
    n_y::Int)

    f_proj = projection(f_bar)

    A = zeros(n_y, n_x)
    for i in 1: n_y
        for j in 1: n_x
            A[i, j] = (f_bar - f_proj)' * payoff(j, i) 
        end
    end

    p, q = APD(ε, A, "d")
    return rand(Categorical(p))
end
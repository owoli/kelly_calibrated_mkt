#project a given vector onto the probability simplex with respect to the 2-norm
#michelot: iteratively projects the vector onto the simplex until convergence
#input: average reward vector c 
#output: its proj on the simplex
#1. finds the non-zero elements of vector x, denoted by v
#2. calculates the average of the non-zero elements, ρ
#3. shifts v by subtracting ρ from each element
#4. If all elements of v are non-negative, returns the projected vector
# else, sets negative elements of v to zero and repeats the process

#condat:  can be faster than Michelot for larger dimensions
#input : average reward vector y , scalar a(set to 1.0 if not provided)
#projects y onto the probability simplex, with a controlling the sparsity of the result
#output: its projection onto the simplex

#test_proj_simplex: tests the projection accuracy of michelot and condat
#input: integer n, generates a random vector of length n, and compares the results of both projection algos
#returns the difference between the outputs of the two algorithms (measured by the Euclidean norm). If difference close to zero, it indicates that both algorithms provide similar results.

# Author: Yen-Huan Li
# Email: yenhuan.li@csie.ntu.edu.tw
# Projection onto the probability simplex wrt the 2-norm 

using LinearAlgebra

# L. Condat. Fast projection onto the simplex and the ell1 ball. Math. Program., Ser. A. 2016. 
# About 7x--8x faster than Michelot empirically
function condat(y::Vector{Float64}, a::Float64) #simplex sum up to a = 1.00 * R
    N       = length(y)
    v       = [y[1]]
    #print("v = y[1]: ", v, "\n")
    v_tilde = Float64[]
    ρ       = y[1] - a
    #print("ρ: ", ρ,"\n")
    #ρ       = abs(y[1] - a)
    #print("ρ(abs): ", ρ, "\n")

    for n in 2:N
        if y[n] > ρ
            #print("if the avg reward element is greater than", ρ, "\n")
            ρ = ρ + (( y[n] - ρ ) / (length(v) + 1.0)) #length(v), |v|
            #print("reset ρ value to be ρ+((y[n]-ρ)/(length(v)+1))", ρ, "\n")
            if ρ > y[n] - a #1.0
                push!(v, y[n])
            else
                append!(v_tilde, v)
                v = [y[n]]
                ρ = y[n] - a
            end
        end
    end

    if ~isempty(v_tilde)
        for y in v_tilde
            if y > ρ
                push!(v, y)
                ρ = ρ + ((y - ρ) / length(v)) #length(v), |v|
            end
        end
    end

    while ~all(v .> ρ) #while |v| changes; while there remain element in v that <= ρ
        i = 1
        while i <= length(v)
            z = v[i]
            if z <= ρ
                ρ = ρ + ((ρ - z) / ( length(v) - 1.0 )) # length(v), |v|
                deleteat!(v, i)
                i = i - 1
            end
            i = i + 1
        end
    end

    #print("v: ", v, "\n")

    return max.(y .- ρ, 0.0)

end

# C. Michelot. A finite algorithm for finding the projection of a point onto the canonical simplex of R^n. J. Optim. Theory Appl. 1986. 
function michelot(c::Vector{Float64}, R::Float64)
    x = c
    while true
        idx = x .!= 0.0
        v   = x[idx]
        ρ   = (sum(v) - 1.00 * R) / length(v) #ρ   = (sum(v) - 1.0) / length(v)
        #ρ = abs(ρ) #add
        v   = v .- ρ
        if all(v .>= 0.0)
            x[idx] = v
            return x
        else
            x[idx] = max.(v, 0.0)
        end
    end

end

function condat(y::Vector{Float64})
    return condat(y, 1.0)
end

function test_proj_simplex(n::Int)
    a = rand(n)
    return norm(michelot(a) - condat(a))
end

#return the max discrepancy between r and all possible strategies in S, based on the inner product difference with the optimization variable w.
function d_r_S(w::Vector{Vector{Float64}}, r::Vector{Vector{Float64}}, S::Vector{Vector{Vector{Float64}}})
    w = w ./ norm(w,2) #normalize w to lie within the 2-norm ball
    #print("w(normalized): ", w, "\n")
    wr = w⋅r
    #maximum value of dot(w,s) for all s in S
    max_ws = maximum([dot(w,s) for s in S]) 
    dist = wr - max_ws

    #=max_discrepancy = -Inf
    for w in B2_ball
        wr = w⋅r
        #maximum value of dot(w,s) for all s in S
        max_ws = maximum([dot(w,s) for s in S]) 
        print("max_ws = ", max_ws, "\n")
        #w = w ./ norm(w,2) #normalize w to lie within the 2-norm ball
        #print("w(normalized): ", w, "\n")
        dist = wr - max_ws
        print("dist: ", dist,"\n")

        if dist > max_discrepancy
            max_discrepancy = dist
        end
    end
    print("max_discrepancy: ", max_discrepancy, "\n")
    return max_discrepancy=#
    return dist
end
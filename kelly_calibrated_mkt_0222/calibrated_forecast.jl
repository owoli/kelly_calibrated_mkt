using Combinatorics
using LinearAlgebra
using Distributions
using Plots

using Logging

include("apd.jl")
include("shimkin.jl")
include("proj_simplex.jl")
include("kelly.jl")
#include("blackwell.jl")
# super naive implementation
# \varepsilon = 0.001, A = 3
# \varepsilon = 0.01, A = 5

#△ discretize unit
function epsilon_net(ε::Float64, # desired accuracy
    m::Int #number of outcomes 
    )

    function c2p(c::Vector{Int})
        p = zeros(length(c) + 1)
        p[1] = c[1]
        p[2: end - 1] = diff(c)
        p[end] = t - c[end]
        return p 
    end

    t = Int(1.0 / ε)
    C = with_replacement_combinations(0:t, m - 1)
    P = [zeros(m) for _ in 1: length(C)]
    for (i, c) in enumerate(C)
        P[i] = c2p(c) .* ε 
    end
    return P
end

function shimkin_calibrated_forecast(
    w::Vector{Vector{Float64}},      # the last direction vector
    loss::Vector{Vector{Float64}},   # reward vector
    t::Int,                          # round index
    P::Vector{Vector{Float64}},      # epsilon net on the prob. simplex market distributions(m組)可能的weight distributions(n種)
    η, cumu_loss, loss_t_strategy, forecastor_strategy, loss_t_action
    )::Int

    # 1. obtain direction matrix w_t via an oco alg applied to loss function f_t(w) = - <w,r_t> + h_s(w) (d(r_t, S))
    if t > 1
        global w = shimkin_step(t, w, loss_t_strategy, η, R) #mixed action at round t
        #global w = shimkin_step(t, w, loss_t_action, η, R)
    end
    n = length(P) # # weight distibutions over m
    m = length(P[1]) # # of market price relative sets
    A = zeros(n, m) # minimax matrix

    # construct A (w ⋅ loss) matrix
    for i in 1:n, j in 1:m # ∀ Forcastor distribution i, ∀ Market distribution j
        δ_ij    = zeros(m)
        δ_ij[j] = 1
        #loss_t: matrix [0(vector),...,p_ij - δ_ij(vector),...,0(vector)]
        loss[i] = P[i] - δ_ij
        A[i,j] = w ⋅ loss
        loss[i] = loss[i] - (P[i] - δ_ij)
    end
    # 2. choose x_t (strategy; mixed action) s.t. <w,r(x,a)> - sup_{s \in S} <w,s> (h_s(w)) <= 0 ∀ Market outcomes a
    # minimax; min_{x \in △(n)} max_{y \in △(m)} <x, Ay>
    y, x, ~ = APD(1e-5, A, "d") # x: Forcastor strategy(mixed action) y: the likelihood of choosing each (m) price relative set
    global forecastor_strategy = copy(x)
    c = rand(Categorical(x)) # chosen action
    return c # index of the chosen action
end

#initialization
ε = 0.01 # partition unit
R = ε # radius/error tolerance of approach set S
returns = ([0.5, 1], [2.0, 1]) # [2.0, 2.0]
P = epsilon_net(ε, length(returns)) # △ partition
m = length(P[1]) # # of price relative sets; dimension d
n = length(P) # # of distibutions over price relative sets

#other parameters
#t = Int(1.0 / ε) # 以ε為partition unit, 需run 1/ε iterations
loss = fill(zeros(m), n)
loss_t_strategy = copy(loss)
loss_t_action = copy(loss)
cumu_loss = copy(loss)
cumu_loss_mixed = copy(loss)

T = 1000 # trading periods
η = 1.0
maxcount = 0

# approachability measurements
dist_condat = zeros(Float64, T)
dist_michelot = copy(dist_condat)
dist_mixedc = copy(dist_condat)

# oco
w = fill(1 / (n * m) * ones(m), n)
regret_w = zeros(Float64, T)
f_wstar = 0

forecastor_strategy = zeros(n) # Forcaster strategy, distributions over actions
x_sequence = Vector{Vector{Float64}}() # Market return vector history
empirical_frequencies = zeros(Float64, length(P))
empirical_counts = copy(empirical_frequencies)

# wealth at each time period
S_k = 1.0
S_b = 1.0
S_ada = 1.0
S_lg = 1.0
S_sb = 1.0
S_kelly = zeros(Float64, T)
S_bcrp = copy(S_kelly)
S_ada_lb_ftrl = copy(S_kelly)
S_last_grad = copy(S_kelly)
S_softb = copy(S_kelly)

# portfolio sequence
b_init = ones(length(returns[1])) / length(returns[1]) #uniform portfolio
kelly_sequence = fill(zeros(length(returns[1])), T+1)
kelly_sequence[1] = copy(b_init)
bcrp_sequence = copy(kelly_sequence)
ada_lb_ftrl_sequence = copy(kelly_sequence)
last_grad_sequence = copy(kelly_sequence)
softb_sequence = copy(kelly_sequence)

#online to batch
cumu_func_val = 0.0
cumu_func_val_batch = 0.0
Forecast_sequence = fill(zeros(m), T)

#optimization error
avg_func_val_kelly = zeros(Float64, T)
kelly_batch = copy(b_init)
avg_func_val_kbatch = zeros(Float64, T)

#export numerical result
#stdout_file = "output.txt"
#open(stdout_file, "w") do file
#    redirect_stdout(file)
#logfile = "logfile.txt"
#Logging.config(Dict(:logfile => logfile))
#redirect_stdout(stdout_file)
#redicrect_stdout()

for t in 1:T
    print("Trading period ", t, ": \n",)

    #A. predicting Market distribution P - calibrated forecasting
    c = shimkin_calibrated_forecast(w, loss, t, P, η, cumu_loss, loss_t_strategy, forecastor_strategy, loss_t_action)
    Forecast_sequence[t] = P[c]
    print("Forecastor prediction: ", P[c], "\n")
    
    #3. Market output action a_t, r_t = r(x_t(mixed action?),a_t)
    a_t = zeros(m)
    
    #oblivious Market: Market knows of Forecastor's strategy/algorithm
    #Mkt_distri = 1.0 .- P[rand(Categorical(forecastor_strategy))]
    
    #adaptive Market: Market knows of Forecastor's action
    #Mkt_distri = 1.0 .- P[c]
    
    #volatile Market
    #if t % 2 == 0
    #    Mkt_distri = [0, 1]
    #else
    #    Mkt_distri = [1, 0]
    #end
    
    if t < T/2
        Mkt_distri = [0.2, 0.8]
        #Mkt_distri = [0.1, 0.1, 0.8]
    else
        Mkt_distri = [0.8, 0.2]
        #Mkt_distri = [0.8, 0.1, 0.1]
    end

    j = rand(Categorical(Mkt_distri))
    a_t[j] = 1
    #x_sequence[t] = returns[j]
    push!(x_sequence, returns[j])

    #j = rand(1:m) #randomly chooses Market outcome
    #values = [i for i in 1:m]

    # Forecastor average cumulative loss
    global cumu_loss[c] += (P[c] - a_t)
    r = cumu_loss / t #avg cumulative loss matrix at round t

    # Forecastor round loss- single action P[c]
    global loss_t_action = copy(loss)
    global loss_t_action[c] += (P[c] - a_t)
    
    # Forecastor round loss- strategy/mixed action x
    global loss_t_strategy = copy(P)
    for i in 1:length(loss_t_strategy)
        global loss_t_strategy[i] -= a_t
    end
    global loss_t_strategy .*= forecastor_strategy
    
    # cumulative loss- strategy/mixed action 
    global cumu_loss_mixed += loss_t_strategy

    # approachability 1/t * sum r(i_t(single action), a_t)
    dist_condat[t] = norm(reduce(vcat,r) - condat(reduce(vcat,r), R), 2)
    dist_michelot[t] = norm(reduce(vcat,r) - michelot(reduce(vcat,r), R), 2)

    # mixed action 1/t * sum r(x_t(strategy/mixed action), a_t)
    dist_mixedc[t] = norm(reduce(vcat, cumu_loss_mixed/t) - condat(reduce(vcat,cumu_loss_mixed/t), R), 2)

    # empirical frequencies
    empirical_counts[c] += 1 # 第c種ditribution預測次數
    if a_t[1] == 1 # 預測第c種分布 且 市場output第1種情形
        empirical_frequencies[c] += 1
    end

    # B. Kelly portfolio selection

    # cumulative wealth
    global S_k *= (kelly_sequence[t] ⋅ x_sequence[t]) #Kelly portfolio(t - 1) ⋅ Market return vector at t
    global S_b *= (bcrp_sequence[t] ⋅ x_sequence[t])
    global S_sb *= (softb_sequence[t] ⋅ x_sequence[t])
    global S_ada *= (ada_lb_ftrl_sequence[t] ⋅ x_sequence[t])
    global S_lg *= (last_grad_sequence[t] ⋅ x_sequence[t])
    S_kelly[t] = S_k
    S_bcrp[t] = S_b
    S_softb[t] = S_sb
    S_ada_lb_ftrl[t] = S_ada
    S_last_grad[t] = S_lg

    # portfolio for t+1
    bcrp_sequence[t+1] = bcrp(x_sequence, t)
    #kelly_sequence[t+1] = kelly(P[c], returns, t, T, b_init)
    kelly_sequence[t+1] = kelly(P[c], returns, t, T, kelly_sequence[t])
    # assuming Kelly knows real Market distribution
    #if t < T/2
    #    push!(kelly_sequence, kelly([0.2, 0.8], returns, t, T, kelly_sequence[t]))
    #else
    #    push!(kelly_sequence, kelly([0.8, 0.2], returns, t, T, kelly_sequence[t]))
    #end
    #softb_sequence[t+1] = soft_bayes(x_sequence[t], T, softb_sequence[t])
    #ada_lb_ftrl_sequence[t+1] = ada_lb_ftrl(x_sequence[t], T, b_init)
    ada_lb_ftrl_sequence[t+1] = ada_lb_ftrl(x_sequence[t], T, ada_lb_ftrl_sequence[t])
    #last_grad_sequence[t+1] = last_grad_opt_lb_ftrl(x_sequence[t], T, b_init)
    last_grad_sequence[t+1] = last_grad_opt_lb_ftrl(x_sequence[t], T, last_grad_sequence[t])

    print("kelly port: ", kelly_sequence[t+1], "\n")

    #online to batch
    kelly_batch .+= kelly_sequence[t+1] #cumulative kelly portfolio
    global cumu_func_val += f_kelly(kelly_sequence[t+1], returns, P[c]) #cumulative function value acquired by kelly portfolio
    global cumu_func_val_batch += f_kelly(kelly_batch ./t, returns, P[c]) #cumulative function value, batch, kelly
    avg_func_val_kelly[t] = (cumu_func_val / t)
    avg_func_val_kbatch[t] = (cumu_func_val_batch / t)  #function value acquired by batch portfolio, kelly

    #=
    print(" A. Calibrated forecasting: ", "\n")
    print("     1. Forcaster outputs predicted Market distribution ", P[c], "\n")
    print("     2. Market outputs distribution ", a_t, "\n")
    print("     3. Forecaster suffers loss_t : ", P[c] - a_t, "\n")
    print("     4. Forecaster cumulative loss: ", r, "\n")
    =#
end

# a_t ~ P, P: (estimated) Market distribution
# f_t(x) = - log <a_t, x> #x: my portfolio f: loss function
# x_1,...,x_t: portfolio history
# F(x): cumulative
kelly_batch ./= T
cumu_func_val /= T
cumu_func_val_batch /= T

regret_kelly = zeros(Float64, length(x_sequence))
regret_softb = copy(regret_kelly)
regret_ada = copy(regret_kelly)
regret_last_grad = copy(regret_kelly)
for t in 1:length(x_sequence)
    regret_kelly[t] = log(S_bcrp[t]) - log(S_kelly[t])
    regret_softb[t] = log(S_bcrp[t]) - log(S_softb[t])
    regret_ada[t] = log(S_bcrp[t]) - log(S_ada_lb_ftrl[t])
    regret_last_grad[t] = log(S_bcrp[t]) - log(S_last_grad[t])
    #push!(regret_kelly, sum(log(1 + bcrp_sequence[i] ⋅ x_sequence[i]) for i in 1:t) - sum(log(1 + kelly_sequence[i] ⋅ x_sequence[i]) for i in 1:t))
end

#plot_opti_error = plot(1:T,[(avg_func_val_kelly[t] - avg_func_val_kbatch[t]) for t in 1:T],yaxis=:log,xlabel = "# of iterations", ylabel = "optimization error")
#savefig(plot_opti_error, "online to batch")

plot_all_log = plot(1:T, dist_condat, yaxis=:log, label = "Condat- action", xlabel = "Number of Epochs", ylabel = "D(r,S)(Log Scale)") #marker=:auto
plot_all_log = plot!(1:T, dist_michelot, yaxis=:log, label = "Michelot") #marker=:auto yaxis前  
one_over_sqrt_T = [1.0 / sqrt(t) for t in 1:T] #convergence rate: 1/sqrt(T)
plot_all_log = plot!(1:T, one_over_sqrt_T, yaxis=:log, label = "1/sqrt(T)")
plot_all_log = plot!(1:T, dist_mixedc, yaxis=:log, label = "Condat- strategy/mixed action")
savefig(plot_all_log, "approachability.png")

threshold = 0.005 * T
colors = [empirical_counts[i] < threshold ? :blue : "#db3f3d" for i in 1:length(empirical_counts)]

for i in 1:length(empirical_counts)
    if empirical_counts[i] != 0
        empirical_frequencies[i] /= empirical_counts[i]
    end
end

plot_distribution = scatter([P[i][1] for i in 1:length(P)], [empirical_frequencies[i] for i in 1:length(empirical_frequencies)], markersize = [log(empirical_counts[i]) for i in 1:length(empirical_counts)], markercolor = colors, fontfamily = "Times New Roman", xlabel = "predictions", ylabel = "empirical frequencies", seriestype = :scatter)
savefig(plot_distribution, "distribution.png")

d = length(returns[1]) # # of targets
plot_regrets = plot(1:T, regret_kelly, label="Kelly", xlabel="Investment period", ylabel="Regrets")
#plot_regrets = plot!(1:T, regret_softb, label="Soft Bayes")
plot_regrets = plot!(1:T, regret_ada, label = "ADA-LB-FTRL")
plot_regrets = plot!(1:T, regret_last_grad, label = "Last gradient optimistic LB-FTRL")
#plot_regrets = plot!(1:T, [sqrt(t) for t in 1:T], label="sqrt(t)")
#plot_regrets = plot!(1:T, [d * log(t) for t in 1:T], label = "O(dlogT)")
#plot_regrets = plot!(1:T, 1:T, label="linear")
savefig(plot_regrets, "regrets_kelly.png")

#redirect_stdout()
#end
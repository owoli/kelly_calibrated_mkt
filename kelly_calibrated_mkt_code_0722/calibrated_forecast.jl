using Combinatorics
using LinearAlgebra
using Distributions
using Plots

using Statistics #main

include("apd_temp.jl")
include("shimkin.jl")
include("proj_simplex.jl")
include("kelly.jl")
#include("blackwell.jl")

# super naive implementation
# \varepsilon = 0.001, A = 3
# \varepsilon = 0.01, A = 5

#discretize unit
function epsilon_net(ε::Float64, # desired accuracy/discretize unit
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
        #P[i] = c2p(c) .* ε
        P[i] = 1.0 .- (c2p(c) .* ε)
    end
    return P
end

function shimkin_calibrated_forecast(
    w::Vector{Vector{Float64}},      # the last direction vector
    loss::Vector{Vector{Float64}},   # reward vector
    t::Int,                          # round index
    P::Vector{Vector{Float64}},      # epsilon net on the prob. simplex market distributions(m組)可能的weight distributions(n種)
    η, cumu_loss, loss_t_strategy, forecastor_strategy, loss_t_action, c
    )::Int

    # 1. obtain direction matrix w_t via an OCO alg applied to f_t(w) = - d(r_t, S) = - <w,r_t> + h_s(w)
    # η = 1 / sqrt(t)
    if t > 1
        global w = shimkin_step(t, w, loss_t_strategy, η, R, c) #forecastor strategy loss
        #global w = shimkin_step(t, w, loss_t_action, η, R, c) #forecastor realization loss
    end
    n = length(P) # # of distibutions over mkt return vectors, forecastor actions
    m = length(P[1]) # # of market return vectors, market actions
    A = zeros(n, m) #minimax matrix

    #A: minimax martix; w ⋅ loss = min_x max_j <w,loss(mixed action x, market action j)> - h_S(w) <= 0
    for i in 1:n, j in 1:m # ∀ forcastor prediction i, ∀ market distribution j
        #market distributions
        δ_ij    = zeros(m)
        δ_ij[j] = 1
        #loss matrix: [0(m dim vector),...,p_ij - δ_ij(m dim vector),...,0]
        loss[i] = P[i] - δ_ij
        A[i,j] = w ⋅ loss - R * norm(w, Inf)
        loss[i] = zeros(m)
    end

    # 2. choose x_t (strategy) s.t. 
    # d(r, S) =  <w,r(x,a)> - max_{s \in S} <w,s> <= 0 (avg r和其在S上的projection dist<=0; 落入S內) 
    # ∀ mkt distribution, exists a strategy x s.t. min_{x in △_n} max_{y in J} <x, Ay>
    # x: mixed action of prob distri over price relative sets
    # y: weighted sum of prog of choosing each (m) price relative set
    x, y, ~ = APD(1e-5, A, "d") #apd_temp.jl
    #print("Forecastor strategy: ", y,"\n")
    global forecastor_strategy_target_sum = copy(y)
    global forecastor_strategy = copy(x)
    c = rand(Categorical(x)) # the index of chosen Forecastor prediction
    return c
end

# initialization
ε = 0.01 # prob distri discretize unit
R = ε # radius of approach set S

#returns = ([1.0, 0.0], [0.0, 1.0]) #prob forecasting
returns = ([0.5, 1.0], [2.0, 1.0])
#shift_const = 0.2
#s_vec = shift_const * ones(size(returns[1]))
#for i in 1:length(returns)
#    returns[i] .+= s_vec
#end
#returns = ([1.0, 0.3], [0.3, 1.0])

P = epsilon_net(ε, length(returns)) # △ partition
m = length(P[1]) # # of mkt return vectors; dimension M
n = length(P) # # of distri over mkt return vectors

#other parameters
#t = Int(1.0 / ε) #以ε discretize, 需run 1/ε iterations

#Forecastor loss action/strategy
loss = fill(zeros(m), n)
loss_t_strategy = copy(loss)
cumu_loss_mixed = copy(loss)
loss_t_action = copy(loss)
cumu_loss = copy(loss)


T = 300 # trading periods
η = 1.0

# approachability measurements
dist_condat = zeros(Float64, T)
dist_michelot = copy(dist_condat)
dist_mixedc = copy(dist_condat)

# oco
last_forecast_index = 0
w = fill(1 / (n * m) * ones(m), n)

forecastor_strategy = zeros(n) #Forcaster strategy
x_sequence = Vector{Vector{Float64}}() #mkt return vector sequence
empirical_frequencies = zeros(Float64, length(P))
empirical_counts = copy(empirical_frequencies)

# wealth sequence
S_kelly_log = zeros(Float64, T)
S_ada_lb_ftrl_log = copy(S_kelly_log)
S_last_grad_log = copy(S_kelly_log)
S_softb_log = copy(S_kelly_log)
S_kt_seq_log = copy(S_kelly_log)
S_kelly_real_log = copy(S_kelly_log)
wealth_sequence_log = [S_kelly_log, S_ada_lb_ftrl_log, S_last_grad_log, S_softb_log, S_kt_seq_log, S_kelly_real_log]

S_bcrp_log = zeros(Float64, T)

# portfolio sequence
b_init = ones(length(returns[1])) / length(returns[1]) #uniform portfolio

kelly_sequence = fill(zeros(length(returns[1])), T+1)
kelly_sequence[1] = copy(b_init)
ada_lb_ftrl_sequence = copy(kelly_sequence)
last_grad_sequence = copy(kelly_sequence)
softb_sequence = copy(kelly_sequence)
kt_sequence = copy(kelly_sequence)
kelly_real_sequence = copy(kelly_sequence)
bcrp_sequence = copy(kelly_sequence)
bcrp_cover_sequence = copy(kelly_sequence)
portfolio_sequence = [kelly_sequence, ada_lb_ftrl_sequence, last_grad_sequence, softb_sequence, kt_sequence, kelly_real_sequence]

# online to batch
cumu_func_val = 0.0
cumu_func_val_batch = 0.0
Forecast_sequence = fill(zeros(m), T)

# optimization error
avg_func_val_kelly = zeros(Float64, T)
kelly_batch = copy(b_init)
avg_func_val_kbatch = zeros(Float64, T)

p = 0.0 #predictions vs. empirical frequencies
kt_counts = zeros(Float64, length(returns[1]))

#ada_lb_ftrl init
ada_cumu_grad = zeros(m) # as the upperbound of regret
ada_cumu_grad_norm2 = 0.0 # dual local norm associated with the log-barrier?

#lg_opt_lb_ftrl init
lg_cumu_grad = zeros(m)
lg_η = 1.0 / (16.0 * 2^0.5)
lg_prev_b = ones(m) / m
lg_prev_x = nothing
lg_variation = 0.0

#Calibration error
forecastor_strategy_target_sum = zeros(m)
cumu_empirical_freq = zeros(m)
prediction_error = zeros(m)

for t in 1:T
    #print("Trading period ", t, ": \n",)

    # A. predicting Market distribution P at t - calibrated forecasting
    c = shimkin_calibrated_forecast(w, loss, t, P, η, cumu_loss, loss_t_strategy, forecastor_strategy, loss_t_action, last_forecast_index)
    global last_forecast_index = c
    Forecast_sequence[t] = P[c]
    #print("Forecastor prediction: ", P[c], "\n")
    
    # 3. Market output action a_t, r_t = r(x_t(strategy?),a_t)
    a_t = zeros(m)
    
    # oblivious market: market knows of Forecastor's strategy
    # single realization (accord. -Forecastor strategy distri)
    #Mkt_distri = 1.0 .- P[rand(Categorical(forecastor_strategy))]

    # oblivious, yet might not be the worst case 
    #Mkt_distri = 1.0 .- P[Int(mean(rand(Categorical(forecastor_strategy))))]

    # adaptive Market: Market knows of Forecastor's realization/action
    #Mkt_distri = 1.0 .- P[c]

    #Chiatse: output the opposite side（Forcastor: [p, 1-p], p < 0.5 output [0,1]...)
    #if P[c][1] > 0.5
    #    Mkt_distri = [0.0, 1.0]
    #else
    #    Mkt_distri = [1.0, 0.0]
    #end

    Mkt_distri = [0.8, 0.2]
    #Mkt_distri = [1.0, 0.0]
    #if t < T/2
    #    Mkt_distri = [1.0, 0.0]
    #   Mkt_distri = [0.2, 0.8]
    #else
    #    Mkt_distri = [0.0, 1.0]
    #   Mkt_distri = [0.8, 0.2]
    #end

    j = rand(Categorical(Mkt_distri))
    a_t[j] = 1
    push!(x_sequence, returns[j])
    #j = rand(1:m) #randomly chooses Market outcome
    #values = [i for i in 1:m]

    #cumulative
    global cumu_empirical_freq += a_t
    #global prediction_error .+= abs.(cumu_empirical_freq/t .- forecastor_strategy_target_sum)
    global prediction_error .+= abs.(cumu_empirical_freq/t .- P[c])
    #print("mkt empirical frequencies: ", cumu_empirical_freq/t,"\n")
    #print("prediction error: ", abs.(cumu_empirical_freq[1]/t .- P[c][1]) + abs.(cumu_empirical_freq[2]/t .- P[c][2]),"\n")
    #global prediction_error .+= abs.(Mkt_distri .- P[c])

    # Forecastor avg cumulative loss
    global cumu_loss[c] += (P[c] - a_t) #forecastor distribution - market action
    r = cumu_loss / t #avg cumulative loss matrix at time period t

    #(r_t) Forecastor round loss- action/realization P[c]
    global loss_t_action = copy(loss)
    global loss_t_action[c] += (P[c] - a_t)
    
    #(r_t) Forecastor round loss- strategy x
    global loss_t_strategy = copy(P) #forecastor distribution
    for i in 1:length(loss_t_strategy)
        global loss_t_strategy[i] -= a_t #market distribution
    end
    global loss_t_strategy .*= forecastor_strategy
    
    # cumulative loss- strategy
    global cumu_loss_mixed += loss_t_strategy

    # approachability 1/t * sum r(i_t(single action), a_t)
    dist_condat[t] = norm(reduce(vcat,r) - condat(reduce(vcat,r), R), 2)
    #dist_michelot[t] = norm(reduce(vcat,r) - michelot(reduce(vcat,r), R), 2)

    # mixed action 1/t * sum r(x_t(strategy/mixed action), a_t)
    dist_mixedc[t] = norm(reduce(vcat, cumu_loss_mixed/t) - condat(reduce(vcat,cumu_loss_mixed/t), R), 2)

    # empirical frequencies
    empirical_counts[c] += 1 # P[c](distribution) prediction count
    if a_t[1] == 1 # predict P[c] && market output the 1st market return vector
        empirical_frequencies[c] += 1
    end
    
    #B. Kelly portfolio selection
    kelly_real_sequence[t] = kelly(Mkt_distri, returns, t, T, kelly_real_sequence[t])

    # cumulative wealth
    for i in 1:length(portfolio_sequence)
        if t == 1
            wealth_sequence_log[i][t] += f(x_sequence[t], portfolio_sequence[i][t])
        else
            wealth_sequence_log[i][t] += (wealth_sequence_log[i][t-1] + f(x_sequence[t], portfolio_sequence[i][t]))
        end
    end

    # portfolio for t+1
    kelly_sequence[t+1] = kelly(P[c], returns, t, T, kelly_sequence[t])
    ada_lb_ftrl_sequence[t+1] = ada_lb_ftrl(x_sequence[t], ada_lb_ftrl_sequence[t], ada_cumu_grad, ada_cumu_grad_norm2)
    last_grad_sequence[t+1] = last_grad_opt_lb_ftrl(x_sequence[t], last_grad_sequence[t], lg_cumu_grad, lg_prev_b, lg_prev_x, lg_variation, lg_η)
    softb_sequence[t+1] = soft_bayes(x_sequence[t], T, softb_sequence[t])
    kt_sequence[t+1] = kt_mix(x_sequence[t], t, kt_counts, kt_sequence[t])
    bcrp_sequence[t] = bcrp(x_sequence, t)

    S_bcrp_log[t] = f_bcrp(bcrp_sequence[t],x_sequence)
end
sum_predict_err = (prediction_error[1] + prediction_error[2])/T
#print("Forecastor avg prediction error (vector): ", prediction_error / T, "\n")
#print("Forecastor avg prediction error: ", sum_predict_err,"\n")

regret_kelly_log = zeros(Float64, length(x_sequence))
regret_ada_log = copy(regret_kelly_log)
regret_last_grad_log = copy(regret_kelly_log)
regret_softb_log = copy(regret_kelly_log)
regret_kt_log = copy(regret_kelly_log)
regret_kelly_real_log = copy(regret_kelly_log)
alg_regret_log = [regret_kelly_log, regret_ada_log, regret_last_grad_log, regret_softb_log, regret_kt_log, regret_kelly_real_log]

for t in 1:length(x_sequence) #T
    for i in 1:length(alg_regret_log)
        alg_regret_log[i][t] = wealth_sequence_log[i][t] - S_bcrp_log[t]
    end
end

plot_all_log = plot(1:T, dist_mixedc, yaxis=:log, label = "Condat- strategy/mixed action", xlabel = "Number of Epochs", ylabel = "D(r,S)(Log Scale)")
plot_all_log = plot!(1:T, dist_condat, yaxis=:log, label = "Condat- action") #marker=:auto
#plot_all_log = plot!(1:T, dist_michelot, yaxis=:log, label = "Michelot") #marker=:auto yaxis前  
one_over_sqrt_T = [1.0 / sqrt(t) for t in 1:T] 
plot_all_log = plot!(1:T, one_over_sqrt_T, yaxis=:log, label = "1/sqrt(T)") #convergence rate
savefig(plot_all_log, "approachability.png")


threshold = 0.005 * T
colors = [empirical_counts[i] < threshold ? :blue : "#db3f3d" for i in 1:length(empirical_counts)]

for i in 1:length(empirical_counts)
    if empirical_counts[i] != 0
        empirical_frequencies[i] /= empirical_counts[i]
    end
end

plot_distribution = scatter([P[i][1] for i in 1:length(P)], [empirical_frequencies[i] for i in 1:length(empirical_frequencies)], markersize = [log(empirical_counts[i]) for i in 1:length(empirical_counts)], markercolor = colors, fontfamily = "Times New Roman", xlabel = "predictions", ylabel = "empirical frequencies", seriestype = :scatter, label="red dots: prediction counts >= 0.005T")
plot_distribution = plot!([P[i][1] for i in 1:length(P)], [P[i][1] for i in 1:length(P)])
savefig(plot_distribution, "distribution.png")

start = round(Int, 1/4 * T)
d = length(returns[1]) # # of targets
plot_regrets_log = plot(1:T, regret_kelly_log, label="Kelly w calibrated mkt prediction", xlabel="Investment period", ylabel="Regrets")
plot_regrets_log = plot!(1:T, regret_kelly_real_log, label="real Kelly")
plot_regrets_log = plot!(1:T, regret_softb_log, label="Soft Bayes")
plot_regrets_log = plot!(1:T, regret_ada_log, label = "ADA-LB-FTRL")
plot_regrets_log = plot!(1:T, regret_last_grad_log, label = "Last gradient optimistic LB-FTRL")
if returns == ([1.0, 0.0], [0.0, 1.0])
    plot_regrets_log = plot!(1:T, regret_kt_log, label = "KT mixture")
end
plot_regrets_log = plot!(legend=:topleft) #top, bottom
savefig(plot_regrets_log, "regrets_kelly_log.png")
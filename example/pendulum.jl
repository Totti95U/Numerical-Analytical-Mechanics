include("../src/main.jl")

@variables t
n = 1
@variables (q(t))[1:n]
D = Differential(t)
q_dot = D.(q)

@parameters g

@variables (x(t))[1:2]
q_to_x = [
    sin(q[1]),
    cos(q[1]),
]

T = (1//2) * q_dot[1]^2
V = -g * cos(q[1])
Lexp = T - V # |> expand_derivatives |> simplify
lag = Lagrangian(t, q, q_dot, Lexp, q_to_x)

ham = hamiltonian(lag)
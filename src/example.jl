include("main.jl")

@variables t
n = 2
@variables (q(t))[1:n]
D = Differential(t)
q_dot = D.(q)

@parameters m, r

@variables (x(t))[1:4]
q_to_x = Dict{Num, Num}()
q_to_x[x[1]] = sin(q[1])
q_to_x[x[2]] = cos(q[1])
q_to_x[x[3]] = sin(q[1]) + r * sin(q[2])
q_to_x[x[4]] = cos(q[1]) + r * cos(q[2])

T = (1//2) * q_dot[1]^2 + (1//2) * m * (q_dot[1]^2 + r^2 * q_dot[2]^2 + 2 * r * cos(q[1] * q[2]) * q_dot[1] * q_dot[2])
V = -9.8 * cos(q[1]) - m * 9.8 * (cos(q[1]) + r * cos(q[2]))
Lexp = T - V
lag = Lagrangian(t, q, q_dot, Lexp, q_to_x)
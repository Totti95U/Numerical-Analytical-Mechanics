include("../src/main.jl")

@variables t
n = 2
@variables (q(t))[1:n]
D = Differential(t)
q_dot = D.(q)

@parameters m, g, r

@variables (x(t))[1:4]
q_to_x = [
    sin(q[1]),
    cos(q[1]),
    sin(q[1]) + r * sin(q[2]),
    cos(q[1]) + r * cos(q[2]),
]

T = (1//2) * q_dot[1]^2 + (1//2) * m * (q_dot[1]^2 + r^2 * q_dot[2]^2 + 2 * r * cos(q[1] - q[2]) * q_dot[1] * q_dot[2])
V = -g * cos(q[1]) - m * g * (cos(q[1]) + r * cos(q[2]))
Lexp = T - V # |> expand_derivatives |> simplify
lag = Lagrangian(t, q, q_dot, Lexp, q_to_x)

ham = hamiltonian(lag)

# parameters and initial conditions
qp0 = [pi, pi - 1e-6, 0.0, 0.0]
param = Dict{Num, Float64}()
param[g] = 9.8; param[m] = 1.0; param[r] = 1.0
tspan = (0.0, 10.0)
dt = 1e-3

# solve the Hamiltonian system
sol_t, sol_q, sol_p = solve(ham, qp0, tspan; param=param, dt=dt);

using Plots
# plot xy trajectory
x = [in_cartesian(ham, sol_q[:, i], param=param) for i in 1:length(sol_t)]
x = hcat(x...)

lim_radius = param[r] + 1.5
anim = @animate for i in 1:30:length(sol_t)
    plot(x[3, 1:i], -x[4, 1:i], xlim=(-lim_radius, lim_radius), ylim=(-lim_radius, lim_radius), aspect_ratio=1, legend=false)
    # draw the rods
    plot!([0, x[1, i]], [0, -x[2, i]], color=:black)
    plot!([x[1, i], x[3, i]], [-x[2, i], -x[4, i]], color=:black)
    # draw the masses
    scatter!([x[1, i]], -[x[2, i]], color=:red)
    scatter!([x[3, i]], -[x[4, i]], color=:blue)
end
gif(anim, "example/double-pendulum.gif", fps=30)
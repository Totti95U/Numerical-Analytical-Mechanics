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

# parameters and initial conditions
qp0 = [pi * 3/4, 0.0]
param = Dict{Num, Float64}()
param[g] = 9.8
tspan = (0.0, 10.0)
dt = 1e-3

# solve the Hamiltonian system
sol_t, sol_q, sol_p = solve(ham, qp0, tspan; param=param, dt=dt);

using Plots
# plot xy trajectory with velocity arrows
xv = [vcat(in_cartesian(ham, sol_q[:, i], sol_p[:, i], param=param)) for i in 1:length(sol_t)]
xv = hcat(xv...)

anim = @animate for i in 1:30:length(sol_t)
    plot(xv[1, 1:i], -xv[2, 1:i], xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect_ratio=1, legend=false)
    # draw the rod
    plot!([0, xv[1, i]], [0, -xv[2, i]], color=:black)
    # draw the mass
    scatter!([xv[1, i]], -[xv[2, i]], color=:red)
    # draw velocity arrow
    quiver!([xv[1, i]], -[xv[2, i]], quiver=(0.1 * [xv[3, i]], 0.1 * [-xv[4, i]]), color=:blue)
end
gif(anim, "example/pendulum.gif", fps=30)
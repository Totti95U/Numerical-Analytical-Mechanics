# 解析力学を扱うためのモジュール

using ModelingToolkit
using Symbolics: derivative, solve_for

"""
    Lagrangian(t, q, q_dot, L, q_to_x=Vector{Num}())

ラグランジアンを表す構造体。
"""
struct Lagrangian
    t::Num
    q::Symbolics.Arr{Num, 1}
    q_dot::Symbolics.Arr{Num, 1}
    L::Num
    q_to_x::Vector{Num} # 一般化座標から直交座標への変換関数を格納するベクトル

    function Lagrangian(t::Num, q::Symbolics.Arr{Num, 1}, q_dot::Symbolics.Arr{Num, 1}, L::Num, q_to_x::Vector{Num}=Vector{Num}())
        new(t, q, q_dot, simplify(L), simplify.(q_to_x))
    end
end

"""
    Hamiltonian(t, q, p, H, p_to_qdot=Vector{Num}())

ハミルトニアンを表す構造体。
"""
struct Hamiltonian
    t::Num
    q::Symbolics.Arr{Num, 1}
    p::Symbolics.Arr{Num, 1}
    H::Num
    q_to_x::Vector{Num} # 一般化座標から直交座標への変換関数を格納するベクトル
    p_to_qdot::Vector{Num} # 一般化運動量 p と速度 ̇q の関係関数を格納するベクトル

    function Hamiltonian(t::Num, q::Symbolics.Arr{Num, 1}, p::Symbolics.Arr{Num, 1}, H::Num, q_to_x::Vector{Num}=Vector{Num}(), p_to_qdot::Vector{Num}=Vector{Num}())
        new(t, q, p, simplify(H), simplify.(q_to_x), simplify.(p_to_qdot))
    end
end

function hamiltonian(lag::Lagrangian)
    t = lag.t
    q = lag.q
    q_dot = lag.q_dot
    L = lag.L
    q_to_x = lag.q_to_x

    n = length(q)
    @variables (p(t))[1:n]

    p_expr = [expand_derivatives(derivative(L, q_dot[i])) |> simplify for i in 1:n]

    # ̇q を (q, p) で解く
    eqs = [p[i] ~ p_expr[i] for i in 1:n]
    sol = solve_for(eqs, q_dot)

    # solve_for が失敗する場合は線形系として解く (T が2次形式の場合など)
    if isempty(sol)
        M = [expand_derivatives(derivative(p_expr[i], q_dot[j])) |> simplify for i in 1:n, j in 1:n]
        try
            sol = inv(M) * collect(p)
        catch e
            throw(ErrorException("Legendre変換に失敗しました。ラグランジアンが速度に退化しているか、拘束条件が適切設定されていない可能性があります。"))
        end
    end

    q_dot_to_p_dict = Dict(q_dot[i] => e for (i, e) in enumerate(sol))
    H = sum(p[i] * q_dot[i] for i in 1:n) - L
    H = substitute(H, q_dot_to_p_dict) # |> expand_derivatives |> simplify

    p_to_qdot = sol .|> Num
    # p_to_qdot = simplify(p_to_qdot)

    return Hamiltonian(t, q, p, H, q_to_x, p_to_qdot)
end

# 解析力学を扱うためのモジュール

using ModelingToolkit

"""
    Lagrangian(t, q, q_dot, L, q_to_x=nothing)

ラグランジアンを表す構造体。
"""
struct Lagrangian
    t::Num
    q::Symbolics.Arr{Num, 1}
    q_dot::Symbolics.Arr{Num, 1}
    L::Num
    q_to_x::Dict{Num, Num}  # 一般化座標から直交座標への変換辞書

    function Lagrangian(t::Num, q::Symbolics.Arr{Num, 1}, q_dot::Symbolics.Arr{Num, 1}, L::Num, q_to_x::Dict{Num, Num}=Dict{Num, Num}())
        new(t, q, q_dot, L, q_to_x)
    end
end

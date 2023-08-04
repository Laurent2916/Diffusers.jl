import NNlib: sigmoid

"""
Sigmoid beta schedule.

## Input
  * `T::Int`: number of timesteps
  * `β₁::Real=1.0f-4`: initial value of β
  * `β₋₁::Real=2.0f-2`: final value of β

## Output
  * `β::Vector{Real}`: βₜ values at each timestep t

## References
  * [[2203.02923] GeoDiff: a Geometric Diffusion Model for Molecular Conformation Generation](https://arxiv.org/abs/2203.02923)
  * [github.com:MinkaiXu/GeoDiff](https://github.com/MinkaiXu/GeoDiff/blob/ea0ca48045a2f7abfccd7f0df449e45eb6eae638/models/epsnet/diffusion.py#L57)
"""
function sigmoid_beta_schedule(T::Integer, β₁::Real=1.0f-4, β₋₁::Real=2.0f-2)
  x = range(start=-6, stop=6, length=T)
  return sigmoid(x) .* (β₋₁ - β₁) .+ β₁
end

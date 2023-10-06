import NNlib: sigmoid

"""
Sigmoid beta schedule.

```math
\\beta_t = \\sigma \\left( 12 \\frac{t - 1}{T - 1} - 6 \\right) ( \\beta_{-1} - \\beta_1 ) + \\beta_1
```

## Input
  * `T::Int`: number of timesteps
  * `β₁::Real=1.0f-4`: initial value of β
  * `β₋₁::Real=2.0f-2`: final value of β

## Output
  * `β::Vector{Real}`: βₜ values at each timestep t

## References
  * [xu2022geodiff; GeoDiff: a Geometric Diffusion Model for Molecular Conformation Generation](@cite)
  * [github.com:MinkaiXu/GeoDiff/models/epsnet/diffusion.py](https://github.com/MinkaiXu/GeoDiff/blob/ea0ca48045a2f7abfccd7f0df449e45eb6eae638/models/epsnet/diffusion.py#L57)
"""
function sigmoid_beta_schedule(T::Integer, β₁::Real=1.0f-4, β₋₁::Real=2.0f-2)
  x = range(start=-6, stop=6, length=T)
  return sigmoid(x) .* (β₋₁ - β₁) .+ β₁
end

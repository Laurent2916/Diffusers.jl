"""
Scaled linear beta schedule.

```math
\\beta_t = \\left( \\sqrt{\\beta_1} + \\frac{t - 1}{T - 1} \\left( \\sqrt{\\beta_{-1}} - \\sqrt{\\beta_1} \\right) \\right)^2
```

## Input
  * `T::Int`: number of timesteps
  * `β₁::Real=1.0f-4`: initial value of β
  * `β₋₁::Real=2.0f-2`: final value of β

## Output
  * `β::Vector{Real}`: βₜ values at each timestep t

## References
  * [ho2020denoising; Denoising Diffusion Probabilistic Models](@cite)
"""
function scaled_linear_beta_schedule(T::Integer, β₁::Real=1.0f-4, β₋₁::Real=2.0f-2)
  return range(start=√β₁, stop=√β₋₁, length=T) .^ 2
end

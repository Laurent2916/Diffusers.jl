"""
Linear beta schedule.

```math
\\beta_t = \\beta_1 + \\frac{t - 1}{T - 1} (\\beta_{-1} - \\beta_1)
```

## Input
  * `T::Integer`: number of timesteps
  * `β₁::Real=1.0f-4`: initial (t=1) value of β
  * `β₋₁::Real=2.0f-2`: final (t=T) value of β

## Output
  * `β::Vector{Real}`: βₜ values at each timestep t

## References
  * [ho2020denoising; Denoising Diffusion Probabilistic Models](@cite)
"""
function linear_beta_schedule(T::Integer, β₁::Real=1.0f-4, β₋₁::Real=2.0f-2)
  return range(start=β₁, stop=β₋₁, length=T)
end

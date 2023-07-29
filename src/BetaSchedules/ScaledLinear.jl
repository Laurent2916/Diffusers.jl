"""
Scaled linear beta schedule.

## Input
  * `T::Int`: number of timesteps
  * `β₁::Real=0.0001f0`: initial value of β
  * `β₋₁::Real=0.02f0`: final value of β

## Output
  * `β::Vector{Real}`: βₜ values at each timestep t

## References
  * [[2006.11239] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
"""
function scaled_linear_beta_schedule(T::Integer, β₁::Real=0.0001f0, β₋₁::Real=0.02f0)
  return range(start=β₁^0.5, stop=β₋₁^0.5, length=T) .^ 2
end

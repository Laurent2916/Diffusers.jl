"""
Linear beta schedule.

## Input
  * `T::Integer`: number of timesteps
  * `β₁::Real=0.0001f0`: initial (t=1) value of β
  * `β₋₁::Real=0.02f0`: final (t=T) value of β

## Output
  * `β::Vector{Real}`: βₜ values at each timestep t

## References
  * [[2006.11239] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
"""
function linear_beta_schedule(T::Integer, β₁::Real=0.0001f0, β₋₁::Real=0.02f0)
  return range(start=β₁, stop=β₋₁, length=T)
end

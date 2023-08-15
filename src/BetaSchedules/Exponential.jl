"""
Exponential beta schedule.

## Input
  * `T::Int`: number of timesteps
  * `βₘₐₓ::Real=0.999f0`: maximum value of β

## Output
  * `β::Vector{Real}`: βₜ values at each timestep t

## References
"""
function exponential_beta_schedule(T::Integer, βₘₐₓ::Real=0.999f0)
  α̅(t) = exp(-12 * t / T)

  β = Vector{Real}(undef, T)
  for t in 1:T
    αₜ = α̅(t) / α̅(t - 1)

    βₜ = 1 - αₜ
    βₜ = min(βₘₐₓ, βₜ)

    β[t] = βₜ
  end

  return β
end

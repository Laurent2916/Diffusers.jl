abstract type Scheduler end

"""
Add noise to clean data using the forward diffusion process.

## Input
  * `scheduler::Scheduler`: scheduler to use
  * `x₀::AbstractArray`: clean data to add noise to
  * `ϵ::AbstractArray`: noise to add to clean data
  * `t::AbstractArray`: timesteps used to weight the noise

## Output
  * `xₜ::AbstractArray`: noisy data at the given timesteps
"""
function add_noise(
  scheduler::Scheduler,
  x₀::AbstractArray,
  ϵ::AbstractArray,
  t::AbstractArray,
)
  ⎷α̅ₜ = scheduler.⎷α̅[t]
  ⎷β̅ₜ = scheduler.⎷β̅[t]

  # noisify clean data
  # arxiv:2006.11239 Eq. 4
  xₜ = ⎷α̅ₜ' .* x₀ + ⎷β̅ₜ' .* ϵ

  return xₜ
end

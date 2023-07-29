abstract type Scheduler end

"""
Add noise to clean data using the forward diffusion process.

cf. [[2006.11239] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Eq. 4)

## Input
  * `scheduler::Scheduler`: scheduler to use
  * `clean_data::AbstractArray`: clean data to add noise to
  * `noise::AbstractArray`: noise to add to clean data
  * `timesteps::AbstractArray`: timesteps used to weight the noise

## Output
  * `noisy_data::AbstractArray`: noisy data at the given timesteps
"""
function add_noise(
  scheduler::Scheduler,
  x₀::AbstractArray,
  ϵ::AbstractArray,
  t::AbstractArray,
)
  ⎷α̅ₜ = scheduler.⎷α̅[t]
  ⎷β̅ₜ = scheduler.⎷β̅[t]

  xₜ = ⎷α̅ₜ' .* x₀ + ⎷β̅ₜ' .* ϵ

  return xₜ
end

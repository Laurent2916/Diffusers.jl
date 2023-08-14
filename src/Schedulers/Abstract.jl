"""
Abstract type for schedulers.

"""
abstract type Scheduler end

@enum VarianceType FIXED_SMALL FIXED_SMALL_LOG FIXED_LARGE FIXED_LARGE_LOG LEARNED
@enum PredictionType EPSILON SAMPLE VELOCITY

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
function forward(
  scheduler::Scheduler,
  x₀::AbstractArray,
  ϵ::AbstractArray,
  t::AbstractArray,
) end

"""
Remove noise from model output using the backward diffusion process.

## Input
  * `scheduler::Scheduler`: scheduler to use
  * `xₜ::AbstractArray`: sample to be denoised
  * `ϵᵧ::AbstractArray`: predicted noise to remove
  * `t::AbstractArray`: timestep t of `xₜ`

## Output
  * `xₜ₋₁::AbstractArray`: denoised sample at t=t-1
  * `x̂₀::AbstractArray`: denoised sample at t=0
"""
function reverse(
  scheduler::Scheduler,
  xₜ::AbstractArray,
  ϵᵧ::AbstractArray,
  t::AbstractArray,
) end

"""
Compute the velocity of the diffusion process.

## Input
  * `scheduler::Scheduler`: scheduler to use
  * `x₀::AbstractArray`: clean data to add noise to
  * `ϵ::AbstractArray`: noise to add to clean data
  * `t::AbstractArray`: timesteps used to weight the noise

## Output
  * `vₜ::AbstractArray`: velocity at the given timesteps

## References
  * [[2202.00512] Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512) (Ann. D)
"""
function get_velocity(
  scheduler::Scheduler,
  x₀::AbstractArray,
  ϵ::AbstractArray,
  t::AbstractArray,
) end

abstract type Scheduler end

"""
Add noise to clean data using the forward diffusion process.

## Input
  * scheduler (`Scheduler`): scheduler object.
  * clean_data (`AbstractArray`): clean data to add noise to.
  * noise (`AbstractArray`): noise to add to clean data.
  * timesteps (`AbstractArray`): timesteps used to weight the noise.

## Output
  * noisy_data (`AbstractArray`): noisy data at the given timesteps.
"""
function add_noise(
  scheduler::Scheduler,
  clean_data::AbstractArray,
  noise::AbstractArray,
  timesteps::AbstractArray,
)
  sqrt_α_cumprod_t = scheduler.sqrt_α_cumprods[timesteps]
  sqrt_one_minus_α_cumprod_t = scheduler.sqrt_one_minus_α_cumprods[timesteps]

  noisy_data = sqrt_α_cumprod_t' .* clean_data + sqrt_one_minus_α_cumprod_t' .* noise

  return noisy_data
end

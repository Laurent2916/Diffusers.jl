abstract type Scheduler end

function add_noise(
  scheduler::Scheduler,
  original_samples::AbstractArray,
  noise::AbstractArray,
  timesteps::AbstractArray,
)
  alphas_cumprod = scheduler.α_cumprods[timesteps]
  sqrt_alpha_prod = sqrt.(alphas_cumprod)
  sqrt_one_minus_alpha_prod = sqrt.(1 .- alphas_cumprod)

  sqrt_alpha_prod .* original_samples .+ sqrt_one_minus_alpha_prod .* noise
end

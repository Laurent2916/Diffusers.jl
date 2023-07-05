abstract type Scheduler end

function add_noise(
  scheduler::Scheduler,
  clean_data::AbstractArray,
  noise::AbstractArray,
  timesteps::AbstractArray,
)
  """
  Add noise to clean data using the forward diffusion process.

  Args:
    scheduler (`Scheduler`): scheduler object.
    clean_data (`AbstractArray`): clean data to add noise to.
    noise (`AbstractArray`): noise to add to clean data.
    timesteps (`AbstractArray`): timesteps used to weight the noise.

  Returns:
    `AbstractArray`: noisy data at the given timesteps.
  """
  sqrt_α_cumprod_t = scheduler.sqrt_α_cumprods[timesteps]
  sqrt_one_minus_α_cumprod_t = scheduler.sqrt_one_minus_α_cumprods[timesteps]

  sqrt_α_cumprod_t .* clean_data .+ sqrt_one_minus_α_cumprod_t .* noise
end

function step(
  scheduler::Scheduler,
  sample::AbstractArray,
  model_output::AbstractArray,
  timestep::Int,
)
  """
  Remove noise from model output using the backward diffusion process.

  Args:
    scheduler (`Scheduler`): scheduler object.
    sample (`AbstractArray`): sample to remove noise from, i.e. model_input.
    model_output (`AbstractArray`): predicted noise from the model.
    timestep (`Int`): timestep to remove noise from.

  Returns:
    `AbstractArray`: denoised model output at the given timestep.
  """

  # 1. compute alphas, betas
  α_cumprod_t = scheduler.α_cumprods[timestep]
  α_cumprod_t_prev = scheduler.α_cumprods[timestep - 1]
  β_cumprod_t = 1 - α_cumprod_t
  β_cumprod_t_prev = 1 - α_cumprod_t_prev
  current_α_t = α_cumprod_t / α_cumprod_t_prev
  current_β_t = 1 - current_α_t

  # 2. compute predicted original sample from predicted noise also called
  # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
  # epsilon prediction type
  x_0 = (noise - √β_cumprod_t_prev * model_output) / √α_cumprod_t_prev

  # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
  # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
  pred_original_sample_coeff = (√α_cumprod_t_prev * current_β_t) / β_cumprod_t
  current_sample_coeff = √current_α_t * β_cumprod_t_prev / β_cumprod_t

  # 5. Compute predicted previous sample µ_t
  # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
  pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

  # 6. Add noise
  variance = √scheduler.βs[timestep] * randn(size(model_output))
  pred_prev_sample = pred_prev_sample + variance

  return pred_prev_sample, pred_original_sample
include("Schedulers.jl")

"""
Denoising Diffusion Probabilistic Models (DDPM) scheduler.

cf. [[2006.11239] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006).
"""
struct DDPM{V<:AbstractVector} <: Scheduler
  # number of diffusion steps used to train the model.
  num_train_timesteps::Int

  # the betas to use for the diffusion steps
  βs::V
  αs::V
  α_cumprods::V
  α_cumprod_prevs::V

  sqrt_α_cumprods::V
  sqrt_one_minus_α_cumprods::V
end

function DDPM(V::DataType, βs::AbstractVector)
  αs = 1 .- βs
  α_cumprods = cumprod(αs)
  α_cumprod_prevs = [1, (α_cumprods[1:end-1])...]

  sqrt_α_cumprods = sqrt.(α_cumprods)
  sqrt_one_minus_α_cumprods = sqrt.(1 .- α_cumprods)

  DDPM{V}(
    length(βs),
    βs,
    αs,
    α_cumprods,
    α_cumprod_prevs,
    sqrt_α_cumprods,
    sqrt_one_minus_α_cumprods,
  )
end

"""
Remove noise from model output using the backward diffusion process.

## Input
  * scheduler (`DDPM`): scheduler to use
  * sample (`AbstractArray`): sample to remove noise from, i.e. model_input
  * model_output (`AbstractArray`): predicted noise from the model
  * timesteps (`AbstractArray`): timesteps to remove noise from

## Output
  * pred_prev_sample (`AbstractArray`): denoised sample at t=t-1
  * x_0_pred (`AbstractArray`): denoised sample at t=0
"""
function step(
  scheduler::DDPM,
  sample::AbstractArray,
  model_output::AbstractArray,
  timesteps::AbstractArray,
)
  # 1. compute alphas, betas
  α_cumprod_t = scheduler.α_cumprods[timesteps]
  α_cumprod_t_prev = scheduler.α_cumprods[timesteps.-1]
  β_cumprod_t = 1 .- α_cumprod_t
  β_cumprod_t_prev = 1 .- α_cumprod_t_prev
  current_α_t = α_cumprod_t ./ α_cumprod_t_prev
  current_β_t = 1 .- current_α_t

  # 2. compute predicted original sample from predicted noise also called
  # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
  # epsilon prediction type
  # print shapes of thingies
  x_0_pred = (sample - sqrt.(β_cumprod_t)' .* model_output) ./ sqrt.(α_cumprod_t)'

  # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
  # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
  pred_original_sample_coeff = (sqrt.(α_cumprod_t_prev) .* current_β_t) ./ β_cumprod_t
  current_sample_coeff = sqrt.(current_α_t) .* β_cumprod_t_prev ./ β_cumprod_t

  # 5. Compute predicted previous sample µ_t
  # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
  pred_prev_sample = pred_original_sample_coeff' .* x_0_pred + current_sample_coeff' .* sample

  # 6. Add noise
  variance = sqrt.(scheduler.βs[timesteps])' .* randn(size(model_output))
  pred_prev_sample = pred_prev_sample + variance

  return pred_prev_sample, x_0_pred
end

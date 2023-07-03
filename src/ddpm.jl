include("scheduler.jl")

"""
Denoising Diffusion Probabilistic Models (DDPM) scheduler.

https://arxiv.org/abs/2006.11239
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

function DDPM(V::DataType, beta_scheduler)
  DDPM(V, beta_scheduler)
end

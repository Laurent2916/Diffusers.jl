include("Abstract.jl")

using ShiftedArrays

function _extract(
  target::AbstractArray,
  reference::AbstractArray,
)
  new_size = tuple(
    fill(1, ndims(reference) - 1)...,
    size(target, 1)
  )
  return reshape(target, new_size)
end

"""
Denoising Diffusion Probabilistic Models (DDPM) scheduler.

## References
  * [[2006.11239] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
"""
struct DDPM{V<:AbstractVector} <: Scheduler
  T::Integer # length of markov chain

  α::V # 1 - beta
  β::V # beta variance schedule

  ⎷α::V # square root of α
  ⎷β::V # square root of β

  α̅::V # cumulative product of α
  β̅::V # 1 - α̅ (≠ cumprod(β))

  α̅₋₁::V # right-shifted α̅
  β̅₋₁::V # right-shifted β̅

  ⎷α̅::V # square root of α̅
  ⎷β̅::V # square root of β̅

  ⎷α̅₋₁::V # right-shifted ⎷α̅
  ⎷β̅₋₁::V # right-shifted ⎷β̅
end

function DDPM(β::AbstractVector)
  T = length(β)

  α = 1 .- β

  ⎷α = sqrt.(α)
  ⎷β = sqrt.(β)

  α̅ = cumprod(α)
  β̅ = 1 .- α̅

  α̅₋₁ = ShiftedArray(α̅, 1, default=1.0)
  β̅₋₁ = ShiftedArray(β̅, 1, default=0.0)

  ⎷α̅ = sqrt.(α̅)
  ⎷β̅ = sqrt.(β̅)

  ⎷α̅₋₁ = ShiftedArray(⎷α̅, 1, default=1.0)
  ⎷β̅₋₁ = ShiftedArray(⎷β̅, 1, default=0.0)

  DDPM{typeof(β)}(
    T,
    α, β,
    ⎷α, ⎷β,
    α̅, β̅,
    α̅₋₁, β̅₋₁,
    ⎷α̅, ⎷β̅,
    ⎷α̅₋₁, ⎷β̅₋₁,
  )
end

function forward(
  scheduler::DDPM,
  x₀::AbstractArray,
  ϵ::AbstractArray,
  t::AbstractArray,
)
  # retreive scheduler variables at timesteps t
  ⎷α̅ₜ = _extract(scheduler.⎷α̅[t], x₀)
  ⎷β̅ₜ = _extract(scheduler.⎷β̅[t], x₀)

  # noisify clean data
  # arxiv:2006.11239 Eq. 4
  xₜ = ⎷α̅ₜ .* x₀ + ⎷β̅ₜ .* ϵ

  return xₜ
end

function reverse(
  scheduler::DDPM,
  xₜ::AbstractArray,
  ϵᵧ::AbstractArray,
  t::AbstractArray,
  ;
  prediction_type::PredictionType=EPSILON,
  variance_type::VarianceType=FIXED_SMALL
)
  # retreive scheduler variables at timesteps t
  βₜ = _extract(scheduler.β[t], xₜ)
  β̅ₜ = _extract(scheduler.β̅[t], xₜ)
  β̅ₜ₋₁ = _extract(scheduler.β̅₋₁[t], xₜ)
  ⎷αₜ = _extract(scheduler.⎷α[t], xₜ)
  ⎷α̅ₜ₋₁ = _extract(scheduler.⎷α̅₋₁[t], xₜ)

  # compute x₀ (approximation)
  x̂₀ = get_prediction(scheduler, prediction_type, xₜ, ϵᵧ, t)

  # compute μₜ (approximation)
  # arxiv:2006.11239 Eq. 7
  # arxiv:2208.11970 Eq. 84
  λ₀ = ⎷α̅ₜ₋₁ .* βₜ ./ β̅ₜ
  λₜ = ⎷αₜ .* β̅ₜ₋₁ ./ β̅ₜ
  μ̃ₜ = λ₀ .* x̂₀ + λₜ .* xₜ

  # compute σ² (exact)
  σ²ₜ = get_variance(scheduler, variance_type, xₜ, t)

  # sample xₜ₋₁ using μₜ and σ²
  σₜ = sqrt.(σ²ₜ)
  ϵ = randn(size(ϵᵧ))
  xₜ₋₁ = μ̃ₜ + σₜ .* ϵ

  return xₜ₋₁, x̂₀
end

function get_velocity(
  scheduler::DDPM,
  x₀::AbstractArray,
  ϵ::AbstractArray,
  t::AbstractArray,
)
  ⎷α̅ₜ = _extract(scheduler.⎷α̅[t], x₀)
  ⎷β̅ₜ = _extract(scheduler.⎷β̅[t], x₀)

  vₜ = ⎷α̅ₜ .* ϵ - ⎷β̅ₜ .* x₀

  return vₜ
end

function get_prediction(
  scheduler::DDPM,
  prediction_type::PredictionType,
  xₜ::AbstractArray,
  ϵᵧ::AbstractArray,
  t::AbstractArray,
)
  ⎷α̅ₜ = _extract(scheduler.⎷α̅[t], xₜ)
  ⎷β̅ₜ = _extract(scheduler.⎷β̅[t], xₜ)

  if prediction_type == EPSILON
    # arxiv:2006.11239 Eq. 15
    # arxiv:2208.11970 Eq. 115
    x̂₀ = (xₜ - ⎷β̅ₜ .* ϵᵧ) ./ ⎷α̅ₜ
  elseif prediction_type == SAMPLE
    # arxiv:2208.11970 Eq. 99
    x̂₀ = ϵᵧ
  elseif prediction_type == VELOCITY
    # arxiv:2202.00512 Eq. 31
    x̂₀ = ⎷α̅ₜ .* xₜ - ⎷β̅ₜ .* ϵᵧ
  else
    throw("unimplemented prediction type")
  end

  return x̂₀
end

function get_variance(
  scheduler::DDPM,
  variance_type::VarianceType,
  xₜ::AbstractArray,
  t::AbstractArray,
)
  βₜ = _extract(scheduler.β[t], xₜ)
  β̅ₜ = _extract(scheduler.β̅[t], xₜ)
  β̅ₜ₋₁ = _extract(scheduler.β̅₋₁[t], xₜ)

  if variance_type == FIXED_SMALL
    # arxiv:2006.11239 Eq. 6
    # arxiv:2208.11970 Eq. 70
    σ²ₜ = β̅ₜ₋₁ ./ β̅ₜ .* βₜ
  elseif variance_type == FIXED_SMALL_LOG
    σ²ₜ = β̅ₜ₋₁ ./ β̅ₜ .* βₜ
    σ²ₜ = log.(σ²ₜ)
  elseif variance_type == FIXED_LARGE
    σ²ₜ = βₜ
  elseif variance_type == FIXED_LARGE_LOG
    σ²ₜ = βₜ
    σ²ₜ = log.(σ²ₜ)
  else
    throw("unimplemented variance type")
  end

  return σ²ₜ
end

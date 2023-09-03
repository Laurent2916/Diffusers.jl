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
Denoising Diffusion Implicit Models (DDIM) scheduler.

## References
  * [[2010.02502] Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
"""
struct DDIM{V<:AbstractVector} <: Scheduler
  T::Integer # length of markov chain

  α::V # 1 - beta
  β::V # beta variance schedule

  ⎷α::V # square root of α
  ⎷β::V # square root of β

  α̅::V # cumulative product of α
  β̅::V # 1 - α̅ (≠ cumprod(β))

  # α̅₋₁::V # right-shifted α̅
  # β̅₋₁::V # right-shifted β̅

  ⎷α̅::V # square root of α̅
  ⎷β̅::V # square root of β̅

  # ⎷α̅₋₁::V # right-shifted ⎷α̅
  # ⎷β̅₋₁::V # right-shifted ⎷β̅
end

function DDIM(β::AbstractVector)
  T = length(β)

  α = 1 .- β

  ⎷α = sqrt.(α)
  ⎷β = sqrt.(β)

  α̅ = cumprod(α)
  β̅ = 1 .- α̅

  # α̅₋₁ = ShiftedArray(α̅, 1, default=1.0)
  # β̅₋₁ = ShiftedArray(β̅, 1, default=0.0)

  ⎷α̅ = sqrt.(α̅)
  ⎷β̅ = sqrt.(β̅)

  # ⎷α̅₋₁ = ShiftedArray(⎷α̅, 1, default=1.0)
  # ⎷β̅₋₁ = ShiftedArray(⎷β̅, 1, default=0.0)

  DDIM{typeof(β)}(
    T,
    α, β,
    ⎷α, ⎷β,
    α̅, β̅,
    # α̅₋₁, β̅₋₁,
    ⎷α̅, ⎷β̅,
    # ⎷α̅₋₁, ⎷β̅₋₁,
  )
end

function forward(
  scheduler::DDIM,
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
  scheduler::DDIM,
  xₜ::AbstractArray,
  ϵᵧ::AbstractArray,
  t::AbstractArray,
  η::Real=0.0f0,
  prediction_type::PredictionType=EPSILON,
)
  Δₜ = 1
  α̅ = _extract(scheduler.α̅, xₜ)
  α̅₋ₚ = ShiftedArray(scheduler.α̅, Δₜ, default=1.0f0)
  α̅ₜ = _extract(α̅[t], xₜ)
  α̅ₜ₋ₚ = _extract(α̅₋ₚ[t], xₜ)
  β̅ = _extract(scheduler.β̅, xₜ)
  β̅₋ₚ = ShiftedArray(scheduler.β̅, Δₜ, default=0.0f0)
  β̅ₜ = _extract(β̅[t], xₜ)
  β̅ₜ₋ₚ = _extract(β̅₋ₚ[t], xₜ)

  println("α̅ₜ = ", α̅ₜ)
  println("α̅ₜ₋ₚ = ", α̅ₜ₋ₚ)

  # compute x₀ (approximation)
  x̂₀ = get_prediction(scheduler, prediction_type, xₜ, ϵᵧ, t)

  # clip between -1 and 1
  x̂₀ = clamp.(x̂₀, -3, 3)

  # compute σ (exact)
  σ²ₜ = (β̅ₜ₋ₚ ./ β̅ₜ) .* (1 .- α̅ₜ ./ α̅ₜ₋ₚ)
  println("β̅ₜ₋ₚ = ", β̅ₜ₋ₚ)
  println("β̅ₜ = ", β̅ₜ)
  println("α̅ₜ = ", α̅ₜ)
  println("α̅ₜ₋ₚ = ", α̅ₜ₋ₚ)
  println("σ²ₜ = ", σ²ₜ)
  σₜ = η * sqrt.(σ²ₜ)

  # compute direction
  Δₓ = sqrt.(1 .- α̅ₜ₋ₚ .- σₜ^2) .* ϵᵧ

  # sample xₜ₋ₚ
  ϵ = randn(Float32, size(xₜ))
  xₜ₋ₚ = sqrt.(α̅ₜ₋ₚ) .* x̂₀ .+ Δₓ .+ σₜ .* ϵ

  return xₜ₋ₚ, x̂₀
end

function get_velocity(
  scheduler::DDIM,
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
  scheduler::DDIM,
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
    # elseif prediction_type == SAMPLE
    #   # arxiv:2208.11970 Eq. 99
    #   x̂₀ = ϵᵧ
    # elseif prediction_type == VELOCITY
    #   # arxiv:2202.00512 Eq. 31
    #   x̂₀ = ⎷α̅ₜ .* xₜ - ⎷β̅ₜ .* ϵᵧ
  else
    throw("unimplemented prediction type")
  end

  return x̂₀
end

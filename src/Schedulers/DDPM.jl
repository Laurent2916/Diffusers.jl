include("Abstract.jl")

"""
Denoising Diffusion Probabilistic Models (DDPM) scheduler.

## References
  * [[2006.11239] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
"""
struct DDPM{V<:AbstractVector} <: Scheduler
  T::Integer # length of markov chain

  β::V # beta variance schedule
  α::V # 1 - beta

  ⎷α::V # square root of α
  ⎷β::V # square root of β

  α̅::V # cumulative product of α
  β̅::V # 1 - α̅ (≠ cumprod(β))

  α̅₋₁::V # right-shifted α̅
  β̅₋₁::V # 1 - α̅₋₁

  ⎷α̅::V # square root of α̅
  ⎷β̅::V # square root of β̅

  ⎷α̅₋₁::V # square root of α̅₋₁
  ⎷β̅₋₁::V # square root of β̅₋₁
end

function DDPM(β::AbstractVector)
  T = length(β)

  α = 1 .- β

  ⎷α = sqrt.(α)
  ⎷β = sqrt.(β)

  α̅ = cumprod(α)
  β̅ = 1 .- α̅

  α̅₋₁ = [1, (α̅[1:end-1])...]
  β̅₋₁ = 1 .- α̅₋₁

  ⎷α̅ = sqrt.(α̅)
  ⎷β̅ = sqrt.(β̅)

  ⎷α̅₋₁ = sqrt.(α̅₋₁)
  ⎷β̅₋₁ = sqrt.(β̅₋₁)

  DDPM{typeof(β)}(
    T,
    β,
    α,
    ⎷α,
    ⎷β,
    α̅,
    β̅,
    α̅₋₁,
    β̅₋₁,
    ⎷α̅,
    ⎷β̅,
    ⎷α̅₋₁,
    ⎷β̅₋₁,
  )
end

function add_noise(
  scheduler::DDPM,
  x₀::AbstractArray,
  ϵ::AbstractArray,
  t::AbstractArray,
)
  # retreive scheduler variables at timesteps t
  reshape_size = tuple(
    fill(1, ndims(x₀) - 1)...,
    size(t, 1)
  )
  ⎷α̅ₜ = reshape(scheduler.⎷α̅[t], reshape_size)
  ⎷β̅ₜ = reshape(scheduler.⎷β̅[t], reshape_size)

  # noisify clean data
  # arxiv:2006.11239 Eq. 4
  xₜ = ⎷α̅ₜ .* x₀ + ⎷β̅ₜ .* ϵ

  return xₜ
end

function step(
  scheduler::DDPM,
  xₜ::AbstractArray,
  ϵᵧ::AbstractArray,
  t::AbstractArray,
)
  # retreive scheduler variables at timesteps t
  reshape_size = tuple(
    fill(1, ndims(xₜ) - 1)...,
    size(t, 1)
  )
  βₜ = reshape(scheduler.β[t], reshape_size)
  β̅ₜ = reshape(scheduler.β̅[t], reshape_size)
  β̅ₜ₋₁ = reshape(scheduler.β̅₋₁[t], reshape_size)
  ⎷αₜ = reshape(scheduler.⎷α[t], reshape_size)
  ⎷α̅ₜ = reshape(scheduler.⎷α̅[t], reshape_size)
  ⎷α̅ₜ₋₁ = reshape(scheduler.⎷α̅₋₁[t], reshape_size)
  ⎷β̅ₜ = reshape(scheduler.⎷β̅[t], reshape_size)

  # compute predicted previous sample x̂₀
  # arxiv:2006.11239 Eq. 15
  # arxiv:2208.11970 Eq. 115
  x̂₀ = (xₜ - ⎷β̅ₜ .* ϵᵧ) ./ ⎷α̅ₜ

  # compute predicted previous sample μ̃ₜ
  # arxiv:2006.11239 Eq. 7
  # arxiv:2208.11970 Eq. 84
  λ₀ = ⎷α̅ₜ₋₁ .* βₜ ./ β̅ₜ
  λₜ = ⎷αₜ .* β̅ₜ₋₁ ./ β̅ₜ  # TODO: this could be stored in the scheduler
  μ̃ₜ = λ₀ .* x̂₀ + λₜ .* xₜ

  # sample predicted previous sample xₜ₋₁
  # arxiv:2006.11239 Eq. 6
  # arxiv:2208.11970 Eq. 70
  σₜ = β̅ₜ₋₁ ./ β̅ₜ .* βₜ # TODO: this could be stored in the scheduler
  xₜ₋₁ = μ̃ₜ + σₜ .* randn(size(ϵᵧ))

  return xₜ₋₁, x̂₀
end

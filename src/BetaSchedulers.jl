import NNlib: sigmoid

"""
Linear beta schedule.

## Input
  * `T::Integer`: number of timesteps
  * `β₁::Real=0.0001f0`: initial (t=1) value of β
  * `β₋₁::Real=0.02f0`: final (t=T) value of β

## Output
  * `β::Vector{Real}`: βₜ values at each timestep t

## References
  * [[2006.11239] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
"""
function linear_beta_schedule(T::Integer, β₁::Real=0.0001f0, β₋₁::Real=0.02f0)
  return range(start=β₁, stop=β₋₁, length=T)
end

"""
Scaled linear beta schedule.

## Input
  * `T::Int`: number of timesteps
  * `β₁::Real=0.0001f0`: initial value of β
  * `β₋₁::Real=0.02f0`: final value of β

## Output
  * `β::Vector{Real}`: βₜ values at each timestep t

## References
  * [[2006.11239] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
"""
function scaled_linear_beta_schedule(T::Integer, β₁::Real=0.0001f0, β₋₁::Real=0.02f0)
  return range(start=β₁^0.5, stop=β₋₁^0.5, length=T) .^ 2
end

"""
Sigmoid beta schedule.

## Input
  * `T::Int`: number of timesteps
  * `β₁::Real=0.0001f0`: initial value of β
  * `β₋₁::Real=0.02f0`: final value of β

## Output
  * `β::Vector{Real}`: βₜ values at each timestep t

## References
  * [[2203.02923] GeoDiff: a Geometric Diffusion Model for Molecular Conformation Generation](https://arxiv.org/abs/2203.02923)
  * [github.com:MinkaiXu/GeoDiff](https://github.com/MinkaiXu/GeoDiff/blob/ea0ca48045a2f7abfccd7f0df449e45eb6eae638/models/epsnet/diffusion.py#L57)
"""
function sigmoid_beta_schedule(T::Integer, β₁::Real=0.0001f0, β₋₁::Real=0.02f0)
  x = range(start=-6, stop=6, length=T)
  return sigmoid(x) .* (β₋₁ - β₁) .+ β₁
end

"""
Cosine beta schedule.

## Input
  * `T::Int`: number of timesteps
  * `βₘₐₓ::Real=0.999f0`: maximum value of β
  * `ϵ::Real=1e-3f0`: small value used to avoid division by zero

## Output
  * `β::Vector{Real}`: βₜ values at each timestep t

## References
  * [[2102.09672] Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
  * [github:openai/improved-diffusion](https://github.com/openai/improved-diffusion/blob/783b6740edb79fdb7d063250db2c51cc9545dcd1/improved_diffusion/gaussian_diffusion.py#L36)
"""
function cosine_beta_schedule(T::Integer, βₘₐₓ::Real=0.999f0, ϵ::Real=0.001f0)
  α̅(t) = cos((t / T + ϵ) / (1 + ϵ) * π / 2)^2

  β = Vector{Real}(undef, T)
  for t in 1:T
    αₜ = α̅(t) / α̅(t - 1)

    βₜ = 1 - αₜ
    βₜ = min(βₘₐₓ, βₜ)

    β[t] = βₜ
  end

  return β
end

"""
Rescale betas to have zero terminal Signal to Noise Ratio (SNR).

## Input
  * `β::AbstractArray`: βₜ values at each timestep t

## Output
  * `β::Vector{Real}`: rescaled βₜ values at each timestep t

## References
  * [[2305.08891] Rescaling Diffusion Models](https://arxiv.org/abs/2305.08891) (Alg. 1)
"""
function rescale_zero_terminal_snr(β::AbstractArray)
  # convert β to ⎷α̅
  α = 1 .- β
  α̅ = cumprod(α)
  ⎷α̅ = sqrt.(α̅)

  # store old extrema values
  ⎷α̅₁ = ⎷α̅[1]
  ⎷α̅₋₁ = ⎷α̅[end]

  # shift last timestep to zero
  ⎷α̅ .-= ⎷α̅₋₁

  # scale so that first timestep reaches old values
  ⎷α̅ *= ⎷α̅₁ / (⎷α̅₁ - ⎷α̅₋₁)

  # convert back ⎷α̅ to β
  α̅ = ⎷α̅ .^ 2
  α = α̅[2:end] ./ α̅[1:end-1]
  α = vcat(α̅[1], α)
  β = 1 .- α

  return β
end

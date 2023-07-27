import NNlib: sigmoid

"""
Linear beta schedule.

cf. [[2006.11239] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

## Input
  * T (`Int`): number of timesteps
  * β_1 (`Real := 0.0001f0`): initial value of β
  * β_T (`Real := 0.02f0`): final value of β

## Output
  * β (`Vector{Real}`): β_t values at each timestep t
"""
function linear_beta_schedule(T::Int, β_1::Real=0.0001f0, β_T::Real=0.02f0)
  return range(start=β_1, stop=β_T, length=T)
end

"""
Scaled linear beta schedule.

cf. [[2006.11239] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

## Input
  * T (`Int`): number of timesteps
  * β_1 (`Real := 0.0001f0`): initial value of β
  * β_T (`Real := 0.02f0`): final value of β

## Output
  * β (`Vector{Real}`): β_t values at each timestep t
"""
function scaled_linear_beta_schedule(T::Int, β_1::Real=0.0001f0, β_T::Real=0.02f0)
  return range(start=β_1^0.5, stop=β_T^0.5, length=T) .^ 2
end

"""
Sigmoid beta schedule.

cf. [[2203.02923] GeoDiff: a Geometric Diffusion Model for Molecular Conformation Generation](https://arxiv.org/abs/2203.02923)
and [github.com:MinkaiXu/GeoDiff](https://github.com/MinkaiXu/GeoDiff/blob/ea0ca48045a2f7abfccd7f0df449e45eb6eae638/models/epsnet/diffusion.py#L57)

## Input
  * T (`Int`): number of timesteps
  * β_1 (`Real := 0.0001f0`): initial value of β
  * β_T (`Real := 0.02f0`): final value of β

## Output
  * β (`Vector{Real}`): β_t values at each timestep t
"""
function sigmoid_beta_schedule(T::Int, β_1::Real=0.0001f0, β_T::Real=0.02f0)
  x = range(start=-6, stop=6, length=T)
  return sigmoid(x) .* (β_T - β_1) .+ β_1
end

"""
Cosine beta schedule.

cf. [[2102.09672] Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)

## Input
  * T (`Int`): number of timesteps
  * β_max (`Real := 0.999f0`): maximum value of β
  * ϵ (`Real := 1e-3f0`): small value used to avoid division by zero

## Output
  * β (`Vector{Real}`): β_t values at each timestep t
"""
function cosine_beta_schedule(T::Int, β_max::Real=0.999f0, ϵ::Real=0.001f0)
  α_bar(t) = cos((t + ϵ) / (1 + ϵ) * π / 2)^2

  β = Float32[]
  for t in 1:T
    t1 = (t - 1) / T
    t2 = t / T

    β_t = 1 - α_bar(t2) / α_bar(t1)
    β_t = min(β_max, β_t)

    push!(β, β_t)
  end

  return β
end

"""
Rescale betas to have zero terminal SNR.

cf. [[2305.08891] Rescaling Diffusion Models](https://arxiv.org/abs/2305.08891) (Algorithm 1)

## Input
  * β (`AbstractArray`): β_t values at each timestep t

## Output
  * β (`Vector{Real}`): rescaled β_t values at each timestep t
"""
function rescale_zero_terminal_snr(β::AbstractArray)
  # convert β to sqrt_α_cumprods
  α = 1 .- β
  α_cumprod = cumprod(α)
  sqrt_α_cumprods = sqrt.(α_cumprod)

  # store old extrema values
  sqrt_α_cumprod_1 = sqrt_α_cumprods[1]
  sqrt_α_cumprod_T = sqrt_α_cumprods[end]

  # shift last timestep to zero
  sqrt_α_cumprods .-= sqrt_α_cumprod_T

  # scale so that first timestep reaches old values
  sqrt_α_cumprods *= sqrt_α_cumprod_1 / (sqrt_α_cumprod_1 - sqrt_α_cumprod_T)

  # convert back sqrt_α_cumprods to β
  α_cumprod = sqrt_α_cumprods .^ 2
  α = α_cumprod[2:end] ./ α_cumprod[1:end-1]
  α = vcat(α_cumprod[1], α)
  β = 1 .- α

  return β
end

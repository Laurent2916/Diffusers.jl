import NNlib: sigmoid

"""
Linear beta schedule.

cf. [[2006.11239] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

## Input
  * T (`Int`): number of timesteps
  * β_1 (`Real := 0.0001f0`): initial value of β
  * β_T (`Real := 0.02f0`): final value of β

## Output
  * βs (`Vector{Real}`): β_t values at each timestep t
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
  * βs (`Vector{Real}`): β_t values at each timestep t
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
  * βs (`Vector{Real}`): β_t values at each timestep t
"""
function sigmoid_beta_schedule(T::Int, β_1::Real=0.0001f0, β_T::Real=0.02f0)
  x = range(start=-6, stop=6, length=T)
  return sigmoid(x) * (β_T - β_1) + β_1
end

"""
Cosine beta schedule.

cf. [[2102.09672] Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)

## Input
  * T (`Int`): number of timesteps
  * β_max (`Real := 0.999f0`): maximum value of β
  * ϵ (`Real := 1e-3f0`): small value used to avoid division by zero

## Output
  * βs (`Vector{Real}`): β_t values at each timestep t
"""
function cosine_beta_schedule(T::Int, β_max::Real=0.999f0, ϵ::Real=1e-3f0)
  α_bar(t) = cos((t + ϵ) / (1 + ϵ) * π / 2)^2

  βs = Float32[]
  for t in 1:T
    t1 = (t - 1) / T
    t2 = t / T

    β_t = 1 - α_bar(t2) / α_bar(t1)
    β_t = min(β_max, β_t)

    push!(βs, β_t)
  end

  return βs
end

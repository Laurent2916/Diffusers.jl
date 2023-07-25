import NNlib: sigmoid

"""
Linear beta schedule.

cf. https://arxiv.org/abs/2006.11239
"""
function linear_beta_schedule(num_timesteps::Int, β_start=0.0001f0, β_end=0.02f0)
  range(β_start, β_end, length=num_timesteps)
end

"""
Scaled linear beta schedule.
(very specific to latent diffusion models)

cf. https://arxiv.org/abs/2006.11239
"""
function scaled_linear_beta_schedule(num_timesteps::Int, β_start=0.0001f0, β_end=0.02f0)
  range(β_start^0.5, β_end^0.5, length=num_timesteps) .^ 2
end

"""
Cosine beta schedule.

cf. https://arxiv.org/abs/2102.09672
"""
function cosine_beta_schedule(num_timesteps::Int, max_beta=0.999f0, ϵ=1e-3f0)
  α_bar(t) = cos((t + ϵ) / (1 + ϵ) * π / 2)^2

  βs = Float32[]
  for i in 1:num_timesteps
    t1 = (i - 1) / num_timesteps
    t2 = i / num_timesteps
    push!(βs, min(1 - α_bar(t2) / α_bar(t1), max_beta))
  end

  return βs
end

"""
Sigmoid beta schedule.

cf. https://arxiv.org/abs/2203.02923
and https://github.com/MinkaiXu/GeoDiff/blob/main/models/epsnet/diffusion.py#L34
"""
function sigmoid_beta_schedule(num_timesteps::Int, β_start=0.0001f0, β_end=0.02f0)
  x = range(-6, 6, length=num_timesteps)
  sigmoid(x) * (β_end - β_start) + β_start
end

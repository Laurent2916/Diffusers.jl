"""
Cosine beta schedule.

```math
\\overline{\\alpha}_t = \\cos \\left( \\frac{t / T + \\epsilon}{1 + \\epsilon} \\frac{\\pi}{2} \\right)
```

## Input
  * `T::Int`: number of timesteps
  * `βₘₐₓ::Real=0.999f0`: maximum value of β
  * `ϵ::Real=1.0f-3`: small value used to avoid division by zero

## Output
  * `β::Vector{Real}`: βₜ values at each timestep t

## References
* [nichol2021improved; Improved Denoising Diffusion Probabilistic Models](@cite)
* [github:openai/improved-diffusion/improved_diffusion/gaussian_diffusion.py](https://github.com/openai/improved-diffusion/blob/783b6740edb79fdb7d063250db2c51cc9545dcd1/improved_diffusion/gaussian_diffusion.py#L36)
"""
function cosine_beta_schedule(T::Integer, βₘₐₓ::Real=0.999f0, ϵ::Real=1.0f-3)
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

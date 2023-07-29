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

module BetaSchedules

# variance schedulers
include("Linear.jl")
include("ScaledLinear.jl")
include("Cosine.jl")
include("Sigmoid.jl")

export
  linear_beta_schedule,
  scaled_linear_beta_schedule,
  cosine_beta_schedule,
  sigmoid_beta_schedule

# utils
include("ZeroSNR.jl")

export
  rescale_zero_terminal_snr

end # module BetaSchedules

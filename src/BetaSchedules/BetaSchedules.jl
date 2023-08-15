module BetaSchedules

# variance schedulers
include("Linear.jl")
include("ScaledLinear.jl")
include("Cosine.jl")
include("Sigmoid.jl")
include("Exponential.jl")

# utils
include("ZeroSNR.jl")

export
  # Beta Schedules
  linear_beta_schedule,
  scaled_linear_beta_schedule,
  cosine_beta_schedule,
  sigmoid_beta_schedule,
  exponential_beta_schedule,

  # Beta Schedule utils
  rescale_zero_terminal_snr

end # module BetaSchedules

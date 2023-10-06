module Schedulers

include("Abstract.jl")
include("DDPM.jl")
include("DDIM.jl")

export
  # Schedulers
  DDPM,
  DDIM,

  # Scheduler methods
  forward,
  reverse,
  get_velocity,

  # VarianceType enum
  VarianceType,
  FIXED_SMALL,
  FIXED_SMALL_LOG,
  FIXED_LARGE,
  FIXED_LARGE_LOG,
  LEARNED,

  # PredictionType enum
  PredictionType,
  EPSILON,
  SAMPLE,
  VELOCITY

end # module Schedulers

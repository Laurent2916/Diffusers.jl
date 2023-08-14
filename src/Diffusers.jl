module Diffusers

include("BetaSchedules/BetaSchedules.jl")

import .BetaSchedules:
  # Beta Schedules
  linear_beta_schedule,
  scaled_linear_beta_schedule,
  cosine_beta_schedule,
  sigmoid_beta_schedule,

  # Beta Schedule utils
  rescale_zero_terminal_snr

export
  # Beta Schedules
  linear_beta_schedule,
  scaled_linear_beta_schedule,
  cosine_beta_schedule,
  sigmoid_beta_schedule,

  # Beta Schedule utils
  rescale_zero_terminal_snr

include("Schedulers/Schedulers.jl")

import .Schedulers:
  # Scheduler
  DDPM,

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

export
  # Scheduler
  DDPM,

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

end # module Diffusers

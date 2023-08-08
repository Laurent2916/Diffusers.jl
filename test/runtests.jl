using Diffusers
using Aqua

Aqua.test_all(Diffusers)

include("Schedulers.jl")
include("BetaSchedules.jl")

import Diffusers
using Random
using Plots

function make_spiral(rng::AbstractRNG, n_samples::Int=1000)
  t_min = 1.5π
  t_max = 4.5π

  t = rand(rng, n_samples) * (t_max - t_min) .+ t_min

  x = t .* cos.(t)
  y = t .* sin.(t)

  permutedims([x y], (2, 1))
end

make_spiral(n_samples::Int=1000) = make_spiral(Random.GLOBAL_RNG, n_samples)

function normalize_zero_to_one(x)
  x_min, x_max = extrema(x)
  x_norm = (x .- x_min) ./ (x_max - x_min)
  x_norm
end

function normalize_neg_one_to_one(x)
  2 * normalize_zero_to_one(x) .- 1
end

n_samples = 1000
data = normalize_neg_one_to_one(make_spiral(n_samples))
scatter(data[1, :], data[2, :],
  alpha=0.5,
  aspectratio=:equal,
)

num_timesteps = 1000
scheduler = Diffusers.DDPM(
  Vector{Float64},
  Diffusers.cosine_beta_schedule(num_timesteps, 0.999f0, 0.001f0),
)

noise = randn(size(X))

anim = @animate for i in 1:num_timesteps
  noisy_data = Diffusers.add_noise(scheduler, X, noise, [i])
  scatter(noise[1, :], noise[2, :],
    alpha=0.1,
    aspectratio=:equal,
    label="noise",
    legend=:topright,
  )
  scatter!(noisy_data[1, :], noisy_data[2, :],
    alpha=0.5,
    aspectratio=:equal,
    label="noisy data",
  )
  scatter!(data[1, :], data[2, :],
    alpha=0.5,
    aspectratio=:equal,
    label="data",
  )
  i_str = lpad(i, 3, "0")
  title!("t = $(i_str)")
  xlims!(-3, 3)
  ylims!(-3, 3)
end

gif(anim, "swissroll.gif", fps=50)

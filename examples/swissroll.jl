import Diffusers
using Flux
using Random
using Plots
using ProgressMeter

# utils
include("Embeddings.jl")
include("ConditionalChain.jl")

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

num_timesteps = 100
scheduler = Diffusers.DDPM(
  Vector{Float64},
  Diffusers.cosine_beta_schedule(num_timesteps, 0.999f0, 0.001f0),
)

noise = randn(size(data))

anim = @animate for i in cat(1:num_timesteps, repeat([num_timesteps], 50), dims=1)
  noisy_data = Diffusers.add_noise(scheduler, data, noise, [i])
  scatter(noise[1, :], noise[2, :],
    alpha=0.1,
    aspectratio=:equal,
    label="noise",
    legend=:outertopright,
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

d_hid = 32
model = ConditionalChain(
  Parallel(
    .+,
    Dense(2, d_hid),
    Chain(
      SinusoidalPositionEmbedding(num_timesteps, d_hid),
      Dense(d_hid, d_hid))
  ),
  relu,
  Parallel(
    .+,
    Dense(d_hid, d_hid),
    Chain(
      SinusoidalPositionEmbedding(num_timesteps, d_hid),
      Dense(d_hid, d_hid))
  ),
  relu,
  Parallel(
    .+,
    Dense(d_hid, d_hid),
    Chain(
      SinusoidalPositionEmbedding(num_timesteps, d_hid),
      Dense(d_hid, d_hid))
  ),
  relu,
  Dense(d_hid, 2),
)

model(data, [100])

num_epochs = 1000
loss = Flux.Losses.mse
opt = Flux.setup(Adam(0.0001), model)
dataloader = Flux.DataLoader(data |> cpu; batchsize=32, shuffle=true);
progress = Progress(num_epochs; desc="training", showspeed=true)
for epoch = 1:num_epochs
  params = Flux.params(model)
  for data in dataloader
    noise = randn(size(data))
    timesteps = rand(2:num_timesteps, size(data)[2]) # TODO: fix start at timestep=2, bruh
    noisy_data = Diffusers.add_noise(scheduler, data, noise, timesteps)
    grads = Flux.gradient(model) do m
      model_output = m(noisy_data, timesteps)
      noise_prediction, _ = Diffusers.step(scheduler, noisy_data, model_output, timesteps)
      loss(noise, noise_prediction)
    end
    Flux.update!(opt, params, grads)
  end
  ProgressMeter.next!(progress)
end

# sampling animation
anim = @animate for timestep in num_timesteps:-1:2
  model_output = model(data, [timestep])
  sampled_data, x0_pred = Diffusers.step(scheduler, data, model_output, [timestep])

  p1 = scatter(sampled_data[1, :], sampled_data[2, :],
    alpha=0.5,
    aspectratio=:equal,
    label="sampled data",
    legend=false,
  )
  scatter!(data[1, :], data[2, :],
    alpha=0.5,
    aspectratio=:equal,
    label="data",
  )

  p2 = scatter(x0_pred[1, :], x0_pred[2, :],
    alpha=0.5,
    aspectratio=:equal,
    label="sampled data",
    legend=false,
  )
  scatter!(data[1, :], data[2, :],
    alpha=0.5,
    aspectratio=:equal,
    label="data",
  )

  l = @layout [a b]
  i_str = lpad(timestep, 3, "0")
  plot(p1, p2,
    layout=l,
    plot_title="t = $(i_str)",
  )
  xlims!(-2, 2)
  ylims!(-2, 2)
end

gif(anim, "sampling.gif", fps=30)

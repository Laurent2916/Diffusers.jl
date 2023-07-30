import Diffusers
import Diffusers.Schedulers
import Diffusers.Schedulers: DDPM
import Diffusers.BetaSchedules: cosine_beta_schedule, rescale_zero_terminal_snr
using Flux
using Random
using Plots
using ProgressMeter
using DenoisingDiffusion
using LaTeXStrings

function make_spiral(n_samples::Integer=1000, t_min::Real=1.5π, t_max::Real=4.5π)
  t = rand(n_samples) * (t_max - t_min) .+ t_min

  x = t .* cos.(t)
  y = t .* sin.(t)

  permutedims([x y], (2, 1))
end

function normalize_zero_to_one(x)
  x_min, x_max = extrema(x)
  x_norm = (x .- x_min) ./ (x_max - x_min)
  x_norm
end

function normalize_neg_one_to_one(x)
  2 * normalize_zero_to_one(x) .- 1
end

# make a dataset of 100 spirals
n_points = 2500
dataset = make_spiral(n_points, 1π, 5π)
dataset = normalize_neg_one_to_one(dataset)
scatter(dataset[1, :], dataset[2, :],
  alpha=0.5,
  aspectratio=:equal,
)

num_timesteps = 100
scheduler = DDPM(
  Vector{Float64},
  rescale_zero_terminal_snr(
    cosine_beta_schedule(num_timesteps)
  )
);

data = dataset[:, 1:100]
noise = randn(size(data))

anim = @animate for t in cat(fill(0, 25), 1:num_timesteps, fill(num_timesteps, 50), dims=1)
  if t == 0
    scatter(noise[1, :], noise[2, :],
      alpha=0.3,
      aspectratio=:equal,
      label="noise",
      legend=:outertopright,
    )
    scatter!(data[1, :], data[2, :],
      alpha=0.3,
      aspectratio=:equal,
      label="data",
    )
    scatter!(data[1, :], data[2, :],
      aspectratio=:equal,
      label="noisy data",
    )
    title!("t = " * lpad(t, 3, "0"))
    xlims!(-3, 3)
    ylims!(-3, 3)
  else
    noisy_data = Diffusers.Schedulers.add_noise(scheduler, data, noise, [t])
    scatter(noise[1, :], noise[2, :],
      alpha=0.3,
      aspectratio=:equal,
      label="noise",
      legend=:outertopright,
    )
    scatter!(data[1, :], data[2, :],
      alpha=0.3,
      aspectratio=:equal,
      label="data",
    )
    scatter!(noisy_data[1, :], noisy_data[2, :],
      aspectratio=:equal,
      label="noisy data",
    )
    title!(latexstring("t = " * lpad(t, 3, "0")))
    xlims!(-3, 3)
    ylims!(-3, 3)
  end
end
gif(anim, anim.dir * ".gif", fps=50)

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

num_epochs = 100;
loss = Flux.Losses.mse;
opt = Flux.setup(Adam(0.0001), model);
dataloader = Flux.DataLoader(dataset |> cpu; batchsize=32, shuffle=true);
progress = Progress(num_epochs; desc="training", showspeed=true);
for epoch = 1:num_epochs
  params = Flux.params(model)
  for data in dataloader
    noise = randn(size(data))
    timesteps = rand(2:num_timesteps, size(data, ndims(data))) # TODO: fix start at timestep=2, bruh
    noisy_data = Diffusers.Schedulers.add_noise(scheduler, data, noise, timesteps)
    grads = Flux.gradient(model) do m
      model_output = m(noisy_data, timesteps)
      noise_prediction, _ = Diffusers.Schedulers.step(scheduler, noisy_data, model_output, timesteps)
      loss(noise, noise_prediction)
    end
    Flux.update!(opt, params, grads)
  end
  ProgressMeter.next!(progress)
end

## sampling animation

sample = randn(2, 100)
sample_old = sample
predictions = []
anim = for timestep in num_timesteps:-1:1
  model_output = model(data, [timestep])
  sample, x0_pred = Diffusers.Schedulers.step(scheduler, data, model_output, [timestep])
  push!(predictions, (sample, x0_pred, timestep))
end

anim = @animate for i in cat(fill(0, 50), 1:num_timesteps, fill(num_timesteps, 50), dims=1)
  if i == 0
    p1 = scatter(dataset[1, :], dataset[2, :],
      alpha=0.01,
      aspectratio=:equal,
      title=L"x_t",
      legend=false,
    )
    scatter!(sample_old[1, :], sample_old[2, :])

    p2 = scatter(dataset[1, :], dataset[2, :],
      alpha=0.01,
      aspectratio=:equal,
      title=L"x_0",
      legend=false,
    )

    l = @layout [a b]
    t_str = lpad(num_timesteps, 3, "0")
    plot(p1, p2,
      layout=l,
      plot_title=latexstring("t = $(t_str)"),
    )
    xlims!(-2, 2)
    ylims!(-2, 2)
  else
    sample, x_0, timestep = predictions[i]
    p1 = scatter(dataset[1, :], dataset[2, :],
      alpha=0.01,
      aspectratio=:equal,
      legend=false,
      title=L"x_t",
    )
    scatter!(sample[1, :], sample[2, :])

    p2 = scatter(dataset[1, :], dataset[2, :],
      alpha=0.01,
      aspectratio=:equal,
      legend=false,
      title=L"x_0",
    )
    scatter!(x_0[1, :], x_0[2, :])

    l = @layout [a b]
    t_str = lpad(timestep - 1, 3, "0")
    plot(p1, p2,
      layout=l,
      plot_title=latexstring("t = $(t_str)"),
    )
    xlims!(-2, 2)
    ylims!(-2, 2)
  end
end
gif(anim, anim.dir * ".gif", fps=50)

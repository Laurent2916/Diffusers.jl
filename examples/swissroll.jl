import Diffusers
using Flux
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

num_timesteps = 100
scheduler = Diffusers.DDPM(
  Vector{Float64},
  Diffusers.cosine_beta_schedule(num_timesteps, 0.999f0, 0.001f0),
)

noise = randn(size(data))

anim = @animate for i in cat(collect(1:num_timesteps), repeat([num_timesteps], 50), dims=1)
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
model = Diffusers.ConditionalChain(
  Parallel(
    .+,
    Dense(2, d_hid),
    Chain(
      Diffusers.SinusoidalPositionEmbedding(num_timesteps, d_hid),
      Dense(d_hid, d_hid))
  ),
  relu,
  Parallel(
    .+,
    Dense(d_hid, d_hid),
    Chain(
      Diffusers.SinusoidalPositionEmbedding(num_timesteps, d_hid),
      Dense(d_hid, d_hid))
  ),
  relu,
  Parallel(
    .+,
    Dense(d_hid, d_hid),
    Chain(
      Diffusers.SinusoidalPositionEmbedding(num_timesteps, d_hid),
      Dense(d_hid, d_hid))
  ),
  relu,
  Dense(d_hid, 2),
)

model(data, [100])


num_epochs = 10
loss = Flux.Losses.mse
dataloader = Flux.DataLoader(X |> to_device; batchsize=32, shuffle=true);
for epoch = 1:num_epochs
  progress = Progress(length(data); desc="epoch $epoch/$num_epochs")
  params = Flux.params(model)
  for data in dataloader
    grads = Flux.gradient(model) do m
      model_output = m(data)
      noise_prediction = Diffusers.step(model_output, timesteps, scheduler)
      loss(noise, noise_prediction)
    end
    Flux.update!(opt, params, grads)
    ProgressMeter.next!(progress; showvalues=[("batch loss", @sprintf("%.5f", batch_loss))])
  end
end


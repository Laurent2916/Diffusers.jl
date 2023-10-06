# This example compares the different beta schedules available in Diffusers.jl.
# Code related to the generation of the datasets and the plots is hidden.

using Diffusers.Schedulers: DDPM, forward # hide
using Diffusers.BetaSchedules # hide
using ProgressMeter # hide
using LaTeXStrings # hide
using Random # hide
using Plots # hide
using Flux # hide
using MLDatasets # hide

function normalize_zero_to_one(x) # hide
  x_min, x_max = extrema(x) # hide
  x_norm = (x .- x_min) ./ (x_max - x_min) # hide
  x_norm # hide
end # hide

function normalize_neg_one_to_one(x) # hide
  2 * normalize_zero_to_one(x) .- 1 # hide
end # hide

num_timesteps = 100 # hide
beta_schedules = [ # hide
  linear_beta_schedule, # hide
  scaled_linear_beta_schedule, # hide
  cosine_beta_schedule, # hide
  sigmoid_beta_schedule, # hide
  exponential_beta_schedule, # hide
] # hide
schedulers = [ # hide
  DDPM(collect(schedule(num_timesteps))) for schedule in beta_schedules # hide
]; # hide

# ## Swiss Roll

function make_spiral(n_samples::Integer=1000, t_min::Real=1.5π, t_max::Real=4.5π) # hide
  t = rand(typeof(t_min), n_samples) * (t_max - t_min) .+ t_min # hide

  x = t .* cos.(t) # hide
  y = t .* sin.(t) # hide

  permutedims([x y], (2, 1)) # hide
end # hide

n_points = 1000; # hide
dataset = make_spiral(n_points, 1.5f0 * π, 4.5f0 * π); # hide
dataset = normalize_neg_one_to_one(dataset); # hide

noise = randn(Float32, size(dataset)) # hide
anim = @animate for t in cat(fill(0, 20), 1:num_timesteps, fill(num_timesteps, 20), dims=1) # hide
  plots = [] # hide
  for (i, (scheduler, beta_schedule)) in enumerate(zip(schedulers, beta_schedules)) # hide
    if t == 0 # hide
      scatter(dataset[1, :], dataset[2, :], # hide
        alpha=0.5, # hide
        aspectratio=:equal, # hide
        legend=false, # hide
      ) # hide
      plot = scatter!(dataset[1, :], dataset[2, :], # hide
        alpha=0.5, # hide
        aspectratio=:equal, # hide
      ) # hide
      title!(string(beta_schedule)) # hide
      xlims!(-3, 3) # hide
      ylims!(-3, 3) # hide
    else # hide
      scatter(dataset[1, :], dataset[2, :], # hide
        alpha=0.5, # hide
        aspectratio=:equal, # hide
        legend=false, # hide
      ) # hide
      noisy_data = forward(scheduler, dataset, noise, [t]) # hide
      plot = scatter!(noisy_data[1, :], noisy_data[2, :], # hide
        alpha=0.5, # hide
        aspectratio=:equal, # hide
      ) # hide
      title!(string(beta_schedule)) # hide
      xlims!(-3, 3) # hide
      ylims!(-3, 3) # hide
    end # hide
    push!(plots, plot) # hide
  end # hide
  plot(plots...; size=(1200, 800)) # hide
end # hide

gif(anim, anim.dir * ".gif", fps=20) # hide

# ## Double Square

function make_square(n_samples::Integer=1000) # hide
  x = rand(n_samples) .* 2 .- 1 # hide
  y = rand(n_samples) .* 2 .- 1 # hide
  p = permutedims([x y], (2, 1)) # hide
  p ./ maximum(abs.(p), dims=1) # hide
end # hide

dataset = hcat( # hide
  make_square(Int(n_points / 2)) ./ 2 .- 1.5, # hide
  make_square(Int(n_points / 2)) ./ 2 .+ 1.5 # hide
) # hide

noise = randn(Float32, size(dataset)) # hide
anim = @animate for t in cat(fill(0, 20), 1:num_timesteps, fill(num_timesteps, 20), dims=1) # hide
  plots = [] # hide
  for (i, (scheduler, beta_schedule)) in enumerate(zip(schedulers, beta_schedules)) # hide
    if t == 0 # hide
      scatter(dataset[1, :], dataset[2, :], # hide
        alpha=0.5, # hide
        aspectratio=:equal, # hide
        legend=false, # hide
      ) # hide
      plot = scatter!(dataset[1, :], dataset[2, :], # hide
        alpha=0.5, # hide
        aspectratio=:equal, # hide
      ) # hide
      title!(string(beta_schedule)) # hide
      xlims!(-3, 3) # hide
      ylims!(-3, 3) # hide
    else # hide
      scatter(dataset[1, :], dataset[2, :], # hide
        alpha=0.5, # hide
        aspectratio=:equal, # hide
        legend=false, # hide
      ) # hide
      noisy_data = forward(scheduler, dataset, noise, [t]) # hide
      plot = scatter!(noisy_data[1, :], noisy_data[2, :], # hide
        alpha=0.5, # hide
        aspectratio=:equal, # hide
      ) # hide
      title!(string(beta_schedule)) # hide
      xlims!(-3, 3) # hide
      ylims!(-3, 3) # hide
    end # hide
    push!(plots, plot) # hide
  end # hide
  plot(plots...; size=(1200, 800)) # hide
end # hide

gif(anim, anim.dir * ".gif", fps=20) # hide

# ## MNIST

dataset = MNIST(:test)[16].features # hide
dataset = rotl90(dataset) # hide
dataset = normalize_neg_one_to_one(dataset) # hide

noise = randn(Float32, size(dataset)) # hide
anim = @animate for t in cat(fill(0, 20), 1:num_timesteps, fill(num_timesteps, 20), dims=1) # hide
  plots = [] # hide
  for (i, (scheduler, beta_schedule)) in enumerate(zip(schedulers, beta_schedules)) # hide
    if t == 0 # hide
      plot = heatmap( # hide
        dataset, # hide
        c=:grayC, # hide
        legend=:none, # hide
        aspect_ratio=:equal, # hide
        grid=false, # hide
        axis=false # hide
      ) # hide
      title!(string(beta_schedule)) # hide
    else # hide
      noisy_data = forward(scheduler, dataset, noise, [t]) # hide
      plot = heatmap( # hide
        noisy_data, # hide
        c=:grayC, # hide
        legend=:none, # hide
        aspect_ratio=:equal, # hide
        grid=false, # hide
        axis=false # hide
      ) # hide
      title!(string(beta_schedule)) # hide
    end # hide

    push!(plots, plot) # hide
  end # hide
  plot(plots...; size=(1200, 800)) # hide
end # hide

gif(anim, anim.dir * ".gif", fps=20) # hide
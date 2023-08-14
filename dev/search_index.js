var documenterSearchIndex = {"docs":
[{"location":"schedulers/","page":"Schedulers","title":"Schedulers","text":"(Image: Markovian Hierarchical Variational Autoencoder)","category":"page"},{"location":"schedulers/","page":"Schedulers","title":"Schedulers","text":"Modules = [Diffusers.Schedulers]","category":"page"},{"location":"schedulers/#Diffusers.Schedulers.DDPM","page":"Schedulers","title":"Diffusers.Schedulers.DDPM","text":"Denoising Diffusion Probabilistic Models (DDPM) scheduler.\n\nReferences\n\n[2006.11239] Denoising Diffusion Probabilistic Models\n\n\n\n\n\n","category":"type"},{"location":"schedulers/#Diffusers.Schedulers.Scheduler","page":"Schedulers","title":"Diffusers.Schedulers.Scheduler","text":"Abstract type for schedulers.\n\n\n\n\n\n","category":"type"},{"location":"schedulers/#Diffusers.Schedulers.forward-Tuple{Diffusers.Schedulers.Scheduler, AbstractArray, AbstractArray, AbstractArray}","page":"Schedulers","title":"Diffusers.Schedulers.forward","text":"Add noise to clean data using the forward diffusion process.\n\nInput\n\nscheduler::Scheduler: scheduler to use\nx₀::AbstractArray: clean data to add noise to\nϵ::AbstractArray: noise to add to clean data\nt::AbstractArray: timesteps used to weight the noise\n\nOutput\n\nxₜ::AbstractArray: noisy data at the given timesteps\n\n\n\n\n\n","category":"method"},{"location":"schedulers/#Diffusers.Schedulers.get_velocity-Tuple{Diffusers.Schedulers.Scheduler, AbstractArray, AbstractArray, AbstractArray}","page":"Schedulers","title":"Diffusers.Schedulers.get_velocity","text":"Compute the velocity of the diffusion process.\n\nInput\n\nscheduler::Scheduler: scheduler to use\nx₀::AbstractArray: clean data to add noise to\nϵ::AbstractArray: noise to add to clean data\nt::AbstractArray: timesteps used to weight the noise\n\nOutput\n\nvₜ::AbstractArray: velocity at the given timesteps\n\nReferences\n\n[2202.00512] Progressive Distillation for Fast Sampling of Diffusion Models (Ann. D)\n\n\n\n\n\n","category":"method"},{"location":"schedulers/#Diffusers.Schedulers.reverse-Tuple{Diffusers.Schedulers.Scheduler, AbstractArray, AbstractArray, AbstractArray}","page":"Schedulers","title":"Diffusers.Schedulers.reverse","text":"Remove noise from model output using the backward diffusion process.\n\nInput\n\nscheduler::Scheduler: scheduler to use\nxₜ::AbstractArray: sample to be denoised\nϵᵧ::AbstractArray: predicted noise to remove\nt::AbstractArray: timestep t of xₜ\n\nOutput\n\nxₜ₋₁::AbstractArray: denoised sample at t=t-1\nx̂₀::AbstractArray: denoised sample at t=0\n\n\n\n\n\n","category":"method"},{"location":"beta_schedules/","page":"Beta Schedules","title":"Beta Schedules","text":"using Diffusers.BetaSchedules\nusing LaTeXStrings\nusing PlotlyJS\n\nT = 1000\n\nβ_linear = linear_beta_schedule(T)\nβ_scaled_linear = scaled_linear_beta_schedule(T)\nβ_cosine = cosine_beta_schedule(T)\nβ_sigmoid = sigmoid_beta_schedule(T)\n\nα̅_linear = cumprod(1 .- β_linear)\nα̅_scaled_linear = cumprod(1 .- β_scaled_linear)\nα̅_cosine = cumprod(1 .- β_cosine)\nα̅_sigmoid = cumprod(1 .- β_sigmoid)\n\np1 = plot(\n  [\n    scatter(y=β_linear, name=\"Linear\"),\n    scatter(y=β_scaled_linear, name=\"Scaled linear\", visible=\"legendonly\"),\n    scatter(y=β_cosine, name=\"Cosine\"),\n    scatter(y=β_sigmoid, name=\"Sigmoid\", visible=\"legendonly\"),\n  ],\n  Layout(\n    updatemenus=[\n      attr(\n        type=\"buttons\",\n        active=1,\n        buttons=[\n          attr(\n            label=\"Linear\",\n            method=\"relayout\",\n            args=[\"yaxis.type\", \"linear\"],\n          ),\n          attr(\n            label=\"Log\",\n            method=\"relayout\",\n            args=[\"yaxis.type\", \"log\"],\n          ),\n        ]\n      ),\n    ],\n    xaxis=attr(\n      title=L\"t\",\n    ),\n    yaxis=attr(\n      type=\"log\",\n      title=L\"\\beta\",\n    )\n  )\n)\n\np2 = plot(\n  [\n    scatter(y=α̅_linear, name=\"Linear\"),\n    scatter(y=α̅_scaled_linear, name=\"Scaled linear\", visible=\"legendonly\"),\n    scatter(y=α̅_cosine, name=\"Cosine\"),\n    scatter(y=α̅_sigmoid, name=\"Sigmoid\", visible=\"legendonly\"),\n  ],\n  Layout(\n    updatemenus=[\n      attr(\n        type=\"buttons\",\n        buttons=[\n          attr(\n            label=\"Linear\",\n            method=\"relayout\",\n            args=[\"yaxis.type\", \"linear\"],\n          ),\n          attr(\n            label=\"Log\",\n            method=\"relayout\",\n            args=[\"yaxis.type\", \"log\"],\n          ),\n        ],\n      ),\n    ],\n    xaxis=attr(\n      title=L\"t\",\n    ),\n    yaxis=attr(\n      title=L\"\\overline\\alpha\",\n    )\n  )\n)\n\nmkpath(\"beta_schedules\")\nsavefig(p1, \"beta_schedules/beta_schedules.html\")\nsavefig(p2, \"beta_schedules/alpha_bar_schedules.html\")\nnothing","category":"page"},{"location":"beta_schedules/","page":"Beta Schedules","title":"Beta Schedules","text":"<object type=\"text/html\" data=\"beta_schedules.html\" style=\"width:100%;height:420px;\"></object>\n<object type=\"text/html\" data=\"alpha_bar_schedules.html\" style=\"width:100%;height:420px;\"></object>","category":"page"},{"location":"beta_schedules/","page":"Beta Schedules","title":"Beta Schedules","text":"Modules = [Diffusers.BetaSchedules]","category":"page"},{"location":"beta_schedules/#Diffusers.BetaSchedules.cosine_beta_schedule","page":"Beta Schedules","title":"Diffusers.BetaSchedules.cosine_beta_schedule","text":"Cosine beta schedule.\n\nInput\n\nT::Int: number of timesteps\nβₘₐₓ::Real=0.999f0: maximum value of β\nϵ::Real=1.0f-3: small value used to avoid division by zero\n\nOutput\n\nβ::Vector{Real}: βₜ values at each timestep t\n\nReferences\n\n[2102.09672] Improved Denoising Diffusion Probabilistic Models\ngithub:openai/improved-diffusion\n\n\n\n\n\n","category":"function"},{"location":"beta_schedules/#Diffusers.BetaSchedules.linear_beta_schedule","page":"Beta Schedules","title":"Diffusers.BetaSchedules.linear_beta_schedule","text":"Linear beta schedule.\n\nInput\n\nT::Integer: number of timesteps\nβ₁::Real=1.0f-4: initial (t=1) value of β\nβ₋₁::Real=2.0f-2: final (t=T) value of β\n\nOutput\n\nβ::Vector{Real}: βₜ values at each timestep t\n\nReferences\n\n[2006.11239] Denoising Diffusion Probabilistic Models\n\n\n\n\n\n","category":"function"},{"location":"beta_schedules/#Diffusers.BetaSchedules.rescale_zero_terminal_snr-Tuple{AbstractArray}","page":"Beta Schedules","title":"Diffusers.BetaSchedules.rescale_zero_terminal_snr","text":"Rescale betas to have zero terminal Signal to Noise Ratio (SNR).\n\nInput\n\nβ::AbstractArray: βₜ values at each timestep t\n\nOutput\n\nβ::Vector{Real}: rescaled βₜ values at each timestep t\n\nReferences\n\n[2305.08891] Rescaling Diffusion Models (Alg. 1)\n\n\n\n\n\n","category":"method"},{"location":"beta_schedules/#Diffusers.BetaSchedules.scaled_linear_beta_schedule","page":"Beta Schedules","title":"Diffusers.BetaSchedules.scaled_linear_beta_schedule","text":"Scaled linear beta schedule.\n\nInput\n\nT::Int: number of timesteps\nβ₁::Real=1.0f-4: initial value of β\nβ₋₁::Real=2.0f-2: final value of β\n\nOutput\n\nβ::Vector{Real}: βₜ values at each timestep t\n\nReferences\n\n[2006.11239] Denoising Diffusion Probabilistic Models\n\n\n\n\n\n","category":"function"},{"location":"beta_schedules/#Diffusers.BetaSchedules.sigmoid_beta_schedule","page":"Beta Schedules","title":"Diffusers.BetaSchedules.sigmoid_beta_schedule","text":"Sigmoid beta schedule.\n\nInput\n\nT::Int: number of timesteps\nβ₁::Real=1.0f-4: initial value of β\nβ₋₁::Real=2.0f-2: final value of β\n\nOutput\n\nβ::Vector{Real}: βₜ values at each timestep t\n\nReferences\n\n[2203.02923] GeoDiff: a Geometric Diffusion Model for Molecular Conformation Generation\ngithub.com:MinkaiXu/GeoDiff\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = Diffusers","category":"page"},{"location":"#Diffusers","page":"Home","title":"Diffusers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for Diffusers.jl.","category":"page"},{"location":"#Index","page":"Home","title":"Index","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}

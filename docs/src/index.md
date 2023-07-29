```@meta
CurrentModule = Diffusers
```

# Diffusers

Documentation for [Diffusers.jl](https://github.com/Laurent2916/Diffusers.jl).

```@eval
using Diffusers.BetaSchedules
using Plots
plotlyjs()

T = 1000
linear = linear_beta_schedule(T)
scaled_linear = scaled_linear_beta_schedule(T)
cosine = cosine_beta_schedule(T)
sigmoid = sigmoid_beta_schedule(T)

plot(
  [linear, scaled_linear, cosine, sigmoid],
  label=["linear" "scaled_linear" "cosine" "sigmoid"],
  xlabel="t",
  ylabel="β",
  title="Beta schedules",
  legend=:topleft,
  yscale=:log10,
)

savefig("beta_schedules.html")
nothing
```

```@raw html
<object type="text/html" data="beta_schedules.html" style="width:100%;height:420px;"></object>
```

```@index
```

```@autodocs
Modules = [Diffusers]
```

```@autodocs
Modules = [Diffusers.BetaSchedules]
```

```@autodocs
Modules = [Diffusers.Schedulers]
```

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

savefig("fig_beta_schedules.html")
nothing
```

```@raw html
<object type="text/html" data="fig_beta_schedules.html" style="width:100%;height:420px;"></object>
```

```@autodocs
Modules = [Diffusers.BetaSchedules]
```

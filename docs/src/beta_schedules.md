```@eval
using Diffusers.BetaSchedules
using LaTeXStrings
using PlotlyJS

T = 1000

β_linear = linear_beta_schedule(T)
β_scaled_linear = scaled_linear_beta_schedule(T)
β_cosine = cosine_beta_schedule(T)
β_sigmoid = sigmoid_beta_schedule(T)

α̅_linear = cumprod(1 .- β_linear)
α̅_scaled_linear = cumprod(1 .- β_scaled_linear)
α̅_cosine = cumprod(1 .- β_cosine)
α̅_sigmoid = cumprod(1 .- β_sigmoid)

p1 = plot(
  [
    scatter(y=β_linear, name="Linear"),
    scatter(y=β_scaled_linear, name="Scaled linear", visible="legendonly"),
    scatter(y=β_cosine, name="Cosine"),
    scatter(y=β_sigmoid, name="Sigmoid", visible="legendonly"),
  ],
  Layout(
    updatemenus=[
      attr(
        type="buttons",
        active=1,
        buttons=[
          attr(
            label="Linear",
            method="relayout",
            args=["yaxis.type", "linear"],
          ),
          attr(
            label="Log",
            method="relayout",
            args=["yaxis.type", "log"],
          ),
        ]
      ),
    ],
    xaxis=attr(
      title=L"$t$",
    ),
    yaxis=attr(
      type="log",
      title=L"\beta",
    )
  )
)

p2 = plot(
  [
    scatter(y=α̅_linear, name="Linear"),
    scatter(y=α̅_scaled_linear, name="Scaled linear", visible="legendonly"),
    scatter(y=α̅_cosine, name="Cosine"),
    scatter(y=α̅_sigmoid, name="Sigmoid", visible="legendonly"),
  ],
  Layout(
    updatemenus=[
      attr(
        type="buttons",
        buttons=[
          attr(
            label="Linear",
            method="relayout",
            args=["yaxis.type", "linear"],
          ),
          attr(
            label="Log",
            method="relayout",
            args=["yaxis.type", "log"],
          ),
        ],
      ),
    ],
    xaxis=attr(
      title=L"$t$",
    ),
    yaxis=attr(
      title=L"\overline\alpha",
    )
  )
)

mkpath("beta_schedules")
savefig(p1, "beta_schedules/beta_schedules.html")
savefig(p2, "beta_schedules/alpha_bar_schedules.html")
nothing
```

```@raw html
<object type="text/html" data="beta_schedules.html" style="width:100%;height:420px;"></object>
<object type="text/html" data="alpha_bar_schedules.html" style="width:100%;height:420px;"></object>
```


```@autodocs
Modules = [Diffusers.BetaSchedules]
```

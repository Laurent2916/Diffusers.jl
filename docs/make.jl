using Diffusers
using Documenter
using Plots

DocMeta.setdocmeta!(Diffusers, :DocTestSetup, :(using Diffusers); recursive=true)

makedocs(;
  modules=[Diffusers],
  authors="Laurent Fainsin <laurent@fainsin.bzh>",
  repo="https://github.com/Laurent2916/Diffusers.jl/blob/{commit}{path}#{line}",
  sitename="Diffusers.jl",
  format=Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    edit_link="main",
    assets=String[]
  ),
  pages=[
    "Home" => "index.md",
    "Schedulers" => "schedulers.md",
    "Beta Schedules" => "beta_schedules.md",
  ]
)

deploydocs(
  repo="github.com/Laurent2916/Diffusers.jl.git",
)

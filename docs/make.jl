using Diffusers
using Documenter
using DocumenterCitations
using Literate

Literate.markdown(joinpath(@__DIR__, "..", "examples", "beta_schedulers_comparison.jl"), joinpath(@__DIR__, "src", "generated"))

DocMeta.setdocmeta!(Diffusers, :DocTestSetup, :(using Diffusers); recursive=true)
bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(bib;
  modules=[Diffusers],
  authors="Laurent Fainsin <laurent@fainsin.bzh>",
  repo="https://github.com/Laurent2916/Diffusers.jl/blob/{commit}{path}#{line}",
  sitename="Diffusers.jl",
  format=Documenter.HTML(;
    prettyurls=true,
    edit_link="master",
    assets=String[]
  ),
  linkcheck=true,
  pages=[
    "Home" => "index.md",
    "API" => [
      "Schedulers" => "schedulers.md",
      "Beta Schedules" => "beta_schedules.md",
    ],
    "Examples" => [
      "Beta Schedules Comparison" => "generated/beta_schedulers_comparison.md",
    ],
    "References" => "references.md",
  ]
)

deploydocs(
  repo="github.com/Laurent2916/Diffusers.jl.git",
)

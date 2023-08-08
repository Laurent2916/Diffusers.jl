import Diffusers: step, add_noise, DDPM, cosine_beta_schedule, rescale_zero_terminal_snr
using Statistics
using Test

@testset "Schedulers tests" begin
  @testset "check `step` correctness" begin
    T = 10
    batch_size = 8
    size = 128

    # create a DDPM with a cosine beta schedule
    ddpm = Diffusers.DDPM(
      Vector{Float32},
      Diffusers.cosine_beta_schedule(T),
    )

    # create some dummy data
    x₀ = ones(Float32, size, size, batch_size)
    ϵ = randn(Float32, size, size, batch_size)

    for t in 1:T
      t = ones(UInt32, batch_size) .* t
      # corrupt x₀ with noise
      xₜ = Diffusers.add_noise(ddpm, x₀, ϵ, t)
      # suppose a model predicted ϵ perfectly
      _, x̂₀ = Diffusers.step(ddpm, xₜ, ϵ, t)
      # test that we recover x₀
      @test x̂₀ ≈ x₀
    end
  end

  @testset "check `add_noise` terminal SNR" begin
    T = 10
    batch_size = 1
    size = 2500

    # create a DDPM with a terminal SNR cosine beta schedule
    ddpm = Diffusers.DDPM(
      Vector{Float32},
      Diffusers.rescale_zero_terminal_snr(
        Diffusers.cosine_beta_schedule(T),
      ),
    )

    # create some dummy data
    x₀ = ones(Float32, size, size, batch_size)
    ϵ = randn(Float32, size, size, batch_size)

    t = ones(UInt32, batch_size) .* T
    # corrupt x₀ with noise
    xₜ = Diffusers.add_noise(ddpm, x₀, ϵ, t)

    @test std(xₜ) ≈ 1.0 atol = 1.0f-3
    @test mean(xₜ) ≈ 0.0 atol = 1.0f-2
  end
end

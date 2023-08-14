import Diffusers: reverse, forward, DDPM, cosine_beta_schedule
import Statistics: mean, std
using Test

@testset "Schedulers tests" begin
  @testset "check `reverse` correctness" begin
    T = 10
    batch_size = 8
    size = 128

    # create a DDPM with a cosine beta schedule
    ddpm = DDPM(cosine_beta_schedule(T))

    # create some dummy data
    x₀ = ones(Float32, size, size, batch_size)
    ϵ = randn(Float32, size, size, batch_size)

    for t in 1:T
      t = ones(UInt32, batch_size) .* t
      # corrupt x₀ with noise
      xₜ = forward(ddpm, x₀, ϵ, t)
      # suppose a model predicted ϵ perfectly
      _, x̂₀ = reverse(ddpm, xₜ, ϵ, t)
      # test that we recover x₀
      @test x̂₀ ≈ x₀
    end
  end

  @testset "check `forward` terminal SNR" begin
    T = 10
    batch_size = 1
    size = 2500

    # create a DDPM with a terminal SNR cosine beta schedule
    ddpm = DDPM(cosine_beta_schedule(T))

    # create some dummy data
    x₀ = ones(Float32, size, size, batch_size)
    ϵ = randn(Float32, size, size, batch_size)

    t = ones(UInt32, batch_size) .* T
    # corrupt x₀ with noise
    xₜ = forward(ddpm, x₀, ϵ, t)

    @test std(xₜ) ≈ 1.0 atol = 1.0f-3
    @test mean(xₜ) ≈ 0.0 atol = 1.0f-2
  end
end

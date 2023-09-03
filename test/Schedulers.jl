import Diffusers: reverse, forward, get_velocity, DDPM, cosine_beta_schedule, VELOCITY, EPSILON, SAMPLE
import Statistics: mean, std
using Test

@testset "Schedulers tests" begin

  @testset "check `reverse` correctness" begin
    T = 10
    batch_size = 8
    data_size = 128

    # create a DDPM with a cosine beta schedule
    for scheduler_type in [DDPM, DDIM]

      scheduler = scheduler_type(cosine_beta_schedule(T))
      @testset "Scheduler == $scheduler_type" begin

        # create some dummy data
        x₀ = ones(Float32, data_size, data_size, batch_size)
        ϵ = randn(Float32, data_size, data_size, batch_size)

        @testset "PredictionType == EPSILON" begin
          for t in 1:T
            t = ones(UInt32, batch_size) .* t
            # get xₜ from forward diffusion process
            xₜ = forward(scheduler, x₀, ϵ, t)
            # suppose a model predicted ϵ perfectly
            ϵᵧ = ϵ
            # use reverse diffusion process to retreive x̂₀
            _, x̂₀ = reverse(scheduler, xₜ, ϵᵧ, t; prediction_type=EPSILON)
            # test that we recover x₀
            @test x̂₀ ≈ x₀
          end
        end

        @testset "PredictionType == SAMPLE" begin
          for t in 1:T
            t = ones(UInt32, batch_size) .* t
            # get xₜ from forward diffusion process
            xₜ = forward(scheduler, x₀, ϵ, t)
            # suppose a model predicted x₀ perfectly
            ϵᵧ = x₀
            # use reverse diffusion process to retreive x̂₀
            _, x̂₀ = reverse(scheduler, xₜ, ϵᵧ, t; prediction_type=SAMPLE)
            # test that we recover x₀
            @test x̂₀ ≈ x₀
          end
        end

        @testset "PredictionType == VELOCITY" begin
          for t in 1:T
            t = ones(UInt32, batch_size) .* t
            # get xₜ from forward diffusion process
            xₜ = forward(scheduler, x₀, ϵ, t)
            # compute vₜ to train model
            vₜ = get_velocity(scheduler, x₀, ϵ, t)
            # suppose a model predicted vₜ perfectly
            vᵧ = vₜ
            # use reverse diffusion process to retreive x̂₀
            _, x̂₀ = Diffusers.reverse(scheduler, xₜ, vₜ, t; prediction_type=VELOCITY)
            # test that we recover x₀
            @test x̂₀ ≈ x₀
          end
        end

      end

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

using Diffusers.BetaSchedules
using Test

@testset "Variance schedules tests" begin
  @testset "β increases monotonically" begin
    T = 1000

    β_linear = linear_beta_schedule(T)
    β_scaled_linear = scaled_linear_beta_schedule(T)
    β_cosine = cosine_beta_schedule(T)
    β_sigmoid = sigmoid_beta_schedule(T)

    @test all(diff(β_linear) .>= 0)
    @test all(diff(β_scaled_linear) .>= 0)
    @test all(diff(β_cosine) .>= 0)
    @test all(diff(β_sigmoid) .>= 0)
  end
end

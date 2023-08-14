using Diffusers.BetaSchedules
using Test

@testset "Variance schedules tests" begin
  @testset "SNR decreases monotonically" begin
    T = 1000

    β_linear = linear_beta_schedule(T)
    β_scaled_linear = scaled_linear_beta_schedule(T)
    β_cosine = cosine_beta_schedule(T)
    β_sigmoid = sigmoid_beta_schedule(T)

    α̅_linear = cumprod(1 .- β_linear)
    α̅_scaled_linear = cumprod(1 .- β_scaled_linear)
    α̅_cosine = cumprod(1 .- β_cosine)
    α̅_sigmoid = cumprod(1 .- β_sigmoid)

    # arxiv:2208.11970 Eq. 109
    SNR_linear = α̅_linear ./ (1 .- α̅_linear)
    SNR_scaled_linear = α̅_scaled_linear ./ (1 .- α̅_scaled_linear)
    SNR_cosine = α̅_cosine ./ (1 .- α̅_cosine)
    SNR_sigmoid = α̅_sigmoid ./ (1 .- α̅_sigmoid)

    @test all(diff(SNR_linear) .<= 0)
    @test all(diff(SNR_scaled_linear) .<= 0)
    @test all(diff(SNR_cosine) .<= 0)
    @test all(diff(SNR_sigmoid) .<= 0)
  end
end

using Diffusers.BetaSchedules
using Test

@testset "Variance schedules tests" begin

  @testset "SNR decreases monotonically" begin
    T = 1000

    for beta_schedule_type in [
      linear_beta_schedule,
      scaled_linear_beta_schedule,
      cosine_beta_schedule,
      sigmoid_beta_schedule,
      exponential_beta_schedule,
    ]
      @testset "Variance schedule == $beta_schedule_type" begin
        β = beta_schedule_type(T)
        α = 1 .- β
        α̅ = cumprod(α)

        SNR = α̅ ./ (1 .- α̅)

        @test all(diff(SNR) .<= 0)
      end
    end
  end

  @testset "ZeroSNR rescaling" begin
    T = 1000

    for beta_schedule_type in [
      linear_beta_schedule,
      scaled_linear_beta_schedule,
      cosine_beta_schedule,
      sigmoid_beta_schedule,
      exponential_beta_schedule,
    ]
      @testset "Variance schedule == $beta_schedule_type" begin
        β = rescale_zero_terminal_snr(beta_schedule_type(T))
        α = 1 .- β
        α̅ = cumprod(α)

        SNR = α̅ ./ (1 .- α̅)

        @test SNR[end] ≈ 0
      end
    end
  end

end

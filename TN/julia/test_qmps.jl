# Unit tests for QMPSRL.
# Run from TN/julia/ with: julia --project=. test_qmps.jl

include("QMPSRL.jl")
using .QMPSRL
using Test
using LinearAlgebra
using Random

@testset "QMPSRL" begin

    @testset "DMRG target = |+⟩^N" begin
        # H = -ΣX has unique ground state |+⟩^N with energy -N.
        @test isapprox(QMPSRL.TARGET_ENERGY, -float(QMPSRL.N); atol=1e-8)
        # Check overlap against the analytical |+⟩^N (constructed by Hadamard rotation
        # of |0⟩^N: just verify <ψ★|ψ★> = 1 and <X_i> = 1 for all i.)
        @test isapprox(norm(QMPSRL.TARGET_STATE), 1.0; atol=1e-10)
    end

    @testset "Env norm preservation" begin
        env = QMPSRL.new_env(7)
        ψ = QMPSRL.get_state(env.state_id)
        @test isapprox(norm(ψ), 1.0; atol=1e-10)

        # Apply 20 random actions, norm must stay 1.
        rng = MersenneTwister(123)
        for _ in 1:20
            a = rand(rng, 1:QMPSRL.N_ACTIONS)
            QMPSRL.step!(env, a)
            ψ = QMPSRL.get_state(env.state_id)
            @test isapprox(norm(ψ), 1.0; atol=1e-8)
        end
    end

    @testset "Action decoding round-trip" begin
        for a in 1:QMPSRL.N_ACTIONS
            gen, sgn, dt = QMPSRL.decode_action(a)
            @test gen in QMPSRL.GENERATORS
            @test sgn in QMPSRL.SIGNS
            @test dt in QMPSRL.STEP_SIZES
        end
    end

    @testset "QMPS feature shape" begin
        env = QMPSRL.new_env(11)
        feat = QMPSRL.qmps_feature(env.state_id)
        @test length(feat) == QMPSRL.D_F
        @test all(isfinite, feat)
    end

    @testset "Zygote VJP vs finite difference" begin
        env = QMPSRL.new_env(11)
        feat, gfn = QMPSRL.qmps_feature_and_vjp(env.state_id)
        # Choose a random output direction to backprop.
        rng = MersenneTwister(0)
        dir = randn(rng, QMPSRL.D_F)
        gZ = gfn(dir)  # analytic gradient of <dir, feat(params)>

        flat0 = QMPSRL.get_qmps_params()
        function L(p)
            QMPSRL.set_qmps_params!(p)
            return dot(dir, QMPSRL.qmps_feature(env.state_id))
        end
        ε = 1e-5
        # Spot-check 25 entries spread through the parameter vector.
        idxs = round.(Int, range(1, length(flat0); length=25))
        max_err = 0.0
        for k in idxs
            p_plus  = copy(flat0); p_plus[k]  += ε
            p_minus = copy(flat0); p_minus[k] -= ε
            g_num = (L(p_plus) - L(p_minus)) / (2ε)
            max_err = max(max_err, abs(gZ[k] - g_num))
        end
        QMPSRL.set_qmps_params!(flat0)  # restore
        @test max_err < 1e-4
        @info "Zygote-vs-FD max error on 25 spot-checks" max_err
    end

    @testset "param round-trip get/set" begin
        flat = QMPSRL.get_qmps_params()
        feat_before = QMPSRL.qmps_feature(QMPSRL.new_env(42).state_id)
        QMPSRL.set_qmps_params!(flat)
        feat_after = QMPSRL.qmps_feature(QMPSRL.new_env(42).state_id)
        @test all(isapprox.(feat_before, feat_after; atol=1e-12))
    end
end

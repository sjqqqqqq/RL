# Unit tests for QMPSRL (Paper Study A reproduction).
# Run from TN/julia/ with: julia --project=. test_qmps.jl

include("QMPSRL.jl")
using .QMPSRL
using Test
using LinearAlgebra
using Random
using ITensors, ITensorMPS

@testset "QMPSRL" begin

    @testset "Target state = |↑↑↑↑⟩" begin
        # H = -ΣZ has unique ground state |↑⟩^N with energy -N (each Z eigenvalue +1).
        @test isapprox(QMPSRL.TARGET_ENERGY, -float(QMPSRL.N); atol=1e-8)
        @test isapprox(norm(QMPSRL.TARGET_STATE), 1.0; atol=1e-10)

        # Overlap with the analytical Z-basis product state.
        ψ_up = MPS(QMPSRL.SITES, ["Up" for _ in 1:QMPSRL.N])
        @test isapprox(abs2(inner(ψ_up, QMPSRL.TARGET_STATE)), 1.0; atol=1e-8)
    end

    @testset "TFIM ground-state energy at (J=1, gx=1.05, gz=0)" begin
        ψ, E = QMPSRL.dmrg_ground(1.0, 1.05, 0.0)
        # ED reference for N=4 TLFI at (J=1, gx=1.05): compute eigenvalues of dense H.
        H = QMPSRL.tlfi_mpo(1.0, 1.05, 0.0)
        # Build dense Hamiltonian by contracting the MPO over a complete basis.
        # Easy way: take an ITensor of H by sequential product, then matrix-form.
        # For N=4 this is 16×16, trivial.
        E_dense = let
            T = H[1]
            for i in 2:QMPSRL.N
                T = T * H[i]
            end
            # Reshape into a 16×16 matrix over (s1..s4 ; s1'..s4').
            sites_in  = [QMPSRL.SITES[i] for i in 1:QMPSRL.N]
            sites_out = [prime(QMPSRL.SITES[i]) for i in 1:QMPSRL.N]
            M = Array(T, sites_in..., sites_out...)
            M = reshape(M, 2^QMPSRL.N, 2^QMPSRL.N)
            real(eigvals(Hermitian((M + M') / 2))[1])
        end
        @test isapprox(E, E_dense; atol=1e-6)
        @info "TFIM (J=1, gx=1.05, gz=0) energy" E E_dense
    end

    @testset "Env norm preservation under 20 random actions" begin
        env = QMPSRL.new_env(7)
        ψ = QMPSRL.get_state(env.state_id)
        @test isapprox(norm(ψ), 1.0; atol=1e-10)

        rng = MersenneTwister(123)
        for _ in 1:20
            a = rand(rng, 0:QMPSRL.N_ACTIONS - 1)  # 0-indexed action
            QMPSRL.step!(env, a)
            ψ = QMPSRL.get_state(env.state_id)
            @test isapprox(norm(ψ), 1.0; atol=1e-8)
        end
    end

    @testset "Action decoding round-trip" begin
        for a in 0:QMPSRL.N_ACTIONS - 1
            gen, h_max, kind = QMPSRL.decode_action(a)
            @test gen in ("X", "Y", "Z", "XX", "YY", "ZZ")
            @test h_max in (+QMPSRL.H_MAX, -QMPSRL.H_MAX)
            @test kind in (:one, :two)
            # Even index → δt+, odd → δt-.
            dt = QMPSRL.duration_for(a)
            @test dt == (a % 2 == 0 ? QMPSRL.DT_PLUS : QMPSRL.DT_MINUS)
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

        rng = MersenneTwister(0)
        dir = randn(rng, QMPSRL.D_F)
        gZ  = gfn(dir)  # analytic gradient of <dir, feat(params)>

        flat0 = QMPSRL.get_qmps_params()
        function L(p)
            QMPSRL.set_qmps_params!(p)
            return dot(dir, QMPSRL.qmps_feature(env.state_id))
        end

        ε = 1e-5
        idxs = round.(Int, range(1, length(flat0); length=25))
        max_err = 0.0
        for k in idxs
            p_plus  = copy(flat0); p_plus[k]  += ε
            p_minus = copy(flat0); p_minus[k] -= ε
            g_num = (L(p_plus) - L(p_minus)) / (2ε)
            max_err = max(max_err, abs(gZ[k] - g_num))
        end
        QMPSRL.set_qmps_params!(flat0)
        @test max_err < 1e-4
        @info "Zygote-vs-FD max error on 25 spot-checks" max_err
    end

    @testset "Param round-trip get/set" begin
        env_id = QMPSRL.new_env(42).state_id
        feat_before = QMPSRL.qmps_feature(env_id)

        flat = QMPSRL.get_qmps_params()
        QMPSRL.set_qmps_params!(flat)

        feat_after = QMPSRL.qmps_feature(env_id)
        @test all(isapprox.(feat_before, feat_after; atol=1e-12))
    end

    @testset "Batched feature matches single feature" begin
        ids = [QMPSRL.new_env(s).state_id for s in (1, 2, 3)]
        feats_single = reduce(hcat, [QMPSRL.qmps_feature(id) for id in ids])
        feats_batch  = QMPSRL.qmps_feature_batch(ids)
        @test size(feats_batch) == (QMPSRL.D_F, length(ids))
        @test isapprox(feats_batch, feats_single; atol=1e-12)
    end
end

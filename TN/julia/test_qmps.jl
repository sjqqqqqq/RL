# Unit tests for QMPSRL.
# Run from TN/julia/ with: julia --project=. test_qmps.jl

include("QMPSRL.jl")
using .QMPSRL
using Test
using LinearAlgebra
using Random
using ITensors, ITensorMPS

@testset "QMPSRL" begin

    @testset "Target state = |↑^N⟩" begin
        # H = -ΣZ has unique ground state |↑⟩^N with energy -N.
        @test isapprox(QMPSRL.TARGET_ENERGY, -float(QMPSRL.N); atol=1e-8)
        @test isapprox(norm(QMPSRL.TARGET_STATE), 1.0; atol=1e-10)

        ψ_up = MPS(QMPSRL.SITES, ["Up" for _ in 1:QMPSRL.N])
        @test isapprox(abs2(inner(ψ_up, QMPSRL.TARGET_STATE)), 1.0; atol=1e-8)
    end

    if QMPSRL.N == 4
        @testset "TFIM ground-state energy at (J=±1, gx=1.05, gz=0) — N=4 ED check" begin
            ψ, E = QMPSRL.dmrg_ground(QMPSRL.J_INIT, 1.05, 0.0)
            H = QMPSRL.tlfi_mpo(QMPSRL.J_INIT, 1.05, 0.0)
            # Build dense Hamiltonian by contracting the MPO and reshape to 16×16.
            E_dense = let
                T = H[1]
                for i in 2:QMPSRL.N
                    T = T * H[i]
                end
                sites_in  = [QMPSRL.SITES[i] for i in 1:QMPSRL.N]
                sites_out = [prime(QMPSRL.SITES[i]) for i in 1:QMPSRL.N]
                M = Array(T, sites_in..., sites_out...)
                M = reshape(M, 2^QMPSRL.N, 2^QMPSRL.N)
                real(eigvals(Hermitian((M + M') / 2))[1])
            end
            @test isapprox(E, E_dense; atol=1e-6)
            @info "TFIM (J, gx=1.05) ground energy" J=QMPSRL.J_INIT E E_dense
        end
    end

    @testset "Env norm preservation under 20 random actions" begin
        env = QMPSRL.new_env(7)
        ψ = QMPSRL.get_state(env.state_id)
        @test isapprox(norm(ψ), 1.0; atol=1e-10)

        rng = MersenneTwister(123)
        for _ in 1:20
            a = rand(rng, 0:QMPSRL.N_ACTIONS - 1)
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
            dt = QMPSRL.duration_for(a)
            @test dt == (a % 2 == 0 ? QMPSRL.DT_PLUS : QMPSRL.DT_MINUS)
        end
    end

    @testset "QMPS feature shape & finiteness" begin
        env  = QMPSRL.new_env(11)
        feat = QMPSRL.qmps_feature(env.state_id)
        @test length(feat) == QMPSRL.D_F
        @test all(isfinite, feat)
    end

    if QMPSRL.N == 4
        @testset "MPS-on-MPS overlap vs brute-force dense (N=4 sanity)" begin
            # Independent verification of the QMPS contraction at small N: build
            # the dense statevector explicitly, contract by element-wise loop.
            env = QMPSRL.new_env(99)
            ψ   = QMPSRL.get_state(env.state_id)

            T = ψ[1]
            for i in 2:QMPSRL.N
                T = T * ψ[i]
            end
            ψ_dense = Array(T, QMPSRL.SITES...)        # (2,2,2,2)
            ψc_dense = conj(ψ_dense)

            chunks = QMPSRL.QMPS_CHUNKS[]
            qmps   = [reshape(chunks[q], QMPSRL.QMPS_SHAPES[q])
                      for q in 1:length(chunks)]

            B1 = size(qmps[1], 2); B2 = size(qmps[2], 3)
            B3 = size(qmps[3], 3); B4 = size(qmps[4], 3)
            F  = size(qmps[3], 2)
            o_naive = zeros(ComplexF64, F)
            @inbounds for s1 in 1:2, s2 in 1:2, s3 in 1:2, s4 in 1:2
                psi_amp = ψc_dense[s1, s2, s3, s4]
                for b1 in 1:B1, b2 in 1:B2, b3 in 1:B3, b4 in 1:B4
                    coef = psi_amp *
                           qmps[1][s1, b1] *
                           qmps[2][s2, b1, b2] *
                           qmps[4][s3, b3, b4] *
                           qmps[5][s4, b4]
                    @inbounds for f in 1:F
                        o_naive[f] += coef * qmps[3][b2, f, b3]
                    end
                end
            end
            feat_naive = [log(abs2(c) + 1e-16) / QMPSRL.N for c in o_naive]
            feat_fast  = QMPSRL.qmps_feature(env.state_id)
            @test isapprox(feat_naive, feat_fast; atol=1e-10)
            @info "MPS-on-MPS vs brute force" max_err=maximum(abs.(feat_naive - feat_fast))
        end
    end

    @testset "Param round-trip get/set" begin
        env_id = QMPSRL.new_env(42).state_id
        feat_before = QMPSRL.qmps_feature(env_id)

        flat = QMPSRL.get_qmps_params()
        QMPSRL.set_qmps_params!(flat)

        feat_after = QMPSRL.qmps_feature(env_id)
        @test all(isapprox.(feat_before, feat_after; atol=1e-12))
    end

    @testset "Half-chain entropy of target state = 0" begin
        # |↑^N⟩ is a product state: zero entanglement at any bond.
        @test QMPSRL.half_chain_entropy(QMPSRL.TARGET_STATE) ≈ 0.0  atol=1e-10
    end
end

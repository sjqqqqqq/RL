module QMPSRL

using ITensors, ITensorMPS
using LinearAlgebra
using Random
using Zygote

# ===================================================================
# Constants / problem definition — Paper Study A (Metz & Bukov 2022/2023)
# ===================================================================

const N             = 4                # number of spins
const SITES         = siteinds("S=1/2", N)
const D_QSTATE      = 4                # quantum-state MPS bond dimension (paper D=4)
const D_F           = 32               # QMPS feature dimension (paper n_feat=32)

# QMPS bond structure for N=4, d_phys=2, d_bond_cap=4 (uniform=False, paper):
# bond after site i: min(2^i, 2^(N-i), d_bond_cap) → [2, 4, 4, 2]
const CHI_BONDS     = (2, 4, 4, 2)

# Per-episode initial state: TFIM ground state at (J=1, gx∈U[gx_lo, gx_hi], gz=0).
const J_INIT        = 1.0
const GX_INIT_LO    = 1.0
const GX_INIT_HI    = 1.1
const GZ_INIT       = 0.0

# Target state: ground state of (J=0, gx=0, gz=1), i.e. H = -ΣZ_i → |↑↑↑↑⟩.
const J_TGT, GX_TGT, GZ_TGT = (0.0, 0.0, 1.0)

# Action set (paper order; the parity-based step-size selector requires this):
#   index 0  gx,  +h_max, 1-site,  even → δt+
#   index 1  ngx, -h_max, 1-site,  odd  → δt-
#   index 2  gy,  +h_max, 1-site,  even → δt+
#   index 3  ngy, -h_max, 1-site,  odd  → δt-
#   index 4  gz,  +h_max, 1-site,  even → δt+
#   index 5  ngz, -h_max, 1-site,  odd  → δt-
#   index 6  Jx,  +h_max, 2-site,  even → δt+
#   index 7  nJx, -h_max, 2-site,  odd  → δt-
#   index 8  Jy,  +h_max, 2-site,  even → δt+
#   index 9  nJy, -h_max, 2-site,  odd  → δt-
#   index 10 Jz,  +h_max, 2-site,  even → δt+
#   index 11 nJz, -h_max, 2-site,  odd  → δt-
#
# Paper time evolution per action: exp(-i·duration·op) with op = -h_max·G_total,
# so the effective gate is exp(+i·duration·h_max·G_total). With h_max ∈ {±1},
# the sign of the rotation comes from h_max and the magnitude from duration.
const H_MAX         = 1.0
const DT_PLUS       = π / 12           # 0.5π/6,    even index
const DT_MINUS      = π / 17           # 0.5π/8.5,  odd  index

const ACTIONS = (
    ("X",  +H_MAX, :one),  ("X",  -H_MAX, :one),
    ("Y",  +H_MAX, :one),  ("Y",  -H_MAX, :one),
    ("Z",  +H_MAX, :one),  ("Z",  -H_MAX, :one),
    ("XX", +H_MAX, :two),  ("XX", -H_MAX, :two),
    ("YY", +H_MAX, :two),  ("YY", -H_MAX, :two),
    ("ZZ", +H_MAX, :two),  ("ZZ", -H_MAX, :two),
)
const N_ACTIONS     = length(ACTIONS)

# Termination threshold (paper: reward > log(1 - 0.005), reward = log(F)/N):
#   F > exp(N · log(1 - 0.005)) = 0.995^N = 0.9801 for N=4.
const F_THRESHOLD   = 0.995 ^ N
const N_STEPS_MAX   = 50

# ===================================================================
# State registry — Python keeps integer handles, Julia owns the MPS.
# ===================================================================

struct RegEntry
    ψ     :: MPS
    dense :: Array{ComplexF64, N}      # (s1,…,sN) at canonical site order
end

const _registry = Dict{Int, RegEntry}()
const _next_id  = Ref{Int}(0)

function _register!(ψ::MPS)
    _next_id[] += 1
    id = _next_id[]
    _registry[id] = RegEntry(ψ, _state_to_dense(ψ))
    return id
end

get_state(id::Int)       = _registry[id].ψ
get_dense(id::Int)       = _registry[id].dense
forget_state!(id::Int)   = (delete!(_registry, id); nothing)
registry_size()          = length(_registry)
clear_registry!()        = (empty!(_registry); _next_id[] = 0; nothing)

# Convert an MPS to a dense statevector of shape (2,2,…,2). Cheap at N=4.
function _state_to_dense(ψ::MPS)
    T = ψ[1]
    for i in 2:N
        T = T * ψ[i]
    end
    return Array(T, SITES...)          # shape (2,2,2,2) for N=4
end

# ===================================================================
# TLFI Hamiltonian + DMRG ground-state helper
# ===================================================================
# H = -J·Σ Z_i Z_{i+1} - gx·Σ X_i - gz·Σ Z_i   (paper sign convention)

function tlfi_mpo(J::Real, gx::Real, gz::Real)
    os = OpSum()
    for i in 1:N
        if gx != 0
            os += -gx, "X", i
        end
        if gz != 0
            os += -gz, "Z", i
        end
    end
    for i in 1:N-1
        if J != 0
            os += -J, "Z", i, "Z", i+1
        end
    end
    return MPO(os, SITES)
end

function dmrg_ground(J::Real, gx::Real, gz::Real; rng::AbstractRNG = Random.default_rng())
    H  = tlfi_mpo(J, gx, gz)
    ψ0 = random_mps(rng, SITES; linkdims = 2)
    sweeps_maxdim = min.([10, 20, 40, 40, 40], D_QSTATE)  # cap at paper's D=4
    E, ψ = dmrg(H, ψ0;
                nsweeps     = 10,
                maxdim      = sweeps_maxdim,
                cutoff      = 1e-12,
                outputlevel = 0)
    return ψ, E
end

# ===================================================================
# Target state — built once at module load.
# ===================================================================

const TARGET_STATE, TARGET_ENERGY = dmrg_ground(J_TGT, GX_TGT, GZ_TGT)

# ===================================================================
# Per-episode initial state sampler.
# ===================================================================
# Paper Study A: random TFIM ground state with gx ∈ U[1.0, 1.1].

function _random_initial_state(rng::AbstractRNG)
    gx = GX_INIT_LO + (GX_INIT_HI - GX_INIT_LO) * rand(rng)
    ψ, _ = dmrg_ground(J_INIT, gx, GZ_INIT; rng = rng)
    return ψ
end

# ===================================================================
# Action application
# ===================================================================
# All gates that appear in a single action commute pairwise:
#   - For Σ G_i (single site), the G_i act on disjoint sites.
#   - For Σ G_i G_{i+1}, adjacent bonds share one site but the same Pauli on
#     that site commutes with itself.
# So exp(iα ΣG) factorises exactly as a product of 1- or 2-site gates.

decode_action(a::Int) = ACTIONS[a + 1]   # Python sends 0-indexed actions

# Per-action step size: even index → δt+, odd → δt-.
duration_for(a::Int) = (a % 2 == 0) ? DT_PLUS : DT_MINUS

function _build_gates(gen::String, h_max::Float64, kind::Symbol, dt::Float64)
    # Effective rotation angle (paper convention: op = -h_max·G, evolution
    # exp(-i·dt·op) = exp(+i·dt·h_max·G)).
    angle = h_max * dt
    gates = ITensor[]
    if kind == :one
        for i in 1:N
            G  = op(gen, SITES[i])
            Id = op("Id", SITES[i])
            push!(gates, cos(angle) * Id + im * sin(angle) * G)
        end
    else  # :two
        g1 = string(gen[1])              # "XX" → "X", etc.
        for i in 1:N-1
            G1  = op(g1, SITES[i])
            G2  = op(g1, SITES[i+1])
            Id1 = op("Id", SITES[i])
            Id2 = op("Id", SITES[i+1])
            push!(gates, cos(angle) * (Id1 * Id2) + im * sin(angle) * (G1 * G2))
        end
    end
    return gates
end

function apply_action(ψ::MPS, action::Int)
    gen, h_max, kind = decode_action(action)
    dt    = duration_for(action)
    gates = _build_gates(gen, h_max, kind, dt)
    ψ′    = apply(gates, ψ; cutoff = 1e-12, maxdim = D_QSTATE)
    normalize!(ψ′)
    return ψ′
end

# ===================================================================
# Env
# ===================================================================

mutable struct Env
    state_id    :: Int
    t           :: Int
    f_threshold :: Float64
    n_steps_max :: Int
end

function new_env(seed::Int = 0;
                 f_threshold::Float64 = F_THRESHOLD,
                 n_steps_max::Int     = N_STEPS_MAX)
    rng = MersenneTwister(seed)
    ψ   = _random_initial_state(rng)
    id  = _register!(ψ)
    return Env(id, 0, f_threshold, n_steps_max)
end

function reset!(env::Env, seed::Int)
    rng = MersenneTwister(seed)
    ψ   = _random_initial_state(rng)
    env.state_id    = _register!(ψ)
    env.t           = 0
    return env.state_id
end

fidelity_id(state_id::Int) = abs2(inner(TARGET_STATE, get_state(state_id)))
fidelity(env::Env)         = fidelity_id(env.state_id)
reward_id(state_id::Int)   = log(max(fidelity_id(state_id), 1e-16)) / N
reward(env::Env)           = reward_id(env.state_id)

# Apply an action; returns (next_state_id, reward, done, fidelity).
function step!(env::Env, action::Int)
    ψ       = get_state(env.state_id)
    ψ_next  = apply_action(ψ, action)
    next_id = _register!(ψ_next)
    env.state_id = next_id
    env.t       += 1
    f    = fidelity_id(next_id)
    r    = log(max(f, 1e-16)) / N
    done = (f ≥ env.f_threshold) || (env.t ≥ env.n_steps_max)
    return next_id, r, done, f
end

# ===================================================================
# QMPS (trainable agent)
# ===================================================================
# 5-tensor "label-MPS" architecture matching the paper for N=4:
#
#   T1 :  (s1, B1)              physical site 1 (boundary)        — (2, 2)
#   T2 :  (s2, B1, B2)          physical site 2                   — (2, 2, 4)
#   T3 :  (B2, F, B3)           feature-only center (NO phys leg) — (4, 32, 4)
#   T4 :  (s3, B3, B4)          physical site 3                   — (2, 4, 2)
#   T5 :  (s4, B4)              physical site 4 (boundary)        — (2, 2)
#
# Overlap (raw, complex):
#   o[F] = Σ conj(ψ)[s1,s2,s3,s4] · T1[s1,B1] · T2[s2,B1,B2]
#                                · T3[B2,F,B3] · T4[s3,B3,B4] · T5[s4,B4]
#
# Feature vector returned to the NN head: feat[F] = log(|o[F]|² + ε) / N.

const QMPS_SHAPES = [
    (2, CHI_BONDS[1]),                       # T1
    (2, CHI_BONDS[1], CHI_BONDS[2]),         # T2
    (CHI_BONDS[2], D_F, CHI_BONDS[3]),       # T3 center (no physical leg)
    (2, CHI_BONDS[3], CHI_BONDS[4]),         # T4
    (2, CHI_BONDS[4]),                       # T5
]
@assert length(QMPS_SHAPES) == N + 1

const QMPS_NUMEL           = [prod(s) for s in QMPS_SHAPES]
const QMPS_NPARAMS_COMPLEX = sum(QMPS_NUMEL)
const QMPS_NPARAMS_REAL    = 2 * QMPS_NPARAMS_COMPLEX

# ----- initialization (paper: identity-near real part + N(0, 0.5) noise) -----

function _qmps_real_init(shape::Tuple, kind::Symbol)
    arr = zeros(Float64, shape)
    if kind == :boundary
        # shape (s, B): paper places the unit weight at bond index 0 (Julia 1)
        # for both physical states.
        arr[:, 1] .= 1.0
    elseif kind == :bulk
        # shape (s, B1, B2): identity matrix in the two bond legs, broadcast
        # over the physical leg.
        m = Matrix{Float64}(I, shape[2], shape[3])
        @inbounds for s in 1:shape[1]
            arr[s, :, :] .= m
        end
    elseif kind == :center
        # shape (B1, F, B2): identity matrix in the two bond legs, broadcast
        # over the feature leg.
        m = Matrix{Float64}(I, shape[1], shape[3])
        @inbounds for f in 1:shape[2]
            arr[:, f, :] .= m
        end
    else
        error("unknown init kind $kind")
    end
    return arr
end

const _QMPS_KINDS = (:boundary, :bulk, :center, :bulk, :boundary)

function _init_qmps(seed::Int = 42; std::Float64 = 0.5)
    rng = MersenneTwister(seed)
    chunks = Vector{ComplexF64}[]
    for (shape, kind) in zip(QMPS_SHAPES, _QMPS_KINDS)
        re = _qmps_real_init(shape, kind) .+ std .* randn(rng, shape)
        im_ = std .* randn(rng, shape)
        push!(chunks, vec(complex.(re, im_)))
    end
    return chunks
end

const QMPS_CHUNKS = Ref(_init_qmps())

# ----- Zygote-friendly contraction -----

# Compute the (D_F,)-vector of complex overlaps o[F] given conj(ψ_dense) and
# the five complex tensors. Only reshape / permutedims / matmul — Zygote-safe.
function _qmps_overlap_dense(ψc::AbstractArray{<:Complex, 4},
                             T1::AbstractArray{<:Complex, 2},
                             T2::AbstractArray{<:Complex, 3},
                             T3::AbstractArray{<:Complex, 3},
                             T4::AbstractArray{<:Complex, 3},
                             T5::AbstractArray{<:Complex, 2})
    B1 = size(T1, 2); B2 = size(T2, 3); B3 = size(T3, 3); B4 = size(T4, 3)
    F  = size(T3, 2)

    # Step 1: ψc[s1,s2,s3,s4] · T1[s1,B1] → M1[B1, s2*s3*s4]
    ψc_mat = reshape(ψc, 2, 2 * 2 * 2)                                 # (s1, 8)
    T1_T   = permutedims(T1, (2, 1))                                   # (B1, s1)
    M1_mat = T1_T * ψc_mat                                             # (B1, 8)
    M1     = reshape(M1_mat, B1, 2, 2, 2)                              # (B1, s2, s3, s4)

    # Step 2: contract (s2, B1) with T2[s2,B1,B2] → M2[B2, s3*s4]
    M1_p   = permutedims(M1, (2, 1, 3, 4))                             # (s2, B1, s3, s4)
    M1_mat2 = reshape(M1_p, 2 * B1, 2 * 2)                             # (s2*B1, s3*s4)
    T2_mat = reshape(T2, 2 * B1, B2)                                   # (s2*B1, B2)
    T2_T   = permutedims(T2_mat, (2, 1))                               # (B2, s2*B1)
    M2_mat = T2_T * M1_mat2                                            # (B2, 4)
    M2     = reshape(M2_mat, B2, 2, 2)                                 # (B2, s3, s4)

    # Step 3: contract B2 with T3[B2,F,B3] → M3[F, B3, s3, s4]
    T3_p   = permutedims(T3, (2, 3, 1))                                # (F, B3, B2)
    T3_mat = reshape(T3_p, F * B3, B2)                                 # (F*B3, B2)
    M2_mat3 = reshape(M2, B2, 2 * 2)                                   # (B2, s3*s4)
    M3_mat = T3_mat * M2_mat3                                          # (F*B3, s3*s4)
    M3     = reshape(M3_mat, F, B3, 2, 2)                              # (F, B3, s3, s4)

    # Step 4: contract (s3, B3) with T4[s3,B3,B4] → M4[B4, F, s4]
    T4_p   = permutedims(T4, (3, 1, 2))                                # (B4, s3, B3)
    T4_mat = reshape(T4_p, B4, 2 * B3)                                 # (B4, s3*B3)
    M3_p   = permutedims(M3, (3, 2, 1, 4))                             # (s3, B3, F, s4)
    M3_mat4 = reshape(M3_p, 2 * B3, F * 2)                             # (s3*B3, F*s4)
    M4_mat = T4_mat * M3_mat4                                          # (B4, F*s4)
    M4     = reshape(M4_mat, B4, F, 2)                                 # (B4, F, s4)

    # Step 5: contract (s4, B4) with T5[s4,B4] → o[F]
    T5_p   = permutedims(T5, (2, 1))                                   # (B4, s4)
    T5_vec = reshape(T5_p, B4 * 2)                                     # (B4*s4,)
    M4_p   = permutedims(M4, (1, 3, 2))                                # (B4, s4, F)
    M4_mat5 = reshape(M4_p, B4 * 2, F)                                 # (B4*s4, F)
    o      = transpose(M4_mat5) * T5_vec                               # (F,) complex
    return o
end

# Feature vector: log(|o|² + ε) / N
function _qmps_feature(ψc::AbstractArray{<:Complex, 4},
                       chunks::Vector{Vector{ComplexF64}})
    T1 = reshape(chunks[1], QMPS_SHAPES[1])
    T2 = reshape(chunks[2], QMPS_SHAPES[2])
    T3 = reshape(chunks[3], QMPS_SHAPES[3])
    T4 = reshape(chunks[4], QMPS_SHAPES[4])
    T5 = reshape(chunks[5], QMPS_SHAPES[5])
    o  = _qmps_overlap_dense(ψc, T1, T2, T3, T4, T5)
    return [log(abs2(c) + 1e-16) / N for c in o]
end

# ----- parameter ↔ flat-real conversion (for torch.nn.Parameter) -----

function _chunks_to_real_flat(chunks::Vector{Vector{ComplexF64}})
    out = zeros(Float64, QMPS_NPARAMS_REAL)
    k = 1
    @inbounds for c in chunks
        for z in c
            out[k]     = real(z)
            out[k + 1] = imag(z)
            k += 2
        end
    end
    return out
end

function _real_flat_to_chunks(flat::AbstractVector)
    # Zygote-friendly: no in-place mutation, no push!.
    # Layout: flat[2k-1], flat[2k] = (real, imag) of complex param k.
    offsets = cumsum([0; QMPS_NUMEL])
    return [
        [complex(flat[2 * (offsets[i] + j) - 1], flat[2 * (offsets[i] + j)])
         for j in 1:QMPS_NUMEL[i]]
        for i in 1:length(QMPS_NUMEL)
    ]
end

# ===================================================================
# Python-facing API
# ===================================================================

n_qmps_params_real() = QMPS_NPARAMS_REAL
d_f()                = D_F
n_actions()          = N_ACTIONS

get_qmps_params() = _chunks_to_real_flat(QMPS_CHUNKS[])

function set_qmps_params!(flat::AbstractVector)
    # Accept any AbstractVector (e.g. PyArray from juliacall, Vector{Float64}).
    QMPS_CHUNKS[] = _real_flat_to_chunks(collect(Float64, flat))
    return nothing
end

# Single forward (no grad): returns Vector{Float64} of length D_F.
function qmps_feature(state_id::Int)
    ψc = conj(get_dense(state_id))
    return _qmps_feature(ψc, QMPS_CHUNKS[])
end

# Forward + VJP. Returns (feat::Vector{Float64}, grad_fn).
# grad_fn(g::Vector{Float64}) returns ∇_params(sum(g .* feat)) as a flat real
# vector of length QMPS_NPARAMS_REAL.
function qmps_feature_and_vjp(state_id::Int)
    ψc   = conj(get_dense(state_id))
    flat = _chunks_to_real_flat(QMPS_CHUNKS[])
    feat_f = real_flat -> _qmps_feature(ψc, _real_flat_to_chunks(real_flat))
    feat, pb = Zygote.pullback(feat_f, flat)
    grad_fn = g -> begin
        gflat, = pb(Vector{Float64}(g))
        return gflat
    end
    return feat, grad_fn
end

# Batched forward (no grad): returns Matrix{Float64} of shape (D_F, batch).
function qmps_feature_batch(state_ids::AbstractVector)
    n = length(state_ids)
    out = Matrix{Float64}(undef, D_F, n)
    chunks = QMPS_CHUNKS[]
    @inbounds for (i, id) in enumerate(state_ids)
        ψc = conj(get_dense(Int(id)))
        out[:, i] = _qmps_feature(ψc, chunks)
    end
    return out
end

# Batched forward + VJP. Returns (feats::Matrix{Float64} of shape (D_F, batch),
# grad_fn). grad_fn(G::Matrix{Float64}) returns ∇_params(sum(G .* feats)) summed
# over the batch as a flat real vector of length QMPS_NPARAMS_REAL.
function qmps_feature_and_vjp_batch(state_ids::AbstractVector)
    ψc_list = [conj(get_dense(Int(id))) for id in state_ids]
    flat = _chunks_to_real_flat(QMPS_CHUNKS[])
    feat_f = real_flat -> begin
        chunks = _real_flat_to_chunks(real_flat)
        cols = [_qmps_feature(ψc, chunks) for ψc in ψc_list]
        return reduce(hcat, cols)             # (D_F, batch)
    end
    feats, pb = Zygote.pullback(feat_f, flat)
    grad_fn = G -> begin
        gflat, = pb(Matrix{Float64}(G))
        return gflat
    end
    return feats, grad_fn
end

end # module QMPSRL

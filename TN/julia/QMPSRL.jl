module QMPSRL

using ITensors, ITensorMPS
using LinearAlgebra
using Random
using Zygote

# ===================================================================
# Constants / problem definition
# ===================================================================

const N = 4                     # number of spins
const SITES = siteinds("S=1/2", N)
const CHI_Q = 4                 # QMPS bond dimension
const D_F   = 8                 # feature dimension (dangling leg)
const CENTRAL_SITE = N ÷ 2      # site that carries the feature leg

# Action set (paper Eq. 2): 6 generators × 2 signs × 2 step sizes = 24 actions
const GENERATORS = ["X", "Y", "Z", "XX", "YY", "ZZ"]
const STEP_SIZES = [π/8, π/16]   # δt₊, δt₋
const SIGNS      = [+1.0, -1.0]
const N_ACTIONS  = length(GENERATORS) * length(SIGNS) * length(STEP_SIZES)

# Index layout: a-1 = ((gen_idx-1)*nSign + (sign_idx-1))*nStep + (step_idx-1)
function decode_action(a::Int)
    a0 = a - 1
    nStep = length(STEP_SIZES)
    nSign = length(SIGNS)
    step_idx = a0 % nStep + 1
    sign_idx = (a0 ÷ nStep) % nSign + 1
    gen_idx  = a0 ÷ (nStep * nSign) + 1
    return GENERATORS[gen_idx], SIGNS[sign_idx], STEP_SIZES[step_idx]
end

# ===================================================================
# State registry — Python keeps integer handles, Julia owns the MPS
# ===================================================================

const _registry = Dict{Int, MPS}()
const _next_id  = Ref{Int}(0)

function _register!(ψ::MPS)
    _next_id[] += 1
    id = _next_id[]
    _registry[id] = ψ
    return id
end

get_state(id::Int) = _registry[id]
forget_state!(id::Int) = (delete!(_registry, id); nothing)
registry_size() = length(_registry)

# ===================================================================
# Target state: DMRG ground state of H = -Σ X_i
# ===================================================================

function _build_target()
    os = OpSum()
    for i in 1:N
        os += -1.0, "X", i
    end
    H = MPO(os, SITES)
    ψ0 = random_mps(SITES; linkdims=2)
    E, ψ★ = dmrg(H, ψ0;
                 nsweeps = 10,
                 maxdim  = [10, 20, 40, 40, 40],
                 cutoff  = 1e-12,
                 outputlevel = 0)
    return ψ★, E
end

const TARGET_STATE, TARGET_ENERGY = _build_target()

# ===================================================================
# Action application
# ===================================================================
# All gates that appear in a single action commute pairwise:
#   - For Σ G_i (single site), the G_i act on disjoint sites.
#   - For Σ G_i G_{i+1}, adjacent bonds share one site but the same Pauli on
#     that site commutes with itself.
# Therefore exp(iα ΣG) factorises exactly as a product of 1- or 2-site gates.

function _build_gates(gen::String, angle::Float64)
    gates = ITensor[]
    if length(gen) == 1
        for i in 1:N
            G  = op(gen, SITES[i])
            Id = op("Id", SITES[i])
            push!(gates, cos(angle) * Id + im * sin(angle) * G)
        end
    else
        g1 = string(gen[1])
        for i in 1:N-1
            G1  = op(g1, SITES[i])
            G2  = op(g1, SITES[i+1])
            Id1 = op("Id", SITES[i])
            Id2 = op("Id", SITES[i+1])
            GG  = G1 * G2
            II  = Id1 * Id2
            push!(gates, cos(angle) * II + im * sin(angle) * GG)
        end
    end
    return gates
end

function apply_action(ψ::MPS, action::Int)
    gen, sgn, dt = decode_action(action)
    angle = sgn * dt
    gates = _build_gates(gen, angle)
    ψ′ = apply(gates, ψ; cutoff=1e-12, maxdim=16)
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

# Paper (Metz & Bukov 2023, p. 782): 25% random polarized product states,
# 75% reflection-symmetric states drawn from the full 2^N-dim Hilbert space
# by sampling wave-function amplitudes from a (complex) normal distribution
# and projecting onto the +1 eigenspace of spatial reflection.
function _polarized_product_state(rng::AbstractRNG)
    ψ = MPS(SITES, ["Up" for _ in 1:N])
    for i in 1:N
        θ = 2π * rand(rng)
        φ = π  * rand(rng)
        Rz = cos(θ/2) * op("Id", SITES[i]) - im * sin(θ/2) * op("Z", SITES[i])
        Ry = cos(φ/2) * op("Id", SITES[i]) - im * sin(φ/2) * op("Y", SITES[i])
        ψ = apply([Rz, Ry], ψ; cutoff=1e-12)
    end
    normalize!(ψ)
    return ψ
end

function _reflection_symmetric_haar_state(rng::AbstractRNG)
    amps = randn(rng, ComplexF64, ntuple(_ -> 2, N))
    # Reflection R: |s_1 ... s_N⟩ ↔ |s_N ... s_1⟩. Project onto +1 eigenspace.
    amps .+= permutedims(amps, ntuple(i -> N - i + 1, N))
    T = ITensor(amps, SITES...)
    ψ = MPS(T, SITES; cutoff=1e-12)
    normalize!(ψ)
    return ψ
end

function _random_initial_state(rng::AbstractRNG)
    if rand(rng) < 0.25
        return _polarized_product_state(rng)
    else
        return _reflection_symmetric_haar_state(rng)
    end
end

function new_env(seed::Int = 0;
                 f_threshold::Float64 = 0.85,
                 n_steps_max::Int     = 50)
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

function fidelity_id(state_id::Int)
    ψ = get_state(state_id)
    return abs2(inner(TARGET_STATE, ψ))
end

fidelity(env::Env) = fidelity_id(env.state_id)
reward_id(state_id::Int) = log(max(fidelity_id(state_id), 1e-16)) / N
reward(env::Env)   = reward_id(env.state_id)

# Apply an action; returns (next_state_id, reward, done, fidelity).
function step!(env::Env, action::Int)
    ψ      = get_state(env.state_id)
    ψ_next = apply_action(ψ, action)
    next_id = _register!(ψ_next)
    env.state_id = next_id
    env.t += 1
    f = fidelity_id(next_id)
    r = log(max(f, 1e-16)) / N
    done = (f ≥ env.f_threshold) || (env.t ≥ env.n_steps_max)
    return next_id, r, done, f
end

# ===================================================================
# QMPS (trainable agent)
# ===================================================================
# The QMPS tensors live as plain Julia arrays (not ITensors): we want
# Zygote to differentiate the contraction wrt every entry, and ITensors'
# Zygote support doesn't cover `array(::ITensor, ::Index)` (which we'd need
# to convert the d_f-dim output back to a Vector). For N=4 the dense
# statevector is only 16 entries, so we contract the QMPS against the
# dense |ψ⟩ vector directly.
#
# QMPS tensor shapes (CENTRAL_SITE = N÷2 = 2):
#   T1 :  (s1, L1)              shape (2, χ)
#   T2 :  (s2, L1, L2, F)       shape (2, χ, χ, d_f)   ← central, feature leg
#   T3 :  (s3, L2, L3)          shape (2, χ, χ)
#   T4 :  (s4, L3)              shape (2, χ)
#
# Overlap:
#   o[F] = Σ conj(ψ)[s1,s2,s3,s4] · T1[s1,L1] · T2[s2,L1,L2,F] · T3[s3,L2,L3] · T4[s4,L3]

const QMPS_SHAPES = [
    (2, CHI_Q),                       # T1
    (2, CHI_Q, CHI_Q, D_F),           # T2 (central)
    (2, CHI_Q, CHI_Q),                # T3
    (2, CHI_Q),                       # T4
]
@assert length(QMPS_SHAPES) == N
@assert CENTRAL_SITE == 2  # hard-coded contraction order below assumes this

const QMPS_NUMEL = [prod(s) for s in QMPS_SHAPES]
const QMPS_NPARAMS_COMPLEX = sum(QMPS_NUMEL)
const QMPS_NPARAMS_REAL    = 2 * QMPS_NPARAMS_COMPLEX

function _init_qmps(seed::Int = 42)
    rng = MersenneTwister(seed)
    chunks = Vector{ComplexF64}[]
    for s in QMPS_SHAPES
        # small-magnitude init so initial overlap with a random product state
        # is finite (avoids log(0))
        c = randn(rng, ComplexF64, prod(s)) ./ sqrt(prod(s))
        push!(chunks, c)
    end
    return chunks
end

const QMPS_CHUNKS = Ref(_init_qmps())

# Convert an MPS to a dense statevector of shape (2,2,2,2). Cheap at N=4.
function _state_to_dense(ψ::MPS)
    T = ψ[1]
    for i in 2:N
        T = T * ψ[i]
    end
    # Order the site indices canonically (SITES[1], SITES[2], ...).
    return Array(T, SITES...)        # shape (2,2,2,2)
end

# Plain-array overlap contraction (Zygote-friendly).
function _qmps_overlap_dense(ψc::AbstractArray{ComplexF64,4},
                             T1::AbstractArray{ComplexF64,2},
                             T2::AbstractArray{ComplexF64,4},
                             T3::AbstractArray{ComplexF64,3},
                             T4::AbstractArray{ComplexF64,2})
    # ψc is already conj(ψ_dense), shape (s1, s2, s3, s4).
    #
    # Step 1: contract s1 with T1 → M1[L1, s2, s3, s4]
    M1 = reshape(reshape(ψc, 2, 8)' * T1, 8, CHI_Q)      # (s2*s3*s4, L1)
    M1 = permutedims(reshape(M1, 2, 2, 2, CHI_Q), (4, 1, 2, 3))  # (L1, s2, s3, s4)

    # Step 2: contract (s2, L1) with T2 → M2[s3, s4, L2, F]
    M1f = reshape(M1, CHI_Q*2, 4)                # (L1*s2, s3*s4)  -- need (s2*L1)
    # actually order matters: T2 has axes (s2, L1, L2, F). Reshape T2 to (s2*L1, L2*F),
    # reshape M1 to (L1*s2, s3*s4) and permute properly.
    M1_s2L1 = permutedims(M1, (2, 1, 3, 4))                 # (s2, L1, s3, s4)
    M1_mat  = reshape(M1_s2L1, 2*CHI_Q, 4)                  # (s2*L1, s3*s4)
    T2_mat  = reshape(T2, 2*CHI_Q, CHI_Q*D_F)               # (s2*L1, L2*F)
    M2_mat  = T2_mat' * M1_mat                              # (L2*F, s3*s4)
    M2      = reshape(M2_mat, CHI_Q, D_F, 2, 2)             # (L2, F, s3, s4)

    # Step 3: contract (s3, L2) with T3 → M3[L3, F, s4]
    M2_s3L2 = permutedims(M2, (3, 1, 2, 4))                 # (s3, L2, F, s4)
    M2_mat3 = reshape(M2_s3L2, 2*CHI_Q, D_F*2)              # (s3*L2, F*s4)
    T3_mat  = reshape(T3, 2*CHI_Q, CHI_Q)                   # (s3*L2, L3)
    M3_mat  = T3_mat' * M2_mat3                             # (L3, F*s4)
    M3      = reshape(M3_mat, CHI_Q, D_F, 2)                # (L3, F, s4)

    # Step 4: contract (s4, L3) with T4 → o[F]
    M3_s4L3 = permutedims(M3, (3, 1, 2))                    # (s4, L3, F)
    M3_mat4 = reshape(M3_s4L3, 2*CHI_Q, D_F)                # (s4*L3, F)
    T4_vec  = reshape(T4, 2*CHI_Q)                          # (s4*L3,)
    o       = (T4_vec' * M3_mat4)                           # row-vector (1, F)
    return vec(collect(o))                                  # Vector{ComplexF64} length D_F
end

# Wrap to take a chunks vector and produce the feature vector.
# Pure-functional in `chunks` for Zygote.
function _qmps_feature(ψc::AbstractArray{ComplexF64,4},
                       chunks::Vector{Vector{ComplexF64}})
    T1 = reshape(chunks[1], QMPS_SHAPES[1])
    T2 = reshape(chunks[2], QMPS_SHAPES[2])
    T3 = reshape(chunks[3], QMPS_SHAPES[3])
    T4 = reshape(chunks[4], QMPS_SHAPES[4])
    o  = _qmps_overlap_dense(ψc, T1, T2, T3, T4)
    return [log(abs2(c) + 1e-16) / N for c in o]
end

# ----- parameter ↔ flat-real conversion (for torch.nn.Parameter) -----

function _chunks_to_real_flat(chunks::Vector{Vector{ComplexF64}})
    out = zeros(Float64, QMPS_NPARAMS_REAL)
    k = 1
    for c in chunks
        for z in c
            out[k]   = real(z)
            out[k+1] = imag(z)
            k += 2
        end
    end
    return out
end

function _real_flat_to_chunks(flat::AbstractVector)
    # Zygote-friendly: no in-place mutation, no push!.
    # Layout: flat[2k-1], flat[2k] = (real, imag) of complex param k.
    # Boundaries: chunk i occupies complex params (offsets[i]+1 .. offsets[i+1]).
    offsets = cumsum([0; QMPS_NUMEL])           # complex-param offsets
    return [
        [complex(flat[2*(offsets[i]+j) - 1], flat[2*(offsets[i]+j)])
         for j in 1:QMPS_NUMEL[i]]
        for i in 1:N
    ]
end

# ----- Python-facing API -----

n_qmps_params_real() = QMPS_NPARAMS_REAL
d_f() = D_F
n_actions() = N_ACTIONS

get_qmps_params() = _chunks_to_real_flat(QMPS_CHUNKS[])

function set_qmps_params!(flat::AbstractVector)
    # Accept any AbstractVector (e.g. PyArray from juliacall, Vector{Float64}, ...).
    QMPS_CHUNKS[] = _real_flat_to_chunks(collect(Float64, flat))
    return nothing
end

# Single forward (no grad): returns Vector{Float64} of length d_f.
function qmps_feature(state_id::Int)
    ψ = get_state(state_id)
    ψc = conj(_state_to_dense(ψ))
    return _qmps_feature(ψc, QMPS_CHUNKS[])
end

# Forward + VJP. Returns (feat::Vector{Float64}, grad_fn).
# grad_fn(g::Vector{Float64}) returns ∇_params(sum(g .* feat)) as a flat real vector
# of length QMPS_NPARAMS_REAL.
function qmps_feature_and_vjp(state_id::Int)
    ψ = get_state(state_id)
    ψc = conj(_state_to_dense(ψ))
    flat = _chunks_to_real_flat(QMPS_CHUNKS[])
    feat_f = real_flat -> _qmps_feature(ψc, _real_flat_to_chunks(real_flat))
    feat, pb = Zygote.pullback(feat_f, flat)
    grad_fn = g -> begin
        gflat, = pb(Vector{Float64}(g))
        return gflat
    end
    return feat, grad_fn
end

# Batched forward (no grad). Returns Matrix{Float64} of shape (D_F, batch).
function qmps_feature_batch(state_ids::AbstractVector)
    n = length(state_ids)
    out = Matrix{Float64}(undef, D_F, n)
    chunks = QMPS_CHUNKS[]
    for (i, id) in enumerate(state_ids)
        ψc = conj(_state_to_dense(get_state(Int(id))))
        out[:, i] = _qmps_feature(ψc, chunks)
    end
    return out
end

# Batched forward + VJP. Returns (feats::Matrix{Float64} of shape (D_F, batch), grad_fn).
# grad_fn(G::Matrix{Float64}) returns ∇_params(sum(G .* feats)) summed over the batch
# as a flat real vector of length QMPS_NPARAMS_REAL.
function qmps_feature_and_vjp_batch(state_ids::AbstractVector)
    # Cache dense conjugated states once (they don't depend on params).
    ψc_list = [conj(_state_to_dense(get_state(Int(id)))) for id in state_ids]
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

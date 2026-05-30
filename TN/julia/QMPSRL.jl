module QMPSRL

using ITensors, ITensorMPS
using LinearAlgebra
using Random

# ===================================================================
# Constants / problem definition
# ===================================================================
# Study B settings (L=32, FM init J=+1, 7 actions, χ_q=32, D_F=72, init norm
# factor=4). All shape-dependent constants below are derived from these — no
# hard-coded 5-tensor structure remains.

const N             = 32                   # chain length (Study B: L=32)
const SITES         = siteinds("S=1/2", N)
const D_QSTATE      = 16                   # quantum-state MPS bond cap
const D_F           = 72                   # QMPS feature dimension
const CHI_Q         = 32                   # QMPS bond cap

# Feature transform: feat = FEATURE_SCALE · log(|o|² + ε) / N. FEATURE_SCALE is
# the reference `scale` (csB = 4.0; `QMPS/dqn/models.py:301` computes
# `out = scale·2·log|o|/L`, and 2·log|o| = log|o|²). This is SEPARATE from
# INIT_NORM_FACTOR (reference `factor`, which scales the MPS-tensor *init*);
# both happen to be 4.0 in Study B, which is easy to conflate.
const FEATURE_SCALE = 4.0

# Per-episode initial state: TFIM ground state at (J=J_INIT, gx∈U[lo,hi], gz=GZ_INIT).
# Study B: FM Ising (J=+1) near-critical near gx ≈ 1, matching the reference
# (`QMPS/environment/env.py:400` builds the init ground state with J=1.0; same
# H sign convention −J·ΣZZ as `tlfi_tn_model.py`).
const J_INIT        = 1.0
const GX_INIT_LO    = 1.0
const GX_INIT_HI    = 1.1
const GZ_INIT       = 0.0

# Target state: ground state of (J_TGT, GX_TGT, GZ_TGT) = (0, 0, 1) → |↑^N⟩.
const J_TGT, GX_TGT, GZ_TGT = (0.0, 0.0, 1.0)

# Action set — paper Study B order (`QMPS/environment/env.py:221` `# CS B`).
# Seven actions: one single-site gy+, then ±gz, then ±Jx, ±Jy.
# The step-size selector keys on action INDEX parity (paper convention,
# `QMPS/environment/env.py:272`), NOT on the sign of h_max:
#   even index → DT_PLUS,  odd index → DT_MINUS
# So with this ordering: gy+ gets DT_PLUS; gz+ gets DT_MINUS; gz− gets DT_PLUS;
# Jx+ gets DT_MINUS; Jx− gets DT_PLUS; Jy+ gets DT_MINUS; Jy− gets DT_PLUS.
# Asymmetric, but it's the paper's actual rule.
const H_MAX         = 1.0
const DT_PLUS       = π / 12
const DT_MINUS      = π / 17

const ACTIONS = (
    ("Y",  +H_MAX, :one),       # 0  gy   even → DT_PLUS
    ("Z",  +H_MAX, :one),       # 1  gz   odd  → DT_MINUS
    ("Z",  -H_MAX, :one),       # 2  ngz  even → DT_PLUS
    ("XX", +H_MAX, :two),       # 3  Jx   odd  → DT_MINUS
    ("XX", -H_MAX, :two),       # 4  nJx  even → DT_PLUS
    ("YY", +H_MAX, :two),       # 5  Jy   odd  → DT_MINUS
    ("YY", -H_MAX, :two),       # 6  nJy  even → DT_PLUS
)
const N_ACTIONS     = length(ACTIONS)

# Single-particle threshold: F^(1/N) ≥ 0.99  ⇔  F ≥ 0.99^N.
const F_THRESHOLD   = 0.99 ^ N
const N_STEPS_MAX   = 50

# Initial-state cache. DMRG every reset is wasteful at large L. The TFIM
# ground-state manifold at gx ∈ U[1.0, 1.1] is thin — 200 pre-sampled states
# is plenty. INIT_CACHE_SEED keeps the cache reproducible.
const INIT_CACHE_SIZE = 200
const INIT_CACHE_SEED = 12345

# ===================================================================
# QMPS bond structure — derived
# ===================================================================
# Center tensor sits between physical sites c-1 and c, where c = N÷2 + 1.
# The QMPS has N+1 tensors total (N physical + 1 feature-only center).
# Bond dim at QMPS bond b ∈ 1..N corresponds to the cut between two physical
# sites; bonds b=c-1 and b=c are both the "center cut" (same dim).

const CENTRAL_TENSOR = N ÷ 2 + 1           # 1-indexed position in qmps[1..N+1]

function _phys_bond_dim(p::Int, L::Int = N, χ::Int = CHI_Q, d::Int = 2)
    # bond dim of the cut between physical sites p and p+1
    return min(d^p, d^(L - p), χ)
end

function _qmps_bond_dim(b::Int, L::Int = N, c::Int = CENTRAL_TENSOR,
                        χ::Int = CHI_Q, d::Int = 2)
    if b < c
        p = b
    elseif b == c
        p = c - 1                           # right side of center: same cut
    else
        p = b - 1
    end
    return _phys_bond_dim(p, L, χ, d)
end

const CHI_BONDS = Tuple(_qmps_bond_dim(b) for b in 1:N)

# QMPS tensor shapes:
#   q=1                       (left boundary, physical):  (s, B[1])
#   q=2..CENTRAL_TENSOR-1     (bulk, physical):           (s, B[q-1], B[q])
#   q=CENTRAL_TENSOR          (center, feature-only):     (B[c-1], F, B[c])
#   q=CENTRAL_TENSOR+1..N     (bulk, physical):           (s, B[q-1], B[q])
#   q=N+1                     (right boundary, physical): (s, B[N])
function _qmps_shape(q::Int)
    if q == 1
        return (2, CHI_BONDS[1])
    elseif q == N + 1
        return (2, CHI_BONDS[N])
    elseif q == CENTRAL_TENSOR
        return (CHI_BONDS[q - 1], D_F, CHI_BONDS[q])
    else
        return (2, CHI_BONDS[q - 1], CHI_BONDS[q])
    end
end

function _qmps_kind(q::Int)
    if q == 1 || q == N + 1
        return :boundary
    elseif q == CENTRAL_TENSOR
        return :center
    else
        return :bulk
    end
end

const QMPS_SHAPES          = [_qmps_shape(q) for q in 1:N+1]
const QMPS_KINDS           = [_qmps_kind(q)  for q in 1:N+1]
const QMPS_NUMEL           = [prod(s) for s in QMPS_SHAPES]
const QMPS_NPARAMS_COMPLEX = sum(QMPS_NUMEL)
const QMPS_NPARAMS_REAL    = 2 * QMPS_NPARAMS_COMPLEX

# ===================================================================
# State registry — Python keeps integer handles, Julia owns the MPS.
# ===================================================================
# We store the state as a plain Vector{Array{ComplexF64,3}} alongside the
# ITensor MPS. The plain arrays feed the Zygote-friendly contraction; the MPS
# is what `apply(...)` consumes for the next step.

struct RegEntry
    ψ       :: MPS
    arrays  :: Vector{Array{ComplexF64, 3}}    # canonical (D_l, s, D_r) per site
end

const _registry = Dict{Int, RegEntry}()
const _next_id  = Ref{Int}(0)

function _register!(ψ::MPS)
    _next_id[] += 1
    id = _next_id[]
    _registry[id] = RegEntry(ψ, _mps_to_arrays(ψ))
    return id
end

get_state(id::Int)       = _registry[id].ψ
get_arrays(id::Int)      = _registry[id].arrays
forget_state!(id::Int)   = (delete!(_registry, id); nothing)
registry_size()          = length(_registry)
clear_registry!()        = (empty!(_registry); _next_id[] = 0; nothing)

# Convert an MPS to canonical (D_l, s, D_r) plain arrays.
function _mps_to_arrays(ψ::MPS)
    L = length(ψ)
    arrs = Vector{Array{ComplexF64, 3}}(undef, L)
    for i in 1:L
        s = siteind(ψ, i)
        if i == 1
            r   = linkind(ψ, 1)
            mat = Array(ψ[i], s, r)                        # (2, D[1])
            arrs[i] = reshape(mat, 1, size(mat, 1), size(mat, 2))
        elseif i == L
            l   = linkind(ψ, L - 1)
            mat = Array(ψ[i], l, s)                        # (D[L-1], 2)
            arrs[i] = reshape(mat, size(mat, 1), size(mat, 2), 1)
        else
            l = linkind(ψ, i - 1)
            r = linkind(ψ, i)
            arrs[i] = Array(ψ[i], l, s, r)                 # (D[i-1], 2, D[i])
        end
    end
    return arrs
end

# Half-chain von Neumann entropy at the bond between sites L÷2 and L÷2+1.
# Used for the cs2 inset diagnostic; doesn't enter training.
function half_chain_entropy(ψ::MPS)
    L = length(ψ)
    ψc = copy(ψ)
    orthogonalize!(ψc, L ÷ 2)
    s   = siteind(ψc, L ÷ 2)
    lhs = L ÷ 2 == 1 ? (s,) : (linkind(ψc, L ÷ 2 - 1), s)
    _, S, _ = svd(ψc[L ÷ 2], lhs)
    σ = diag(Array(S, inds(S)...))
    H = 0.0
    @inbounds for x in σ
        p2 = abs2(x)
        if p2 > 1e-16
            H -= p2 * log(p2)
        end
    end
    return H
end

half_chain_entropy_id(id::Int) = half_chain_entropy(get_state(id))

# ===================================================================
# TLFI Hamiltonian + DMRG ground-state helper
# ===================================================================
# H = −J·Σ Z_i Z_{i+1} − gx·Σ X_i − gz·Σ Z_i  (paper sign convention)

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

function dmrg_ground(J::Real, gx::Real, gz::Real;
                     rng::AbstractRNG = Random.default_rng())
    H  = tlfi_mpo(J, gx, gz)
    ψ0 = random_mps(rng, SITES; linkdims = 2)
    sweeps_maxdim = min.([10, 20, 40, 40, 40], D_QSTATE)
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
# Initial-state cache — DMRG once at module load, sample with replacement.
# ===================================================================

function _build_init_cache()
    rng = MersenneTwister(INIT_CACHE_SEED)
    cache = Vector{MPS}(undef, INIT_CACHE_SIZE)
    for i in 1:INIT_CACHE_SIZE
        gx = GX_INIT_LO + (GX_INIT_HI - GX_INIT_LO) * rand(rng)
        ψ, _ = dmrg_ground(J_INIT, gx, GZ_INIT; rng = rng)
        cache[i] = ψ
    end
    return cache
end

const INIT_CACHE = _build_init_cache()

function _random_initial_state(rng::AbstractRNG)
    return INIT_CACHE[rand(rng, 1:INIT_CACHE_SIZE)]
end

# ===================================================================
# Action application
# ===================================================================

decode_action(a::Int)   = ACTIONS[a + 1]         # 0-indexed action from Python
duration_for(a::Int)    = (a % 2 == 0) ? DT_PLUS : DT_MINUS

function _build_gates(gen::String, h_max::Float64, kind::Symbol, dt::Float64)
    angle = h_max * dt
    gates = ITensor[]
    if kind == :one
        for i in 1:N
            G  = op(gen, SITES[i])
            Id = op("Id", SITES[i])
            push!(gates, cos(angle) * Id + im * sin(angle) * G)
        end
    else  # :two
        g1 = string(gen[1])
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

# Out-of-distribution test envs: run DMRG fresh at an arbitrary gx instead of
# sampling from the U[GX_INIT_LO, GX_INIT_HI] cache. Used by the greedy-sweep
# generalization eval (paper csB panels b/c sweep gx ∈ [1.0, 1.5]).
function new_env_at_gx(gx::Real, seed::Int = 0;
                       f_threshold::Float64 = F_THRESHOLD,
                       n_steps_max::Int     = N_STEPS_MAX)
    rng = MersenneTwister(seed)
    ψ, _ = dmrg_ground(J_INIT, gx, GZ_INIT; rng = rng)
    id   = _register!(ψ)
    return Env(id, 0, f_threshold, n_steps_max)
end

function reset!(env::Env, seed::Int)
    rng = MersenneTwister(seed)
    ψ   = _random_initial_state(rng)
    env.state_id = _register!(ψ)
    env.t        = 0
    return env.state_id
end

fidelity_id(state_id::Int) = abs2(inner(TARGET_STATE, get_state(state_id)))
fidelity(env::Env)         = fidelity_id(env.state_id)
reward_id(state_id::Int)   = log(max(fidelity_id(state_id), 1e-16)) / N
reward(env::Env)           = reward_id(env.state_id)

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
# QMPS initialization
# ===================================================================
# Paper init: identity-near real part + N(0, std²) noise on both re and im,
# then the whole (identity+noise) tensor is divided by INIT_NORM_FACTOR.
#   :boundary tensor (s, B):       arr[:, 1] = 1.0
#   :bulk     tensor (s, B_l, B_r): identity over (B_l, B_r), broadcast over s
#   :center   tensor (B_l, F, B_r): identity over (B_l, B_r), broadcast over F
#
# INIT_NORM_FACTOR = reference `factor` (csB uses factor=4.0; Study A used 1.0).
# Maps to `norm_factor` in QMPS/dqn/models_utils.py:initialize_mps_tensor, which
# returns x / norm_factor. Scaling the init down keeps the L=32 overlap |z| in a
# usable range (each of the N+1 tensors contributes a factor; without it the
# feature saturates against the log's ε floor at large L).
const INIT_NORM_FACTOR = 4.0

function _qmps_real_init(shape::Tuple, kind::Symbol)
    arr = zeros(Float64, shape)
    if kind == :boundary
        arr[:, 1] .= 1.0
    elseif kind == :bulk
        m = Matrix{Float64}(I, shape[2], shape[3])
        @inbounds for s in 1:shape[1]
            arr[s, :, :] .= m
        end
    elseif kind == :center
        m = Matrix{Float64}(I, shape[1], shape[3])
        @inbounds for f in 1:shape[2]
            arr[:, f, :] .= m
        end
    else
        error("unknown init kind $kind")
    end
    return arr
end

function _init_qmps(seed::Int = 42; std::Float64 = 0.5,
                    norm_factor::Float64 = INIT_NORM_FACTOR)
    rng = MersenneTwister(seed)
    chunks = Vector{ComplexF64}[]
    for (shape, kind) in zip(QMPS_SHAPES, QMPS_KINDS)
        re = _qmps_real_init(shape, kind) .+ std .* randn(rng, shape)
        im_ = std .* randn(rng, shape)
        push!(chunks, vec(complex.(re, im_)) ./ norm_factor)
    end
    return chunks
end

const QMPS_CHUNKS = Ref(_init_qmps())

# ===================================================================
# MPS-on-MPS overlap contraction — Zygote-friendly, generic in N
# ===================================================================
# Inputs:
#   ψ_conj : Vector of L tensors, each (D_l, 2, D_r). conj of the state MPS.
#   qmps   : Vector of L+1 tensors, shapes per QMPS_SHAPES, last index is the
#            feature leg at the central tensor.
# Output: Vector{ComplexF64} of length D_F, the (complex) feature overlaps.
#
# Sweep left → right. Track a running left-environment:
#   L_env   :: (B_left, D_left)       before the center
#   L_envF  :: (F, B_left, D_left)    after the center
#
# Each step is a sequence of reshape + permutedims + matmul (no in-place
# mutation, no conjugation; ψ is already conjugated by the caller).

# Takes `chunks` (flat per-tensor ComplexF64 vectors) directly and reshapes at
# point of use *inside the loop* — building a heterogeneous Vector of reshaped
# tensors via a comprehension makes Zygote construct an abstractly-typed
# pullback container that blows up at L=32. A `for` loop with locally-typed
# reshapes is handled by Zygote's stack-based loop adjoint and stays cheap.
function _qmps_overlap_mps(ψ_conj::AbstractVector,
                            chunks::AbstractVector)
    L = length(ψ_conj)
    c = L ÷ 2 + 1
    F = QMPS_SHAPES[c][2]

    # ---- Step 1: q=1, left boundary physical site -----------------------
    T1 = reshape(chunks[1], QMPS_SHAPES[1])                      # (2, B1)
    D1 = size(ψ_conj[1], 3)
    psi1  = reshape(ψ_conj[1], 2, D1)                            # (s, D1)
    L_env = transpose(T1) * psi1                                 # (B1, D1)

    # ---- Bulk left of center: q = 2..c-1 --------------------------------
    for q in 2:(c - 1)
        Tq      = reshape(chunks[q], QMPS_SHAPES[q])             # (s, Bl, Br)
        Bl, Br  = size(Tq, 2), size(Tq, 3)
        Dl, Dr  = size(ψ_conj[q], 1), size(ψ_conj[q], 3)
        psi_mat = reshape(ψ_conj[q], Dl, 2 * Dr)                 # (Dl, s*Dr)
        M_mat   = L_env * psi_mat                                # (Bl, s*Dr)
        qp      = permutedims(Tq, (3, 2, 1))                     # (Br, Bl, s)
        q_mat   = reshape(qp, Br, Bl * 2)                        # (Br, Bl*s)
        Mr      = reshape(M_mat, Bl * 2, Dr)                     # (Bl*s, Dr)
        L_env   = q_mat * Mr                                     # (Br, Dr)
    end

    # ---- Step q = c: center tensor, no physical leg ---------------------
    Tc          = reshape(chunks[c], QMPS_SHAPES[c])             # (Bl, F, Br)
    Blc, _, Brc = size(Tc)
    D_unc       = size(L_env, 2)
    qp     = permutedims(Tc, (2, 3, 1))                          # (F, Br, Bl)
    q_mat  = reshape(qp, F * Brc, Blc)                           # (F*Br, Bl)
    M_mat  = q_mat * L_env                                       # (F*Br, D_unc)
    L_envF = reshape(M_mat, F, Brc, D_unc)                       # (F, Br, D_unc)

    # ---- Bulk right of center + right boundary: q = c+1..L+1 ------------
    for q in (c + 1):(L + 1)
        p      = q - 1                                           # physical site index
        Dl, Dr = size(ψ_conj[p], 1), size(ψ_conj[p], 3)
        Tq     = reshape(chunks[q], QMPS_SHAPES[q])

        if q < L + 1
            Bl, Br  = size(Tq, 2), size(Tq, 3)
            psi_mat = reshape(ψ_conj[p], Dl, 2 * Dr)             # (Dl, s*Dr)
            L_mat   = reshape(L_envF, F * Bl, Dl)                # (F*Bl, Dl)
            M_mat   = L_mat * psi_mat                            # (F*Bl, s*Dr)
            M       = reshape(M_mat, F, Bl, 2, Dr)              # (F, Bl, s, Dr)
            qp      = permutedims(Tq, (3, 2, 1))                 # (Br, Bl, s)
            q_mat   = reshape(qp, Br, Bl * 2)                    # (Br, Bl*s)
            M_p     = permutedims(M, (2, 3, 1, 4))               # (Bl, s, F, Dr)
            Mr      = reshape(M_p, Bl * 2, F * Dr)               # (Bl*s, F*Dr)
            res     = q_mat * Mr                                 # (Br, F*Dr)
            L_envF  = permutedims(reshape(res, Br, F, Dr), (2, 1, 3))  # (F, Br, Dr)
        else
            # right boundary: Tq = (s, Bl); ψ_conj[L]: (Dl, 2, 1)
            Bl      = size(Tq, 2)
            psi_mat = reshape(ψ_conj[p], Dl, 2)                  # (Dl, s)
            L_mat   = reshape(L_envF, F * Bl, Dl)                # (F*Bl, Dl)
            M_mat   = L_mat * psi_mat                            # (F*Bl, s)
            M       = reshape(M_mat, F, Bl * 2)                  # (F, Bl*s)
            qvec    = reshape(permutedims(Tq, (2, 1)), Bl * 2)   # (Bl*s,)
            return M * qvec                                      # (F,)
        end
    end
    error("unreachable")
end

# Feature vector: FEATURE_SCALE · log(|o|² + ε) / N (reference scale·2·log|o|/L).
function _qmps_feature(ψ_conj::AbstractVector,
                       chunks::AbstractVector)
    o = _qmps_overlap_mps(ψ_conj, chunks)
    return FEATURE_SCALE .* log.(abs2.(o) .+ 1e-16) ./ N
end

# ===================================================================
# Param flat ↔ chunks
# ===================================================================

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
    # Bulk (re, im) → complex via a single broadcast, then split into chunks
    # with an OUTER comprehension over contiguous slices. Three reasons this is
    # Zygote-safe at L=32 where the obvious encodings fail:
    #   - bulk `complex.(...)` avoids a per-element inner comprehension (the
    #     original blew up the pullback type with ~120K scalar pullbacks),
    #   - the outer comprehension yields a homogeneous Vector{Vector{ComplexF64}}
    #     (33 slice `getindex` pullbacks, all the same type),
    #   - no in-place `setindex!` (Zygote forbids array mutation).
    n_complex = length(flat) ÷ 2
    mat   = reshape(flat, 2, n_complex)
    all_c = complex.(mat[1, :], mat[2, :])
    offsets = cumsum([0; QMPS_NUMEL])
    return [all_c[offsets[i] + 1 : offsets[i + 1]] for i in 1:length(QMPS_NUMEL)]
end

# ===================================================================
# Python-facing API
# ===================================================================

n_qmps_params_real() = QMPS_NPARAMS_REAL
d_f()                = D_F
n_actions()          = N_ACTIONS

get_qmps_params() = _chunks_to_real_flat(QMPS_CHUNKS[])

function set_qmps_params!(flat::AbstractVector)
    QMPS_CHUNKS[] = _real_flat_to_chunks(collect(Float64, flat))
    return nothing
end

# Single forward (no grad): returns Vector{Float64} of length D_F. This is the
# reference the torch contraction (qmps_torch) is checked against; gradients no
# longer cross the Julia line (torch.autograd owns the backward — see CLAUDE.md).
function qmps_feature(state_id::Int)
    ψ_conj = [conj(t) for t in get_arrays(state_id)]
    return _qmps_feature(ψ_conj, QMPS_CHUNKS[])
end

# ===================================================================
# State bond caps — consumed by the Python/torch contraction (qmps_torch).
# ===================================================================
# Structural per-bond dimension caps for the quantum-state MPS. qmps_torch reads
# this to size its zero-padded batched site tensors. The batched, differentiable
# contraction itself now lives in qmps_torch.py via torch.autograd; the Julia
# side only provides the forward reference `qmps_feature` and these caps.
const STATE_BONDS = Tuple(min(2^p, 2^(N - p), D_QSTATE) for p in 1:N-1)

end # module QMPSRL

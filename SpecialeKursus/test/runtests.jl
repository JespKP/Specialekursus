using Test
using jp_test_package
using Tensors

# SymmetricTensor{2,2} expects components as (σ_11, σ_12=σ_21, σ_22).
# Example: σ = SymmetricTensor{2,2}((a, b, c)) means a = σ_11, b = σ_12 = σ_21, c = σ_22.

# ── Shared material ──────────────────────────────────────────────────────────
E    = 200_000.0
ν    = 0.3
σ_y0 = 250.0
h    = 10_000.0

mat = plane_stress_tensors(E, ν)

# ── effective_stress ─────────────────────────────────────────────────────────
# Entry point: jp_test_package.effective_stress(σ)
@testset "effective_stress" begin
    # Uniaxial: σ_e = |σ_11|
    σ = SymmetricTensor{2,2}((100.0, 0.0, 0.0))
    @test jp_test_package.effective_stress(σ) ≈ 100.0

    # Pure shear: σ_e = √3/2 * τ  (von Mises)
    τ = 50.0
    σ = SymmetricTensor{2,2}((0.0, τ, 0.0))
    @test jp_test_package.effective_stress(σ) ≈ sqrt(3) * τ

    # Zero stress
    @test jp_test_package.effective_stress(zero(SymmetricTensor{2,2})) ≈ 0.0

    # Equibiaxial (σ_11 = σ_22, no shear): σ_e = σ_11
    σ = SymmetricTensor{2,2}((200.0, 0.0, 200.0))
    @test jp_test_package.effective_stress(σ) ≈ 200.0
end

# ── yield_function ───────────────────────────────────────────────────────────
# Entry point: jp_test_package.yield_function(σ, σ_y)
@testset "yield_function" begin
    σ_el = SymmetricTensor{2,2}((200.0, 0.0, 0.0))   # inside  (F < 0)
    σ_pl = SymmetricTensor{2,2}((300.0, 0.0, 0.0))   # outside (F > 0)
    σ_on = SymmetricTensor{2,2}((σ_y0,  0.0, 0.0))   # on surface (F = 0)

    @test jp_test_package.yield_function(σ_el, σ_y0) < 0
    @test jp_test_package.yield_function(σ_pl, σ_y0) > 0
    @test jp_test_package.yield_function(σ_on, σ_y0) ≈ 0.0
end

# ── flow_direction ───────────────────────────────────────────────────────────
# Entry point: jp_test_package.flow_direction(σ)
# For uniaxial σ = [σ_11, 0, 0]:
#   n_11 = (2σ_11)/(2σ_11) = 1,  n_22 = (-σ_11)/(2σ_11) = -0.5,  n_12 = 0
@testset "flow_direction" begin
    σ = SymmetricTensor{2,2}((300.0, 0.0, 0.0))
    n = jp_test_package.flow_direction(σ)

    @test n[1,1] ≈  1.0
    @test n[2,2] ≈ -0.5
    @test n[1,2] ≈  0.0

    # Zero stress → zero tensor (no division by zero)
    @test jp_test_package.flow_direction(zero(SymmetricTensor{2,2})) == zero(SymmetricTensor{2,2})
end

# ── compute_Δλ ───────────────────────────────────────────────────────────────
# Entry point: jp_test_package.compute_Δλ(σ_prev, Δε, mat, h)
@testset "compute_Δλ" begin
    σ_prev = SymmetricTensor{2,2}((σ_y0, 0.0, 0.0))  # on yield surface

    # Plastic loading: strain increment that drives further tension → Δλ > 0
    Δε_pl = SymmetricTensor{2,2}((1e-4, 0.0, 0.0))
    @test jp_test_package.compute_Δλ(σ_prev, Δε_pl, mat, h) > 0.0

    # Elastic unloading: compressive strain → Δλ < 0 (clamped to 0 in stress_update!)
    Δε_el = SymmetricTensor{2,2}((-1e-4, 0.0, 0.0))
    @test jp_test_package.compute_Δλ(σ_prev, Δε_el, mat, h) < 0.0
end

# ── stress_update! ───────────────────────────────────────────────────────────
# Entry point: stress_update!(state, Δε, mat, h)
@testset "stress_update!" begin
    @testset "elastic step stays inside yield surface" begin
        state = init_plastic_state(σ_y0)
        Δε    = SymmetricTensor{2,2}((1e-5, 0.0, 0.0))   # tiny elastic step
        stress_update!(state, Δε, mat, h)

        @test state.Δλ == 0.0
        @test jp_test_package.yield_function(state.σ, state.σ_y) < 0.0
    end

@testset "plastic step: explicit plastic increment update" begin
    state = init_plastic_state(σ_y0)
    σ_prev = state.σ = SymmetricTensor{2,2}((σ_y0, 0.0, 0.0))
    σy_prev = state.σ_y

    Δε = SymmetricTensor{2,2}((2σ_y0 / E, 0.0, 0.0))

    n = jp_test_package.flow_direction(σ_prev)
    Δλ_expected = jp_test_package.compute_Δλ(σ_prev, Δε, mat, h)
    σ_expected = σ_prev + mat.C ⊡ (Δε - n * Δλ_expected)
    σy_expected = σy_prev + h * Δλ_expected

    stress_update!(state, Δε, mat, h)

    @test state.Δλ > 0.0
    @test state.Δλ ≈ Δλ_expected
    @test state.σ_y ≈ σy_expected
    @test state.σ ≈ σ_expected
end

    @testset "hardening: yield stress grows with Δλ" begin
        state = init_plastic_state(σ_y0)
        state.σ = SymmetricTensor{2,2}((σ_y0, 0.0, 0.0))

        Δε = SymmetricTensor{2,2}((5e-4, 0.0, 0.0))
        stress_update!(state, Δε, mat, h)

        @test state.σ_y ≈ σ_y0 + h * state.Δλ
    end

    @testset "elastic unloading from yield surface: no plastic flow" begin
        state = init_plastic_state(σ_y0)
        state.σ = SymmetricTensor{2,2}((σ_y0, 0.0, 0.0))

        Δε = SymmetricTensor{2,2}((-1e-4, 0.0, 0.0))   # unload
        stress_update!(state, Δε, mat, h)

        @test state.Δλ == 0.0
    end
end

# ── compute_Cep ──────────────────────────────────────────────────────────────
# Entry point: compute_Cep(σ, mat, h)
@testset "compute_Cep" begin
    σ = SymmetricTensor{2,2}((σ_y0, 0.0, 0.0))

    @testset "major symmetry: C_ep[i,j,k,l] == C_ep[k,l,i,j]" begin
        Cep = compute_Cep(σ, mat, h)
        @test Cep ≈ permutedims(Cep, (3, 4, 1, 2))
    end

    @testset "large h → Cep ≈ C (elastic limit)" begin
        Cep_stiff = compute_Cep(σ, mat, 1e12)
        @test Cep_stiff ≈ mat.C  rtol=1e-6
    end

    @testset "perfect plasticity (h=0): softer than elastic" begin
        Cep0 = compute_Cep(σ, mat, 0.0)
        n    = jp_test_package.flow_direction(σ)
        # n : Cep : n == 0  (no stiffness in flow direction for h=0)
        @test (n ⊡ Cep0 ⊡ n) ≈ 0.0  atol=1e-8
    end
end

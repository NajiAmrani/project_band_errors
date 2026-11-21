Ecut_rough = 30
Ecut_converged = ...
Ecut_reference = round(1.25 * Ecut_converged)
kgrid = (2, 2, 2)
moltiplicators = [ 2.0,2.1,2.2,2.3,2.5, 3.0, 3.5]
Eref_array_rough = Ecut_rough .* moltiplicators
Eref_array_converged = Ecut_converged .* moltiplicators

basis_reference = PlaneWaveBasis(model; kgrid=kgrid, Ecut=Ecut_reference)
scfres_reference = self_consistent_field(basis_reference)
refinement_reference = refine_scfres(round_occupation(scfres_reference), basis_reference)
est_err_reference = norm_l2(basis_reference, refinement_reference.δρ)
ρ_reference = scfres_reference.rho

basis_rough = PlaneWaveBasis(model; kgrid=kgrid, Ecut=Ecut_rough)
scfres_rough = self_consistent_field(basis_rough)

basis_converged = PlaneWaveBasis(model; kgrid=kgrid, Ecut=Ecut_converged)
scfres_converged = self_consistent_field(basis_converged)


#the actual error doesn't depend on Eref, only on E_reference
ρ_rough_in_reference_basis = DFTK.transfer_density(scfres_rough.rho, basis_rough, basis_reference)
act_err_rough = norm_l2(basis_reference, ρ_rough_in_reference_basis - ρ_reference)
ρ_converged_in_reference_basis = DFTK.transfer_density(scfres_converged.rho, basis_converged, basis_reference)
act_err_converged = norm_l2(basis_reference, ρ_converged_in_reference_basis - ρ_reference)

est_err_rough=Float64[]
est_err_converged=Float64[]


for E_ref in Eref_array_rough
    basis_ref=PlaneWaveBasis(model; kgrid=kgrid, Ecut=E_ref)
    refinement_rough = refine_scfres(round_occupation(scfres_rough), basis_ref)
    push!(est_err_rough,norm_l2(basis_rough, refinement_rough.δρ))

end

for E_ref in Eref_array_converged
    basis_ref=PlaneWaveBasis(model; kgrid=kgrid, Ecut=E_ref)
    refinement_converged = refine_scfres(round_occupation(scfres_converged), basis_ref)
    push!(est_err_converged,norm_l2(basis_converged, refinement_converged.δρ))

end


p1 = plot(
    Eref_array_rough,
    est_err_rough;
    yscale = :log10,
    marker = :circle,
    xlabel = "E_ref",
    ylabel = "Estimated error",
    label = "rough",
    title = "Estimated error vs E_ref"
)

plot!(
    p1,
    Eref_array_converged,
    est_err_converged;
    yscale = :log10,
    marker = :square,
    label = "converged"
)

display(p1)
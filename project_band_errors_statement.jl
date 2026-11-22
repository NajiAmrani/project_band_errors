### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 86889076-cc43-11f1-b8e1-b1c256252262
begin
	# https://plutojl.org/en/docs/packages-advanced/#shared-environment
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	
	using AtomsView
	using AtomsBase
	using Brillouin
	using Unitful
	using UnitfulAtomic
	using DFTK
	using PseudoPotentialData
	using Printf
	using LinearAlgebra
	using ComponentArrays
	using StaticArrays
	using Optim
	using LineSearches
	using PlutoTeachingTools
	using PlutoUI
	using Plots
	using ForwardDiff

	#custom
	include("scripts/cache_heavy_func.jl")
end

# ╔═╡ 436c0704-0cb9-444b-918f-4d26fc05268a
md"""
# Errors of band structure calculations
"""

# ╔═╡ 0d3f3fb2-68f1-4a3f-9e1b-4c61114a09cd
md"""
!!! note "Expect long runtime"
	Because of various larger-ish computations running in this notebook, expect
	a larger runtime on startup.
"""

# ╔═╡ a9740f44-86b4-4c7c-84ef-ba065e2be93e
TableOfContents()

# ╔═╡ 9a721598-bcec-4071-a774-3b2d9dd77eb7
md"""
## Introduction
"""

# ╔═╡ c83b7e77-28a3-4d1f-a315-2c59eee5134e
md"""
In this project we will consider BaTiO3, a prototypical [ferroelectric material](https://en.wikipedia.org/wiki/Ferroelectricity), i.e. a material that can exhibit a spontaneous electric polarisation.
At high temperatures, it has a cubic perovskite structure,
but as the temperature is lowered
it undergoes a series of phase transitions that lower its symmetry
to a tetragonal, then orthorhombic and then finally rhombohedral structure.
We are considering in this work the tetragonal phase of BaTiO3.

Our goal will be to understand the errors involved in computing a BaTiO3 band structure.
"""

# ╔═╡ c93bab99-81f2-4e28-adb6-33c890644ac7
md"""
!!! important "Open questions and course-wide discussion needed"
	The questions in this project are deliberately formulated very open and leave you a lot of choice. In fact some of them are actual research tasks (in the sense that we genuinely don't know the answer).

	As a result some tasks ask a lot from you and can be time consuming. Coordinate within your group, but also speak with your other classmates and exchange ideas.

	Furthermore, we will not penalise you on making "wrong choices" if you provide a rationale for your decisions and if your rationale is reasonable. Some techniques required go  beyond what we discussed in class. This is what the exercises sessions are for. Feel free to make use of the TA to give you tips and help you explore the project.
"""

# ╔═╡ 9c630c72-27fb-4d46-a1fa-44802f12553d
begin
batio3_tetragonal(a::ComponentArray) = batio3_tetragonal(; a...)

"""
    batio3_tetragonal(; a, c_over_a, z_Ti, z_O1, z_O2) -> FlexibleSystem

Build tetragonal BaTiO₃ (space group P4mm, No. 99) from its free structural
parameters. All quantities in atomic units:

  - `a`        : in-plane lattice constant (Bohr)
  - `c_over_a` : tetragonality ratio c/a (dimensionless)
  - `z_Ti`     : z of Ti
  - `z_O1`     : z of apical O
  - `z_O2`     : z of equatorial O
"""
function batio3_tetragonal(; a, c_over_a, z_Ti, z_O1, z_O2)
    T = promote_type(typeof(a), typeof(c_over_a),
                     typeof(z_Ti), typeof(z_O1), typeof(z_O2))
    c = c_over_a * a
    
    box = (SVector{3}(T[a, 0, 0] .* u"bohr"),
           SVector{3}(T[0, a, 0] .* u"bohr"),
           SVector{3}(T[0, 0, c] .* u"bohr"))
    atoms = [
        :Ba => SVector{3,T}(0.0, 0.0, 0   ),
        :Ti => SVector{3,T}(0.5, 0.5, z_Ti),
        :O  => SVector{3,T}(0.5, 0.5, z_O1),     # apical O1)
        :O  => SVector{3,T}(0.5, 0.0, z_O2),     # equatorial O2a
        :O  => SVector{3,T}(0.0, 0.5, z_O2),     # equatorial O2b
    ]

    periodic_system(atoms, box; fractional=true)
end
end

# ╔═╡ 4bd0d4f5-de56-4a8e-9c4d-36004073c411
md"""
Let's look first at the experimental structure. As the experimental values
we take the parameters from [Aimi et al., J. Ceram. Soc. Jpn. **129**, 73 (2021)](https://doi.org/10.2109/jcersj2.20147), which can be found in the [Crystallography Open Database](https://www.crystallography.net/cod/1559964.html). We collect these into a `ComponentVector`. For the sake of this notebook you can assume that the absolute error in these experimental values is `0.005` for each parameter[^1]
"""

# ╔═╡ a48ace23-d7e9-416d-bfac-ab3f636e0cd8
p_experiment = let
	a_exp = 3.99048u"Å"
	c_exp = 4.02303u"Å"

	ComponentVector(
		a        = austrip(a_exp),  # We use atomic units in this notebook
		c_over_a = c_exp / a_exp,
		z_Ti     =  0.522,
		z_O1     =  0.979,
		z_O2     =  0.488,
	)
end

# ╔═╡ 9e431fe5-7b7c-4217-bbf3-861f3f6a0a2a
md"""
[^1]: In reality it's a little less and the dependency of the errors is more involved, but we ignore this here.
"""

# ╔═╡ a78e1830-f205-4f7b-9d68-3d183af311ae
batio3_tetragonal(p_experiment)

# ╔═╡ 52f0babb-6ed4-4687-a2ed-27cb777a9482
md"""
We want to proceed to compute the band structure of BaTiO3. In typical calculations this proceeds in two steps as outlined below.
"""

# ╔═╡ 490f6e59-e04a-44a1-9625-520a3929d57e
md"""
### Step 1: Relax the geometry

Depending on the DFT functional chosen, the minimal geometry will be different to the experimental parameters. Using the experimental parameters directly for the band structure computation can sometimes lead to artifacts in the predicted properties. 
Before we can actually compute the band structure we therefore need to relax the geometry.
"""

# ╔═╡ d3ec45b6-c8bf-4d40-9131-84d9a9653601
md"""
#### Numerical parameters
We first choose a set of model and numerical parameters for this notebook:
"""

# ╔═╡ 849bc196-89e0-4cf8-856d-a0943792657c
begin
	# Model choice
	pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")

	functionals = PBE()
end;

# ╔═╡ e11b06bb-0c75-49df-9985-1c555d771946
begin 
	# Numerical parameters choice
	Ecut  = 10
	kgrid = MonkhorstPack(2, 2, 2)
end;

# ╔═╡ c089aa93-91ab-41fe-84d9-759222d12cb3
md"""
#### Geometry relaxation function

The below defines a function, that given some numerical and modelling parameters as well as BaTiO3 structure parameters runs a DFT computation and returns both the total energy and the gradient of the energy with respect to the structural parameters.

Notice, that we use some temperature even though BaTiO3 is an insulator. This often helps during geometry optimisation to avoid numerical issues in the SCF, which is why it is done.

Furthermore, by default for lattice optimization we use a [modified kinetic energy](https://docs.dftk.org/stable/examples/energy_cutoff_smearing/) (`kinetic_blowup=BlowupCHV()`), which smoothly increases the kinetic energy of plane waves close to `Ecut` and regularizes the energy across discontinuities coming from basis size changes during lattice optimization. This helps the optimizer especially at low Ecut, but also changes the bias of the result at low Ecut.
"""

# ╔═╡ e9a5d384-7e2d-45a2-ac08-a34eaae18078
function batio3_energy_and_gradient_from_scfres(scfres, p)
    E = scfres.energies.total
    F_Ba, F_Ti, F_O1, F_O2a, F_O2b = compute_forces_cart(scfres)
    σ = compute_stresses_cart(scfres)  # σ = (1/V) ∂E/∂ε

    # From the computed force and stress, get to the required energy derivatives
    # wrt. the parameters of the structure. This is done using the chain rule.    
    a = p.a
    η = p.c_over_a
    c = η * a
    Ω = scfres.basis.model.unit_cell_volume

    # Atomic coordinates: r_z = c * z_frac at fixed lattice
    #   ∂E/∂z_frac = (∂E/∂r_z) * c = -F_z * c
    dE_zTi = -F_Ti[3] * c
    dE_zO1 = -F_O1[3] * c
    dE_zO2 = -(F_O2a[3] + F_O2b[3]) * c   # both equatorial sites share z_O2

    # Lattice parameters at fixed fractional coordinates: dE = V tr(σ ε)
    #   a → a + da at fixed c/a  ⇒  ε = (da/a) I       ⇒ ∂E/∂a = V tr(σ) / a
    #   η → η + dη at fixed a    ⇒  ε_zz = dη/η        ⇒ ∂E/∂η = V σ_zz / η
    dE_a = Ω * tr(σ) / a
    dE_η = Ω * σ[3, 3] / η

    dE = ComponentVector(a=dE_a, c_over_a=dE_η,
                         z_Ti=dE_zTi, z_O1=dE_zO1, z_O2=dE_zO2)
    (; E, dE)
end

# ╔═╡ d531968e-04a2-4f9b-a1c8-1be62018bf30
function batio3_energy_and_gradient(p::ComponentVector;
                    Ecut, kgrid, functionals, pseudopotentials, tol=1e-8, verbose=true,
                    kinetic_blowup=BlowupCHV())
    system = batio3_tetragonal(p)

    model  = model_DFT(system; functionals, pseudopotentials, smearing=Smearing.Gaussian(), temperature=1e-3, kinetic_blowup)
    basis  = PlaneWaveBasis(model; Ecut, kgrid)
    callback = verbose ? ScfDefaultCallback() : identity
    response = ResponseOptions(; verbose)
    scfres = self_consistent_field(basis; tol, callback, response)
    scfres.converged || error("SCF not converged (Ecut=$Ecut, kgrid=$kgrid)")

    batio3_energy_and_gradient_from_scfres(scfres, p)
end

# ╔═╡ a578c81c-8e47-4b25-8017-ef694cd1b13d
md"""
Finally we need an algorithm to relax the geometry of the system. We provide two algorithms based on `Optim`. `relax_batio3_a` relaxing only the `a` parameter, keeping all others fixed and `relax_batio3_full` doing the full relaxation in all five parameters. The latter is numerically more involved.
"""

# ╔═╡ 4376cf04-2c93-4f3f-a89f-2defe163078a
function relax_batio3_a(p0::ComponentVector;
					Ecut, kgrid, functionals, pseudopotentials,
					kinetic_blowup=BlowupCHV(),
					scf_tol=1e-8, scf_verbose=false,
					optim_tol=1e-4, optim_verbose=true,
                    iterations=50, maxstepsize=0.05)
    to_full_params(x) = (p = copy(p0); p.a = x.a; p)
    
    function fg!(F, G, x)
        E, dE = batio3_energy_and_gradient(to_full_params(x);
                    Ecut, kgrid, functionals, pseudopotentials, kinetic_blowup,
                    tol=scf_tol, verbose=scf_verbose)
        if !isnothing(G)
            G.a = dE.a
        end
        E
    end

    method = LBFGS(; alphaguess=InitialStatic(; alpha=maxstepsize, scaled=true),
                     linesearch=BackTracking(; order=3))
    options = Optim.Options(; g_tol=optim_tol, iterations, show_trace=optim_verbose)
    optim_result = optimize(Optim.only_fg!(fg!), ComponentVector(; p0.a), method, options)
    (; p=to_full_params(Optim.minimizer(optim_result)), optim_result)
end

# ╔═╡ 15e29343-30f6-4d68-bfa3-d14283985a25
function relax_batio3_full(p0::ComponentVector;
					Ecut, kgrid, functionals, pseudopotentials,
					kinetic_blowup=BlowupCHV(),
					scf_tol=1e-8, scf_verbose=false,
					optim_tol=1e-4, optim_verbose=true,
                    iterations=50, maxstepsize=0.05)
	# Apart from p0.a all parameters are unitless. To improve conditioning
    # of the minimisation problem we therefore also express the lattice constant
    # a relative to the initial lattice constant.
	a0 = p0.a

    # This converts between the two parametrisations
    to_relative(p)   = (x = copy(p); x.a = p.a / a0; x)
    from_relative(x) = (p = copy(x); p.a = x.a * a0; p)

    function fg!(F, G, x)    
        E, dE = batio3_energy_and_gradient(from_relative(x);
                    Ecut, kgrid, functionals, pseudopotentials, kinetic_blowup,
                    tol=scf_tol, verbose=scf_verbose)
        if !isnothing(G)
            copyto!(G, dE)
            G.a = dE.a * a0  # ∂E/∂(a/a0) = a0 ∂E/∂a
        end
        E
    end

    method = LBFGS(; alphaguess=InitialStatic(; alpha=maxstepsize, scaled=true),
                     linesearch=BackTracking(; order=3))
    options = Optim.Options(; g_tol=optim_tol, iterations, show_trace=optim_verbose)
    optim_result = optimize(Optim.only_fg!(fg!), to_relative(p0), method, options)
    (; p=from_relative(Optim.minimizer(optim_result)), optim_result)
    
end

# ╔═╡ 283a97c9-8975-4807-9b5c-c303f4a830b8
md"""
#### Minimal-energy geometry

In an actual application we would now run `relax_batio3_full` to relax the full geometry. Here, we will stick to only running `relax_batio3_a` to save computational cost.
"""

# ╔═╡ e06f1089-56fe-409d-bd87-beb9895add85
mingeo = run_or_load(
    relax_batio3_a, 
    p_experiment; 
    Ecut = Ecut, 
    kgrid = kgrid, 
    pseudopotentials = pseudopotentials, 
    functionals = functionals,
    name="example_mingeo"
)

# ╔═╡ f724b839-2d27-45aa-a251-27462124de7f
relaxed = batio3_tetragonal(; mingeo.p...)

# ╔═╡ 6a8103d6-e38b-4fe1-91bf-9b312c848ccd
md"""
### Step 2: Compute the band structure

Based on the relaxed geometry, we now compute the band structure. As in the geometry optimisation we use a small temperature. This makes the SCF robust.
"""

# ╔═╡ 2376a93f-1472-41ca-a867-6e3c52f653a0
model  = model_DFT(relaxed; functionals, pseudopotentials, smearing=Smearing.Gaussian(), temperature=1e-3);

# ╔═╡ 09740792-4a26-4743-9215-c3ec0fea0dd8
basis  = PlaneWaveBasis(model; Ecut, kgrid)

# ╔═╡ 9bffe37b-6322-475e-87ba-f9a7a58adeb3
scfres = self_consistent_field(basis; tol=1e-8);

# ╔═╡ d9f5b2cd-056d-44ce-9d06-e1a3a408258c
begin
	# Decrease kline_density to 15 to speed up computations
	bands = compute_bands(scfres, kline_density=25)
	plot_bandstructure(bands)
end

# ╔═╡ 2bc36bfb-1b8e-402a-b6a4-93adf89cdc15
md"""
## Part 1: Parameter choices and their impact

The above band structure looks very rough and much less smooth than most of the band structures we have seen before. The reason for this is that the employed numerical parameters `Ecut` and `kgrid` are very rough and far from numerically converged.
"""

# ╔═╡ 8020dfc7-c8dd-43d1-8f4c-09a3515b5036
md"""
!!! warning "Task 1.1: Numerical parameters for the SCF"
	In this first task you investigate the impact of the chosen cutoff `Ecut` and the chosen k-point grid `kgrid` on the result. As you vary these parameters investigate the convergence of:
	- the **total energy per atom**,
	- the band **eigenvalues at the $Γ$ point**, which can be read out already from `scfres.eigenvalues[1]` as the $Γ$ point is at the first index of the k-grid and `compute_bands` can be skipped here. Since band energies may include an arbitrary shift, consider the convergence of the differences `scfres.eigenvalues[1] .- scfres.eigenvalues[1][n_occ]`, where `n_occ = model.n_electrons ÷ 2` is the index of the highest occupied eigenvalue.
	Investigate the behaviour of these quantities for multiple DFT functional choices (**LDA**, **PBE** and **r2SCAN**, see details below). Comment on the behaviour.

	Typical accuracy requirements are a total energy converged to about **$10^{-5}$ Hartree per atom** and band eigenvalues converged to about **$10^{-4}$ Hartree**. For each functional suggest values that achieve this. We use these as the **converged numerical parameters** throughout this project.

	Furthermore consider `Ecut=30` and `kgrid=(4, 4, 4)` to be the **rough numerical parameters**. What is the error at this level for each of the functionals ?
"""

# ╔═╡ 7653c7b4-3dd4-4c04-82e1-b31a684e78d0
md"""
!!! warning "Task 1.2: Numerical parameters for the geometry optimisation"
	Consider the geometry optimisation step 1.

	By optimising the geometry using different parameters (different functionals, cutoffs, $k$-point grids), starting each time from the experimental structure, investigate the numerical parameters required to converge the minimal geometry for each functional to experimental accuracy, i.e. absolute error in the structural parameters less than `0.005` in the units we use (atomic units or dimensionless).

	What is the error against experiment? What error dominates? How do your obtained cutoffs and kgrid parameters compare to the converged and rough numerical parameters found in Task 1.1 ?
"""

# ╔═╡ 0628298f-2755-4be3-9d80-10ffd862f4c6
md"""
#### Details on the DFT functional choices

In this part we ask you to investigate three DFT functional choices, characterised by the following three settings of pseudopotentials and functional. To use them set the respective keyword arguments in a call `model_DFT(system; functionals, pseudopotentials, smearing=Smearing.Gaussian(), temperature=1e-3)` where `system` is the result of calling `batio3_tetragonal`
"""

# ╔═╡ bd40d8f5-4376-4c55-a6a0-5ba87a42518a
let
	# LDA model
	
	pseudopotentials = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
	functionals = LDA()
end;

# ╔═╡ 551581a6-1e93-4e7b-9dd9-097aa753d702
let
	# PBE model
	
	pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
	functionals = PBE()
end;

# ╔═╡ e431bf8e-7cde-4a09-8e4a-089d635bfea4
let
	# r2SCAN model
	
	# This uses pseudopotentials that are not yet part of any PseudoFamily.
	# When running this notebook for the first time on your computer they are
	# automatically downloaded to the `pseudos` subfolder.
	pseudopotentials = Dict(
	    :Ba => "pseudos/Ba_v1.upf",
	    :Ti => "pseudos/Ti_v1.upf",
	    :O => "pseudos/O_v2.upf",
	)
	functionals = r2SCAN()
end;

# ╔═╡ 6d7c5cb5-8951-4148-854c-23a68a565fe9
begin
	# Download the pseudopotentials we require for r2SCAN
	
	RobustLocalResource("https://raw.githubusercontent.com/epfl-matmat/supporting-mgga-addfpt/245826609147e4e605e9dc9166ae892edba71c9d/scripts/results2_response_properties/BaTiO3_r2scan01/pseudos/Ba_v1.upf", "pseudos/Ba_v1.upf")
	RobustLocalResource("https://raw.githubusercontent.com/epfl-matmat/supporting-mgga-addfpt/245826609147e4e605e9dc9166ae892edba71c9d/scripts/results2_response_properties/BaTiO3_r2scan01/pseudos/Ti_v1.upf", "pseudos/Ti_v1.upf")
	RobustLocalResource("https://raw.githubusercontent.com/epfl-matmat/supporting-mgga-addfpt/245826609147e4e605e9dc9166ae892edba71c9d/scripts/results2_response_properties/BaTiO3_r2scan01/pseudos/O_v2.upf", "pseudos/O_v2.upf")
end

# ╔═╡ 82f11096-2fff-4ecf-b8fd-b514474c793c
md"""
#### Further tips
- Use `relax_batio3_a` for testing as it converges much faster. Only when you have a rough idea of the range of numerical parameters consider `relax_batio3_full`.
- For most of this notebook we will not consider the dependency of results on the SCF tolerance (i.e. the `scf_tol` parameter of `relax_batio3_full` or the `tol` parameter of `self_consistent_field`). However, when considering convergence with respect to `Ecut` and `kgrid` one needs to make sure that these tolerances are sufficiently small. Otherwise for fine numerical convergence parameters the error is dominated by the finite SCF tolerance.
- **Long-running calculations** (geometry optimisations, convergence studies) quickly become impractical in a single reactive notebook. A better option is to use standalone Julia scripts and save results to JLD2 files for running geometry optimisations. Afterwards, you can load the results in this notebook for plotting. Optionally, there are also strategies to dynamically [cache results of DFTK as shown in this tutorial](https://teaching.matmat.org/math-materials/95_dftk_cache.html). For other results, such as the output of `relax_batio3_full`, use `jldsave` and `load` from JLD2 directly. *Important:* Make sure that every parameter that changes the result (functional, Ecut, kgrid, tolerances) enters the file name, otherwise you may silently load stale results.
- **Scripts and project environment**. If you move parts of your runs into scripts, make sure to run them from the current directory with specifying the current project with
```bash
julia --project=. myscript.jl
```
 This ensures that the consistent package versions and environment specified in `Project.toml` and `Manifest.toml` are used across scripts and this notebook.
"""

# ╔═╡ 2469deb7-c537-47cd-9430-ff7643c9e279
md"""
## Part 2: Discretisation error in band computation
"""

# ╔═╡ 77289233-8009-4e39-b91c-f3ac94a65086
md"""
One usually finds for insulators like BaTiO3 that the error converges fast with respect to increasing $k$-point sampling, while the plane-wave cutoff has a larger influence on the error. In parts 2-4 we will thus focus on techniques to estimate the discretisation error in various aspects of the band computation of BaTiO3.
"""

# ╔═╡ fc4995d7-b5b7-4c7b-a67a-99a5e4889f04
md"""
### Discretisation error due to the band computation

To get us warmed up we compute the error in the band structure computations using the **Bauer-Fike** and **Kato-Temple approaches** discussed in class. Since these approaches (as we presented them) work for linear eigenvalue problems, we ignore in this part the non-linearity of the problem and any error in the structural parameters $\mathbf{p}$. In other words we assume we know the structural parameters $\mathbf{p}$ of BaTiO3 exactly and similarly that we have the self-consistent density $\rho$ to infinite accuracy, such that the only error contribution is due to diagonalising the operator $\mathcal{H}_{k}(\rho)$ for a finite `Ecut`.

To estimate the residual we follow the idea discussed in the lecture on  perturbative error estimates, namely we employ a
secondary, larger grid (using cutoff `Eref`) to estimate the residual of a DFT
computation performed in a smaller basis (with cutoff `Ecut`). If `Eref` is
large enough, then the residual computed using the `Eref`-basis is close to the true
residual, which we can use for *a posteriori* error estimation.

To simplify the setting in this part, we again only consider the case of computing the eigenvalues at the $Γ$ point. That is having converged the SCF using `self_consistent_field` (using possibly a larger `kgrid`), the eigenvalues of the $Γ$ point can be obtained as :
"""

# ╔═╡ f86b7848-100c-4560-a1af-f75809d72272
scfres.eigenvalues[1]

# ╔═╡ ec8b593f-5c6e-4a8a-bcf8-4e6475582e8d
md"""
To compute the eigenvalue residual we want to employ an `Eref`-basis.

For this purpose `DFTK` features the functions `transfer_blochwave` and `transfer_density` to move data to a larger basis. For example, the code block below transfers wave functions and density to a basis with cutoff `Eref=30` and computes the Hamiltonian-vector product $H_k \psi_{nk}$ for all bands $n$ and for $k = (0, 0, 0)$. Since we have $23$ bands in this problem, the result is a matrix of $23$ columns.
"""

# ╔═╡ 7a025b59-c8fc-4028-aa47-a9c297c4a7ac
let
	# Construct Eref basis
	Eref = 30
	basis_ref = PlaneWaveBasis(model; basis.kgrid, Ecut=Eref)

	# Transfer to Eref basis
	ψ_ref = DFTK.transfer_blochwave(scfres.ψ, basis, basis_ref)
	ρ_ref = DFTK.transfer_density(scfres.ρ, basis, basis_ref)

	# Build new Hamiltonian and compute matrix-vector product
	hamr = Hamiltonian(basis_ref; ρ=ρ_ref)
	hamψ_Γ = hamr[1] * ψ_ref[1]

	# Also note that there is scfres.eigenvalues[1]
end

# ╔═╡ 067523fc-c17d-4f80-9b6b-e31eaac7ef60
md"""
!!! warning "Task 2.1: Bauer-Fike and Kato-Temple"
    As discussed above we assume `ρ` and $\mathbf{p}$ are known exactly and
    focus on estimating the discretisation error within the eigenvalue problem $\mathcal{H}_{k}(\mathbf{p}, \rho) \psi_{nk} = ε_{nk} \psi_{nk}$ itself. Consider the setting `kgrid=(2, 2, 2)` (to save computational cost) coupled with the `Ecut` from (a) the  rough and (b) the converged numerical parameters from Part 1. Use the PBE functional for this task.

    For both `Ecut` values first experiment with your choice of `Eref` to ensure the residual is computed reasonably exactly using the `transfer_blochwave` and `transfer_density` techniques mentioned above. Typically `Eref` is at least twice as large as `Ecut`. Report some convergence plot and rationalise your choice of `Eref`.

    Next, use your residual in combination with a suitable gap estimate to estimate the discretisation error in the eigenvalue computation using [Kato-Temple or Bauer-Fike techniques](https://teaching.matmat.org/math-materials/06_Error_bounds_eigenpairs.html) as discussed in the lecture.

    Compare the errors you estimate against the actual errors you obtained in the numerical investigation in Part 1. What do you observe ? Are the discretisation errors we estimate here reasonable for the (a) rough and (b) converged numerical parameters ?
"""

# ╔═╡ a4371699-019a-484e-a9b9-e5a0bf761dcf
md"""
### Discretisation error due to the SCF step

Next we ask the question on the sensitivity of the band structure to changes in the self-consistent density $\rho$, sticking to the assumption that $\mathbf{p}$ is known exactly.

If run at cutoff `Ecut` the self-consistent density `scfres.ρ` carries an error due to the finite sized `Ecut`-basis we employ for computing it.

Let $\delta\rho$ denote the error of the self-consistent density $\rho$ obtained in the `Ecut`-basis. Then $\rho_\text{ref} = \rho + \delta\rho$ is the *exact* density, which is out of reach for us. By using the perturbative techniques discussed in the lecture $\delta\rho$ can be estimated, namely by considering the first-order change of the density when solving the DFT problem in an `Eref`-basis rather than the `Ecut`-basis. Similar to $\delta\rho$ we also obtain an estimate $δψ_{nk}$ for the discretisation error in the orbitals.

DFTK features a function `refine_scfres`, which estimates the error in an `Ecut`-basis by this perturbation-based strategy (following  [DOI 10.1137/21m1456224](http://doi.org/10.1137/21m1456224)).

The theory behind `refine_scfres` assumes an insulator, i.e. occupations which are exactly 0 or 2. Since we use a small temperature in our SCFs, we first explicitly round the occupations using the function `round_occupation` below, which also checks that this rounding is justified.
"""

# ╔═╡ e2d80859-b26e-42e0-bbfc-5eaef7fb0467
"""
Round the occupations of an `scfres` to exactly 0 or 2 (the filled occupation).
Throws an error if the occupations deviate from these values by more than `tol`,
which indicates that the temperature is too large compared to the band gap.
"""
function round_occupation(scfres; tol=1e-10)
    f = DFTK.filled_occupation(scfres.basis.model)
    occupation = [f .* (occk .> f/2) for occk in scfres.occupation]
    deviation  = maximum(maximum(abs, o - r) for (o, r) in zip(scfres.occupation, occupation))
    deviation < tol || error("Occupations deviate by $deviation from 0 or $f. " *
                             "The temperature is too large for this band gap.")
    merge(scfres, (; occupation))
end;

# ╔═╡ 4df01373-0858-44b6-9ea6-4ff1d5b344b0
begin
	Eref = 30
	basis_ref  = PlaneWaveBasis(model; kgrid, Ecut=Eref)
	scfres_ref = refine_scfres(round_occupation(scfres), basis_ref)
	
	scfres_ref.δρ  # Estimated density error
end;

# ╔═╡ 36f9e1da-9677-476c-8516-b321762e9719
md"""
Here is a function to compute the $l_2$-norm of the density error. Notice the additional factor `basis.dvol`, which comes from the fact that in DFTK the density is expressed on a real-space grid as grid points $\rho(\mathbf{r}_p)$.
We can therefore obtain the integral $\int |\rho|^2$ approximately as a trapezoidal rule:
```math
\int |\rho(\mathbf{r})|^2 d\mathbf{r} ≈ \sum_{m\in\text{grid}} w_m \, |\rho(\mathbf{r}_m)|^2 
```
where $w_m$ is an appropriate integration weight, which in this case (regular Cartesian grid) is just a single constant `basis.dvol`.
"""

# ╔═╡ 1b6bc6c4-d62d-4f81-bbb5-948d6be1bd39
function norm_l2(basis, ρ)
	@assert size(ρ)[1:3] == basis.fft_size
	integral = sum(ρ) do ρ_m
		abs(ρ_m)^2 * basis.dvol
	end
	sqrt(integral)
end

# ╔═╡ c2942550-427f-4c71-af72-f468d7d4bb26
norm_l2(basis_ref, scfres_ref.δρ)

# ╔═╡ eba3e75e-ba36-4d26-9399-07f681e6e4d1
md"""
!!! warning "Task 2.2: Investigate convergence of the error wrt. Eref"
	Investigate the convergence of the estimated density error
    with respect to `Eref`. Stick to the PBE functional and consider (a) `Ecut=30` (rough parameters from Task 1.1) and (b) your converged `Ecut`. Stick to a small `kgrid`, e.g. `kgrid=(2,2,2)` (as discussed the actual `kgrid` should have little influence here). Plot the convergence of the estimated density $l_2$-norm error with respect to `Eref` and compare it against the *actual error*.

    For computing this $l_2$ error a challenge is that the density grids between different cutoffs do not agree. Assume `basis_small` is the basis using a small `Ecut` and `ρ_small` the corresponding self-consistent density, while using `basis_large` (with larger `Ecut`) leads to `ρ_large`. Then simply taking `norm_l2(basis_large, ρ_large - ρ_small)` will not work as the array sizes mismatch. Instead you need to use `DFTK.transfer_density`, i.e. do
    ```julia
    ρ_small_in_large_basis = DFTK.transfer_density(ρ_small, basis_small, basis_large)
    norm_l2(basis_large, ρ_small_in_large_basis - ρ_large)
    ```

    To obtain the *actual error*, compute a *reference density* at about 1.25 times the cutoff you used as your converged parameter in (b), using the same `kgrid`. Note that this is a more expensive calculation, so it is advisable to store this result as a JLD2.

    How do your `Eref` values from this task compare to Task 2.1? Rationalise this outcome.
"""

# ╔═╡ 055d2b4e-6bea-41ee-a9b7-1c5e1eea6587
params_2_2 = let

	kgrid = (2, 2, 2)
	
	Ecut_rough = 30
	Ecut_converged = 30 		# to be changed
	
	#used as a reference to get the actual error
	Ecut_reference = round(1.25 * Ecut_converged)	
		
	# the E_ref arrays are toylored to the E_cut used
	moltiplicators = [2.0, 2.1, 2.2, 2.3, 2.5, 3.0, 3.5]
	Eref_array_rough = Ecut_rough .* moltiplicators
	Eref_array_converged = Ecut_converged .* moltiplicators

	# we assume p to be perfect (should we use the same for all Ecuts?)
	# this needs to be replaced by the p found after geom minimization
	# for the converged paramters.
	p = relaxed 	

	# PBE model
	pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
	functionals = PBE()
	
	model  = model_DFT(p; functionals, pseudopotentials, smearing=Smearing.Gaussian(), temperature=1e-3)

	(; kgrid, Ecut_rough, Ecut_converged, Ecut_reference, Eref_array_rough, Eref_array_converged, model)
	
end;

# ╔═╡ 07abf477-595f-4939-8217-3a83cb24c24d
function compute_reference_solution(basis)
	scfres_reference =  self_consistent_field(basis; tol=1e-8)
	return scfres_reference.ρ
end

# ╔═╡ af9660d4-d88c-4d8c-b7f2-ba4c207d9367
results2_2 = let

	# compute reference density for actual error calculation
	basis_reference = PlaneWaveBasis(params_2_2.model; kgrid=params_2_2.kgrid, Ecut=params_2_2.Ecut_reference)
	ρ_reference = run_or_load(compute_reference_solution, basis_reference;
								name="ref_for_actual_density_error")

	#*** not sure this is needed
	#refinement_reference = refine_scfres(round_occupation(scfres_reference), basis_reference)
	#est_err_reference = norm_l2(basis_reference, refinement_reference.δρ)
	#***
	
	# rough solution
	basis_rough = PlaneWaveBasis(params_2_2.model; kgrid=params_2_2.kgrid, Ecut=params_2_2.Ecut_rough)
	scfres_rough = self_consistent_field(basis_rough; tol=1e-8)

	# converged solution
	basis_converged = PlaneWaveBasis(params_2_2.model; kgrid=params_2_2.kgrid, Ecut=params_2_2.Ecut_converged)
	scfres_converged = self_consistent_field(basis_converged; tol=1e-8)


	#the actual error doesn't depend on Eref, only on E_reference
	# actual rough error
	ρ_rough_in_reference_basis = DFTK.transfer_density(scfres_rough.ρ, basis_rough, basis_reference)
	act_err_rough = norm_l2(basis_reference, ρ_rough_in_reference_basis - ρ_reference)
	# actual converged error
	ρ_converged_in_reference_basis = DFTK.transfer_density(scfres_converged.ρ, basis_converged, basis_reference)
	act_err_converged = norm_l2(basis_reference, ρ_converged_in_reference_basis - ρ_reference)
	
	est_err_rough=Float64[]
	est_err_converged=Float64[]

	print("everything before done\n")

	for E_ref in params_2_2.Eref_array_rough
	    basis_ref=PlaneWaveBasis(params_2_2.model; kgrid=params_2_2.kgrid, Ecut=E_ref)
	    refinement_rough = refine_scfres(round_occupation(scfres_rough), basis_ref)
	    push!(est_err_rough, norm_l2(basis_ref, refinement_rough.δρ))
	end

	for E_ref in params_2_2.Eref_array_converged
	    basis_ref=PlaneWaveBasis(params_2_2.model; kgrid=params_2_2.kgrid, Ecut=E_ref)
	    refinement_converged = refine_scfres(round_occupation(scfres_converged), basis_ref)
	    push!(est_err_converged, norm_l2(basis_ref, refinement_converged.δρ))
	end

	(;est_err_rough, est_err_converged, act_err_rough, act_err_converged)
end

# ╔═╡ 98069eca-c0cc-4a3c-b000-e7e6854790f2
let
	p1 = plot( 
	    params_2_2.Eref_array_rough,
	    results2_2.est_err_rough;
	    yscale = :log10,
	    marker = :circle,
	    xlabel = "E_ref",
	    ylabel = "Estimated error",
	    label = "rough",
	    title = "Estimated error vs E_ref"
	)
	
	plot!(
	    p1,
	    params_2_2.Eref_array_converged,
	    results2_2.est_err_converged;
	    yscale = :log10,
	    marker = :square,
	    label = "converged"
	)

	hline!(p1, [results2_2.act_err_converged], yscale = :log10, label = "actual error fro converged", linestyle = :dash, linewidth = 2)
	hline!(p1, [results2_2.act_err_rough], label = "actual error fro rough", linestyle = :dash, linewidth = 2)
end

# ╔═╡ 4ed1f9ed-d404-4a1c-8613-652d1cdf9af5
md"""
With such a $\delta \rho$ at hand we are able to compute a new Hamiltonian $\mathcal{H}_k(\rho_\text{ref}) = \mathcal{H}_k(\rho + \delta\rho)$,
inducing a perturbation $\delta \mathcal{H}_{k} = \mathcal{H}_k(\rho_\text{ref}) - \mathcal{H}_k(\rho)$ to our reference eigenproblem $\mathcal{H}_{k}(\rho) \psi_{nk} = ε_{nk} \psi_{nk}$.

By first order perturbation theory, the first-order change of the eigenvalue $δε_{nk}$ due to this change in Hamiltonian is given by
```math
δε_{nk} = \langle \psi_{nk}, \delta\mathcal{H}_{k} \psi_{nk}\rangle.
```
Therefore
```math
ε_{nk} + δε_{nk} = \langle\psi_{nk}, \mathcal{H}_k(\rho_\text{ref}) \psi_{nk}\rangle,
```
i.e. evaluating the Rayleigh quotient of the new Hamiltonian at the unperturbed eigenvectors.
"""

# ╔═╡ 8ec195ed-3d64-4260-bfd3-94b6c985db7d
md"""
As part of the computation of $\delta\rho$ the `refine_scfres` already executes `transfer_density` and `transfer_blochwave`, such that the original density (`scfres.ρ`) and the original orbitals (`scfres.ψ`) are now available in the `Eref`-basis as part of `scfres_ref`:
"""

# ╔═╡ 2d2375bd-7575-48b9-86ba-6960c0954915
propertynames(scfres_ref)

# ╔═╡ 6ff5fa3c-136c-4f6b-a56c-29dea1b09262
md"""
Note further that `scfres_ref.δρ` is the $δρ$ we are after and that `scfres_ref.basis` is equal to `basis_ref`.
"""

# ╔═╡ 0cf5d9e3-e13e-4e4b-8660-14d4c7670fdc
md"""
!!! warning "Task 2.3: Discretisation error estimation for bands"
	Based on `refine_scfres` and the above procedure estimate the error contribution in the band structures arising from the discretisation error in the self-consistent density.
	Only consider the error in the eigenvalues at the $Γ$ point such that again no call to `compute_bands` is needed here. Use `kgrid=(2,2,2)` with the `Ecut` from (a) the rough and (b) the converged numerical parameters from Part 1 in combination with the `Eref` values from Task 2.2. Use the PBE functional for this task.

	How does the estimate compare in size to the estimates from Task 2.1? How does it compare to the actual error, which you can estimate from your numerical experiments in Task 1.1.

	You should overall find that band structure errors are small when you use the converged cutoff. What is the situation for the reduced cutoff ? How does this estimate perform ?
"""

# ╔═╡ e221fb8e-dcf0-405a-9723-ce1ff95a8306


# ╔═╡ aa80aa2c-f97f-4a9c-b294-c4698e8fb1eb
md"""
## Part 3: Discretisation error estimation for structure optimisation

As you saw in Part 1, performing a structure optimisation (Step 1) can be rather costly,
especially if performed at converged numerical parameters (`Ecut` and `kgrid`)
and with an accurate DFT functional.
Furthermore we saw in Part 2 the error of the actual band structure computation itself (Step 2) can be kept small if we use converged numerical parameters for both the `self_consistent_field` and the `compute_bands` call of this step.

A typical technique is therefore to use *different* computational parameters
for the geometry optimisation (Step 1) and for the actual band structure
computation (Step 2).

For example, one may perform the geometry optimisation with rough cutoff `Ecut = 30`,
resulting in a set of optimal structural parameters $p^{E_\text{cut}=30}$.
The band structure is then done at a converged cutoff (`Ecut=80`, say) but still using
the structure $p^{E_\text{cut}=30}$. This is technically inconsistent,
but typically fine.

In this part we will develop techniques to estimate the error in the structural
parameters from running the optimisation at reduced `Ecut`.
"""

# ╔═╡ b0b544c6-9538-4608-a322-f11fe5175c7d
md"""
The crucial component here is again to consider the discretisation procedure in the SCF, estimated using `refine_scfres` (see [procedure outlined above](#Discretisation-error-due-to-the-SCF-step)). This time this concerns the SCF hidden inside `batio3_energy_and_gradient`, i.e. the SCF whose forces and stresses are used to obtain the derivatives `dE`.

As we already saw above, `scfres_ref` (the result of calling `refine_scfres`) contains both `δρ` and `δψ`, i.e. the errors in the density and the errors in the orbitals $\psi_{nk}$:
"""

# ╔═╡ 4c96c0c3-7e5f-45fa-87d4-d2d09a8473ed
propertynames(scfres_ref)

# ╔═╡ c4138d41-06fa-4864-9f72-e7dc50ec258d
md"""
Based on these errors in $\rho$ and $\psi_{nk}$ one can also obtain
estimates for the error in the force and the error in the stress,
by simply re-evaluating forces and stresses at the corrected density $ρ+δρ$
and corrected orbitals $ψ_{nk} + δψ_{nk}$. This can be achieved by re-using the function `batio3_energy_and_gradient_from_scfres` defined above.
The following function uses this idea to compute improved gradients $\mathbf{G} + \mathbf{\delta G}$, i.e. which are the sum of the gradient $\mathbf{G}$ at `Ecut` *plus* an error estimate $\mathbf{\delta G}$ of its discretisation error.
"""

# ╔═╡ f73a742f-51f2-4cef-a8f1-b4f1ead0d03b
function batio3_energy_and_gradient_with_correction(scfres_ref::DFTK.RefinementResult, p)

    # Compute energy and derivatives including the correction    
    ψ_ref = [DFTK.ortho_qr(ψk + δψk)
             for (ψk, δψk) in zip(scfres_ref.ψ, scfres_ref.δψ)]
    ρ_ref = compute_density(scfres_ref.basis, ψ_ref, scfres_ref.occupation)
    energies = DFTK.energy(scfres_ref.basis, ψ_ref, scfres_ref.occupation; ρ=ρ_ref).energies
    scfres_custom = (; scfres_ref.basis, ψ=ψ_ref, scfres_ref.occupation,
                     ρ=ρ_ref, τ=nothing, eigenvalues=nothing, εF=nothing, energies)

    (; E, dE) = batio3_energy_and_gradient_from_scfres(scfres_custom, p)
end

# ╔═╡ 74dee5ee-0def-4e64-bde2-7b899be4307e
batio3_energy_and_gradient_with_correction(scfres_ref, mingeo.p)

# ╔═╡ c7caf91b-6d1d-4f56-9a79-62be2bc507b3
md"""
!!! warning "Task 3.1: Estimate δG"
	Use your `Eref` from Task 2.2 and `batio3_energy_and_gradient_with_correction` to estimate the error $\mathbf{\delta G}$ in the energy gradient at the relaxed structure when using `Ecut=30` for PBE.
	This requires you to run `relax_batio3_full` first using a particular cutoff `Ecut`, then an SCF at the relaxed parameters `mingeo.p` (with the same model settings as inside `batio3_energy_and_gradient`), then `refine_scfres` using `Eref` (after `round_occupation`), and finally `batio3_energy_and_gradient_with_correction` at `mingeo.p`.

	*Important:* For all of Part 3 turn off the modified kinetic energy, i.e. pass `kinetic_blowup=BlowupIdentity()` to `relax_batio3_full` and use the same setting in the SCF you refine. The blowup depends on `Ecut` itself, such that the `Ecut`- and the `Eref`-discretisations no longer describe the same problem and `refine_scfres` gives meaningless results. Without the blowup, however, the geometry optimisation may stall if `Ecut` is too low (see the [note on the geometry relaxation function](#Geometry-relaxation-function)). If you encounter such problems, please let us know.

	As forces and stresses are more sensitive to the employed `kgrid` this time make sure that you use `kgrid=(4, 4, 4)` (the rough numerical parameter of Part 1).

	Also investigate the error for other cutoffs to understand how it varies. As our procedure cannot estimate the error due to the chosen `kgrid` keep this one fixed throughout. Keep in mind that `batio3_energy_and_gradient_with_correction` returns the sum $\mathbf{\mathbf G} + \mathbf{\delta G}$ as its `dE` parameter. So to get $\mathbf{\delta G}$ you need to again *subtract* $\mathbf{G}$, i.e. the `dE` returned by `batio3_energy_and_gradient_from_scfres(scfres, mingeo.p)` for the SCF result `scfres` you refined.
"""

# ╔═╡ 8e1288c0-8fea-4200-a2b3-c9012586e982
md"""
Knowing $\mathbf{\delta G}$, we can ask how to propagate this error forward towards the structural parameters
`a`, `c_over_a`, `z_Ti` and so on. To see that, collect all these structural
parameters in the column vector $\mathbf{p}$ and denote
by $\mathbf{G}(\mathbf{p})$ the
column vector collecting the derivatives `dE` computed at cutoff `Ecut`,
which depend on the parameters $\mathbf{p}$.
Same as $\mathbf{G}(\mathbf{p})$ also the error estimate $\mathbf{\delta G}(\mathbf{p})$
depends on the parameters $\mathbf{p}$.

We define the auxiliary function
```math
\mathbf{f}(\mathbf{p}, ε) = \mathbf{G}(\mathbf{p}) + ε \mathbf{\delta G}(\mathbf{p}).
```
The idea is that $\mathbf{f}(\mathbf{p}, 0)$ is the gradient
at `Ecut` returned by `batio3_energy_and_gradient` while
$\mathbf{f}(\mathbf{p}, 1)$ is the gradient *plus* the error estimate
as returned by `batio3_energy_and_gradient_with_correction`,
which we assume to be close to the true gradient.

The optimal structure $\mathbf{p}_∗$ is obtained if
```math
\tag{1}
\mathbf{0} = \mathbf{f}(\mathbf{p}_\ast, \varepsilon),
```
which defines an implicit function $\mathbf{p}_∗(ε)$.
Since we know $\mathbf{f}(\mathbf{p}, 0)$ at all $\mathbf{p}$
(this is `batio3_energy_and_gradient`), we are able to compute $\mathbf{p}_∗(0)$.
This is what the function `relax_batio3_full` does.

We could re-run the optimisation with $\mathbf{f}(\mathbf{p}, 1)$,
i.e. using `batio3_energy_and_gradient_with_correction` inside the geometry relaxation
and use the difference $\mathbf{p}_∗(1) - \mathbf{p}_∗(0)$ as our estimate
to the discretisation error in the relaxed parameters.
However, what is more efficient is to consider a first-order development, namely
```math
\mathbf{p}_∗(1) - \mathbf{p}_∗(0) \simeq \mathbf{p}'_∗(0).
```
Computing $\mathbf{p}'_∗$ can be achieved using the implicit function theorem
based on the equation (1) above.
"""

# ╔═╡ a2468c41-f331-42a9-a3ce-7efdccb826d3
md"""
!!! warning "Task 3.2: Derive and implement p'"
	Using the implicit function theorem derive an expression for $p'$
	and implement a routine for computing it.
	For computing the required derivatives it is very useful to employ algorithmic differentiation via the `ForwardDiff` package, e.g.
	```julia 
	using ForwardDiff

	function compute_gradient(p)
		batio3_energy_and_gradient(p; Ecut, kgrid, functionals, pseudopotentials, 
									verbose=true, tol=1e-10,
		                            kinetic_blowup=BlowupIdentity()).dE
	end
	ForwardDiff.jacobian(compute_gradient, mingeo.p)
	```
	where `mingeo` is the result of running your actual structure relaxation
	with `relax_batio3_full`.

	Combine this with your procedure from Task 3.1 to estimate the error in the relaxed parameters of PBE when using `Ecut=30` and `kgrid=(4,4,4)`.
	Compare your findings of the estimated error to your experience of the error behaviour from the convergence study Task 1.2. How well is the error behaviour estimated by the error estimation procedure ? Compare the computational cost of running a convergence study to the computational cost of the error estimate. To what extent is it worth it to consider the error estimate in practice ?

	Since the theory is based on the assumption that $\mathbf{G}(\mathbf{p}_∗, ε) = 0$ it is important to make sure that the relaxation is sufficiently converged. This is the reason the numerical parameters were considerably tightened in the above example compared to the defaults of `batio3_energy_and_gradient`.
"""

# ╔═╡ f5a5651c-5fc7-4a6c-a051-823a4d2db05e
md"""
## Part 4: Propagate structure error to error in bands

In Part 3 we discussed the ability to compute errors in the structural parameters $\mathbf{p}$ originating from an erroneous structure optimisation. 
The goal of this part is to consider the propagation of this error to band structure computations.
"""

# ╔═╡ 2cff4248-4425-4d88-a52a-cb8e6db6e042
md"""
A band structure computation in a given `Ecut`-basis solves a series of eigenvalue problems $\mathcal{H}_{k}(\mathbf{p}, \rho(\mathbf{p})) \psi_{nk} = ε_{nk} \psi_{nk}$. Notably the Hamiltonian $\mathcal{H}_{k}$ depends both directly on $\mathbf{p}$ (via the electron-nuclear interaction) and indirectly, via the self-consistent density $\rho(\mathbf{p})$. Therefore the first-order change of the Hamiltonian contains the two contributions
```math
\begin{align*}
\mathbf{\delta \rho} &= \frac{\partial \rho}{\partial \mathbf{p}} \cdot \mathbf{\delta p} &&\text{(Derivative of SCF)}\\
δ\mathcal{H}_k &= \frac{\partial \mathcal{H}_{k}}{\partial \mathbf{p}} \cdot \mathbf{\delta p} + \frac{\partial \mathcal{H}_{k}}{\partial \rho} \cdot \mathbf{\delta \rho}.&&\text{(Derivative of Hamiltonian)}
\end{align*}
```
By first order perturbation theory, the first-order change of the eigenvalue $δε_{nk}$ due to this change in Hamiltonian is in turn given by
```math
δε_{nk} = \langle \psi_{nk}, \delta\mathcal{H}_{k} \psi_{nk}\rangle.
\qquad \text{(Derivative of band computation)}
```
"""

# ╔═╡ b92cb13e-564c-47c0-9ef9-8ead35870f0d
md"""
Consider the following function
"""

# ╔═╡ c1b0ab62-480f-4d0a-bdb2-bbe458ffadfa
function compute_batio3_bandstructure(p; kgrid, Ecut, kgrid_bands, functionals, pseudopotentials, tol=1e-8, verbose=false)
    system = batio3_tetragonal(p)

    # Define model and run SCF
    model  = model_DFT(system; functionals, pseudopotentials, smearing=Smearing.Gaussian(), temperature=1e-3)
    basis  = PlaneWaveBasis(model; Ecut, kgrid)
    callback = verbose ? ScfDefaultCallback() : identity
    response = ResponseOptions(; verbose)
    scfres = self_consistent_field(basis; tol, callback, response)

    # Compute bands at the k-points kgrid_bands
	bands = compute_bands(basis, kgrid_bands; scfres.ρ, tol=1e-6)

    # Give bands relative to the valence band maximum
    n_occ = model.n_electrons ÷ 2
    vbm = maximum(εk[n_occ] for εk in bands.eigenvalues)
    reduce(hcat, bands.eigenvalues) .- vbm
end

# ╔═╡ e2dba945-1b50-430b-9f73-54a9eb6723e9
md"""
It is exactly a chain of the above three steps: Solve the SCF, build a new Hamiltonian from the self-consistent density `ρ` (happens inside `compute_bands`) then followed by computing the band eigenvalues themselves.

To select the `kgrid_bands` parameter above, it can be useful to re-use the value from a prior band structure computation, which can be obtained using the following. Using the k-points of the band structure you want to plot ensures that the computed eigenvalue errors line up with the plotted bands.
"""

# ╔═╡ 7b8d75e3-68b8-48be-b5c4-9007e2fa2824
bands.basis.kgrid

# ╔═╡ 17c0def2-1ae5-490a-82dc-911b45fb4bd8
md"""
!!! warning "Task 4.1: Differentiate the band structure function"
	We want to understand the error in the band structure caused by
	errors in the structure parameters. 
	In Task 3.2 we used the rough numerical parameters `Ecut=30` and `kgrid=(4,4,4)` 
	to obtain minimal-energy structural parameters $\mathcal{p}$ and associated error estimates $\mathbf{\delta p}$.

	**(a)** Use the same numerical parameters to compute the Jacobian of the function `compute_batio3_bandstructure`. This Jacobian translates $\mathbf{\delta p}$ to an associated error in the band structure. Plot this error as part of the bands plot. See the details below.

	**(b)** Repeat the band computation and the computation of the Jacobian for your converged parameters from Task 1.1. Again use this Jacobian to translate the error to the bands plot. This setting corresponds to the case where a geometry optimisation is run at a lower `Ecut` (here our rough parameters) than the actual band structure computations.

	Comment on your results.
"""

# ╔═╡ 0121df85-b82c-431e-a8de-9de6cdeffe0b
md"""
#### Plotting errors in bands
To plot a band structure with error bars notice the following:
"""

# ╔═╡ 3ae82066-3da1-4838-b43b-773c1576af93
let
	band_data = compute_bands(scfres, kline_density=15)
	
	eigenvalues_error = [0.01 * abs.(randn(size(λk)))
						 for λk in band_data.eigenvalues]  # dummy errors
	data_with_errors = merge(band_data, (; eigenvalues_error))
	plot_bandstructure(data_with_errors)
end

# ╔═╡ 9c0b2f1d-d2a7-45a5-bbb5-f487bb2cf248
md"""
Since `compute_batio3_bandstructure` flattens the eigenvalue data (this is the `reduce(hcat, bands.eigenvalues)` call), its output cannot be directly used as the input to `eigenvalues_error`, one first needs to properly reshape the data. See the example on XC error propagation we discussed in the lecture to see how this can be achieved.
"""

# ╔═╡ 79edbec0-9bc6-47b2-8d3a-eb66aad796b3
md"""
### Band structure cutoff recommendation

!!! warning "Task 4.2: Concluding remarks"
	Using the techniques you developed and the results you have obtained,
	conclude how one should select the cutoff in the geometry optimisation step (Step 1) compared to the cutoff employed for the band computation itself (Step 2). Consider the cases where one wants to target an accuracy in the band structure eigenvalues to $10^{-2}$, $10^{-3}$ or $10^{-4}$ Hartree.

	How would you suggest to run band structure computations in practice ?
"""

# ╔═╡ 09e2cb0b-af42-43a5-93ec-561ea8f6ec50
md"""
## Part 5: Suggesting a faster geometry optimisation

In this part we want to improve the geometry optimisation algorithm
underlying `relax_batio3_full`.

!!! warning "Task 5: Geometry optimisation for perovskites"
    Based on your experience with the geometry optimisation of BaTiO3
    and the ideas suggested below experiment with providing an improved geometry
    optimisation algorithm. Your goal should be to achieve fast structure optimisation
    of BaTiO3 using PBE for the converged
    `kgrid` and `Ecut` parameters obtained in Task 1.1.

    Once you are done, stress-test your algorithm for two related perovskites,
    namely PbTiO₃ and KNbO₃. These follow the same structure as BaTiO3, just with
    different `atoms` arrays in `batio3_tetragonal`, namely
    ```julia
    :Pb => SVector{3,T}(0.0, 0.0, 0   ),
    :Ti => SVector{3,T}(0.5, 0.5, z_Ti),
    :O  => SVector{3,T}(0.5, 0.5, z_O1),     # apical O1)
    :O  => SVector{3,T}(0.5, 0.0, z_O2),     # equatorial O2a
    :O  => SVector{3,T}(0.0, 0.5, z_O2),     # equatorial O2b
    ```
    and
    ```julia
    :K  => SVector{3,T}(0.0, 0.0, 0   ),
    :Nb => SVector{3,T}(0.5, 0.5, z_Ti),
    :O  => SVector{3,T}(0.5, 0.5, z_O1),     # apical O1)
    :O  => SVector{3,T}(0.5, 0.0, z_O2),     # equatorial O2a
    :O  => SVector{3,T}(0.0, 0.5, z_O2),     # equatorial O2b
    ```

    Rough starting points for the initial structures of PbTiO₃ and KNbO₃ are given below.
"""

# ╔═╡ c07725a8-53cc-4248-b924-d2b8996f9634
p_PbTiO3 = ComponentVector(
    a        = 7.379,
    c_over_a = 1.063,
    z_Ti     = 0.540,
    z_O1     = 0.112,
    z_O2     = 0.611,
);

# ╔═╡ 582659bd-5f25-4020-bd29-0f01bfa137a4
p_KNbO3 = ComponentVector(
    a        = 7.551,
    c_over_a = 1.017,
    z_Ti     = 0.482,
    z_O1     = 0.022,
    z_O2     = 0.526,
);

# ╔═╡ 6f376d11-8781-4815-8c4d-7c7576b23492
md"""
#### Some ideas to consider
Here are some ideas for you to try; not all are equally useful and not all of
them are equally easy to do. You are not expected to try all of them, but you
should experiment and report on your findings. Give some reasoning why you
eventually settle on a particular choice.

- Use an optimisation based on a small `Ecut` as an initial guess for an optimisation
  at the `Ecut` of interest. Question is how accurate the first optimisation should
  be converged before switching to the second.
- The same idea also works with `kgrid`.
- The quadratically convergent Newton algorithm can be seen as a steepest descent algorithm
  with the Hessian of the energy (Jacobian of the derivative `dE`) as the preconditioner.
  This suggests preconditioning the LBFGS inside `relax_batio3_full` by such a Hessian,
  see the [Optim documentation](https://julianlsolvers.github.io/Optim.jl/dev/algo/precondition/)
  for more details how to achieve this. Notice that computing the full Hessian is expensive
  and can only be done for very small `Ecut` and `kgrid` values.
- Adapt the tolerances of the SCF along the convergence of the geometry optimisation
  (see `tol` keyword argument in `batio3_energy_and_gradient`).

  This is unfortunately
  a bit annoying to do in Optim. Here is a code snippet, which uses a callback function
  to extract the most recent gradient norm as a measure for the convergence of the
  optimisation:
  ```julia
  function myoptimiser(....)
    gnorm = Ref(Inf)

    callback = state -> begin
        gnorm[] = state.g_norm   # also state.value, state.iteration
        false                    # return true to stop early
    end

    function make_fg(gnorm)
        function fg!(F, G, x)
            # gnorm[] is the gradient norm from the last accepted iteration
            # e.g. adaptive inner tolerance (capped, since gnorm[] = Inf initially)
            tol = min(1e-6, 1e-3 * gnorm[])
            # ... compute F and/or G using tol ...
        end
    end

    res = optimize(Optim.only_fg!(make_fg(gnorm)), x0, LBFGS(),
                   Optim.Options(; callback))
  end
  ```
- Some other idea you can think of ? For example, can the error estimates you developed be used in a creative way to inform your choices for `Ecut` already in the optimisation stage ?
"""

# ╔═╡ 1e5fb45d-4ad0-4d65-9791-281474645317
md"""
## Part 6: Conclusion

To conclude the project, we will zoom out a little from the concrete tasks.

"""

# ╔═╡ 0a667e3e-1b04-4d22-a3f3-f8d0a9df43bd
md"""
!!! warning "Task 6.1: Summary and takeaways from the project"

	Write a short summary of the main outcomes of your project (about one A4 page, can be submitted as a separate PDF).
	
	In your summary you should answer the following questions, making reference to the ideas and examples you explored in this project:
	
	- Provide the key ingredients to efficiently find the band structure of a material. How can error estimation help ? How can thinking about errors help ?
	- Why can it be useful to introduce certain errors (model, discretisation, floating-point) *on purpose* ?
	- Given the effort required to arrive at these results, are you satisfied with the outcome (think about efficiency improvements, sharpness of estimates, usefulness for the geometry optimisation algorithm obtained in Part 5, etc.)
	- Overall, what are your three key achievements in the project. Describe some ideas you could envision testing in follow up work. How could the ideas discussed here be connected to other parts of the class, e.g. the discussion on model error estimation or MLIPs.
"""

# ╔═╡ 9602aa11-8061-4820-ae4f-8b5985863fb8
md"""
!!! warning "Task 6.2: AI declaration"
    For working on this project you have the choice to use LLMs / AI tools (such as Claude or ChatGPT). Please truthfully declare your usage and answer the following questions. The questions and procedures are inspired from the [APS Journal's responsible AI usage website](https://journals.aps.org/authors/appropriate-use-ai-tools).

    - While working on the project you must keep a record for which parts use LLM tools. Any substantive use in the project must be declared.
    - Students should state:
      * LLM tool name and version
      * How the LLM assisted
      * How the students directed (prompting techniques) and verified the LLM output
      * If you used multiple tools all of them have to be declared with a separate answer to the above points for each of the tool.
    - This statement includes the use of LLMs to polish, condense, or otherwise lightly edit code or submitted text.

    - The long-term plan for the evolution of the course is to write a book about its contents. Please therefore do not blindly upload this exercise sheet or the course material to an LLM. Let me also remind you that both the online lecture notes as well as this document are in principle copyrighted material, where you do not own the copyright.
"""

# ╔═╡ e589d763-216e-4d07-8c42-201cf7020c93
md"""
!!! warning "Task 6.3: Group atmosphere and distribution of work"
	Each team member individually should provide 4-5 sentences on their role in the project and the overall experience of working on the project as a group. We will open a separate task on moodle for you to return this feedback there, if you prefer.

	Some questions you should address:
	- Which tasks of the project did you mostly work on ?
	- How did you decide within the group to distribute the workload as such ?
	- In your opinion did each group member contribute equally to the project ?
	- Where could your specific expertise and background from your prior studies contribute most to the project and the exercises we did earlier in the semester?
	- Can you pinpoint aspects about your team members' study subject, which are relevant to the course (either the exercises or this project), which you learned from them in your discussions ?
"""

# ╔═╡ Cell order:
# ╟─436c0704-0cb9-444b-918f-4d26fc05268a
# ╠═86889076-cc43-11f1-b8e1-b1c256252262
# ╟─0d3f3fb2-68f1-4a3f-9e1b-4c61114a09cd
# ╟─a9740f44-86b4-4c7c-84ef-ba065e2be93e
# ╟─9a721598-bcec-4071-a774-3b2d9dd77eb7
# ╟─c83b7e77-28a3-4d1f-a315-2c59eee5134e
# ╟─c93bab99-81f2-4e28-adb6-33c890644ac7
# ╠═9c630c72-27fb-4d46-a1fa-44802f12553d
# ╟─4bd0d4f5-de56-4a8e-9c4d-36004073c411
# ╟─a48ace23-d7e9-416d-bfac-ab3f636e0cd8
# ╟─9e431fe5-7b7c-4217-bbf3-861f3f6a0a2a
# ╠═a78e1830-f205-4f7b-9d68-3d183af311ae
# ╟─52f0babb-6ed4-4687-a2ed-27cb777a9482
# ╟─490f6e59-e04a-44a1-9625-520a3929d57e
# ╟─d3ec45b6-c8bf-4d40-9131-84d9a9653601
# ╠═849bc196-89e0-4cf8-856d-a0943792657c
# ╠═e11b06bb-0c75-49df-9985-1c555d771946
# ╟─c089aa93-91ab-41fe-84d9-759222d12cb3
# ╠═d531968e-04a2-4f9b-a1c8-1be62018bf30
# ╠═e9a5d384-7e2d-45a2-ac08-a34eaae18078
# ╟─a578c81c-8e47-4b25-8017-ef694cd1b13d
# ╠═4376cf04-2c93-4f3f-a89f-2defe163078a
# ╠═15e29343-30f6-4d68-bfa3-d14283985a25
# ╟─283a97c9-8975-4807-9b5c-c303f4a830b8
# ╠═e06f1089-56fe-409d-bd87-beb9895add85
# ╠═f724b839-2d27-45aa-a251-27462124de7f
# ╟─6a8103d6-e38b-4fe1-91bf-9b312c848ccd
# ╠═2376a93f-1472-41ca-a867-6e3c52f653a0
# ╠═09740792-4a26-4743-9215-c3ec0fea0dd8
# ╠═9bffe37b-6322-475e-87ba-f9a7a58adeb3
# ╠═d9f5b2cd-056d-44ce-9d06-e1a3a408258c
# ╟─2bc36bfb-1b8e-402a-b6a4-93adf89cdc15
# ╟─8020dfc7-c8dd-43d1-8f4c-09a3515b5036
# ╟─7653c7b4-3dd4-4c04-82e1-b31a684e78d0
# ╟─0628298f-2755-4be3-9d80-10ffd862f4c6
# ╠═bd40d8f5-4376-4c55-a6a0-5ba87a42518a
# ╠═551581a6-1e93-4e7b-9dd9-097aa753d702
# ╠═e431bf8e-7cde-4a09-8e4a-089d635bfea4
# ╠═6d7c5cb5-8951-4148-854c-23a68a565fe9
# ╟─82f11096-2fff-4ecf-b8fd-b514474c793c
# ╟─2469deb7-c537-47cd-9430-ff7643c9e279
# ╟─77289233-8009-4e39-b91c-f3ac94a65086
# ╟─fc4995d7-b5b7-4c7b-a67a-99a5e4889f04
# ╠═f86b7848-100c-4560-a1af-f75809d72272
# ╟─ec8b593f-5c6e-4a8a-bcf8-4e6475582e8d
# ╠═7a025b59-c8fc-4028-aa47-a9c297c4a7ac
# ╟─067523fc-c17d-4f80-9b6b-e31eaac7ef60
# ╟─a4371699-019a-484e-a9b9-e5a0bf761dcf
# ╠═e2d80859-b26e-42e0-bbfc-5eaef7fb0467
# ╠═4df01373-0858-44b6-9ea6-4ff1d5b344b0
# ╟─36f9e1da-9677-476c-8516-b321762e9719
# ╠═1b6bc6c4-d62d-4f81-bbb5-948d6be1bd39
# ╠═c2942550-427f-4c71-af72-f468d7d4bb26
# ╟─eba3e75e-ba36-4d26-9399-07f681e6e4d1
# ╠═055d2b4e-6bea-41ee-a9b7-1c5e1eea6587
# ╠═07abf477-595f-4939-8217-3a83cb24c24d
# ╠═af9660d4-d88c-4d8c-b7f2-ba4c207d9367
# ╠═98069eca-c0cc-4a3c-b000-e7e6854790f2
# ╟─4ed1f9ed-d404-4a1c-8613-652d1cdf9af5
# ╟─8ec195ed-3d64-4260-bfd3-94b6c985db7d
# ╠═2d2375bd-7575-48b9-86ba-6960c0954915
# ╟─6ff5fa3c-136c-4f6b-a56c-29dea1b09262
# ╟─0cf5d9e3-e13e-4e4b-8660-14d4c7670fdc
# ╠═e221fb8e-dcf0-405a-9723-ce1ff95a8306
# ╟─aa80aa2c-f97f-4a9c-b294-c4698e8fb1eb
# ╟─b0b544c6-9538-4608-a322-f11fe5175c7d
# ╠═4c96c0c3-7e5f-45fa-87d4-d2d09a8473ed
# ╟─c4138d41-06fa-4864-9f72-e7dc50ec258d
# ╠═f73a742f-51f2-4cef-a8f1-b4f1ead0d03b
# ╠═74dee5ee-0def-4e64-bde2-7b899be4307e
# ╟─c7caf91b-6d1d-4f56-9a79-62be2bc507b3
# ╟─8e1288c0-8fea-4200-a2b3-c9012586e982
# ╟─a2468c41-f331-42a9-a3ce-7efdccb826d3
# ╟─f5a5651c-5fc7-4a6c-a051-823a4d2db05e
# ╟─2cff4248-4425-4d88-a52a-cb8e6db6e042
# ╟─b92cb13e-564c-47c0-9ef9-8ead35870f0d
# ╠═c1b0ab62-480f-4d0a-bdb2-bbe458ffadfa
# ╟─e2dba945-1b50-430b-9f73-54a9eb6723e9
# ╠═7b8d75e3-68b8-48be-b5c4-9007e2fa2824
# ╟─17c0def2-1ae5-490a-82dc-911b45fb4bd8
# ╟─0121df85-b82c-431e-a8de-9de6cdeffe0b
# ╠═3ae82066-3da1-4838-b43b-773c1576af93
# ╟─9c0b2f1d-d2a7-45a5-bbb5-f487bb2cf248
# ╟─79edbec0-9bc6-47b2-8d3a-eb66aad796b3
# ╟─09e2cb0b-af42-43a5-93ec-561ea8f6ec50
# ╠═c07725a8-53cc-4248-b924-d2b8996f9634
# ╠═582659bd-5f25-4020-bd29-0f01bfa137a4
# ╟─6f376d11-8781-4815-8c4d-7c7576b23492
# ╟─1e5fb45d-4ad0-4d65-9791-281474645317
# ╟─0a667e3e-1b04-4d22-a3f3-f8d0a9df43bd
# ╟─9602aa11-8061-4820-ae4f-8b5985863fb8
# ╟─e589d763-216e-4d07-8c42-201cf7020c93

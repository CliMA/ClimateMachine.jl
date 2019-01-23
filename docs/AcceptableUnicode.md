# Acceptable Unicode characters:
Using Unicode seems to be irresistible. However, we must ensure
avoiding problematic Unicode usage. Here is a list of acceptable
and forbidden Unicode characters and accents. Please note that
we forbid the use of accents, because this can lead to visually
ambiguous characters (see below).

## Acceptable lower-case Greek letters:
α # (alpha)
β # (beta)
δ # (delta)
ϵ # (epsilon)
γ # (gamma)
κ # (kappa)
λ # (lambda)
μ # (mu)
η # (eta)
ω # (omega)
π # (pi)
ρ # (rho)
σ # (sigma)
θ # (theta)
χ # (chi)
ξ # (xi)
ζ # (zeta)
ϕ # (psi)
φ # (varphi)

## Acceptable upper-case Greek letters:
Δ # (Delta)
∑ # (Sigma)
Γ # (Gamma)
Ω # (Omega)
Φ # (Phi)
Ψ # (Psi)

## Acceptable mathematical symbols:
∫ # (int)
∬ # (iint)
∭ # (iiint)
∈ # (in)
∞ # (infinity)
≈ # (approx)
∂ # (partial)
∇ # (nabla/del), note that nabla and del are indistinguishable
∀ # (forall)
≥ # (greater than equal to)
≤ # (less than equal to)

## Indistinguishable characters
Some characters are indistinguishable. For example:

  ∇ = 1 # nabla
  ∇ = 2 # del
  print(∇,"\n") # nabla prints 2
  print(∇,"\n") # del   prints 2


# On-the-fence characters:
Some characters are visually very close to others (e.g., "vee"
and "nu"), and while using them may result in a more consistent
and systematic use of symbolic variable and function names, we
may want to limit their use to avoid confusion.

  υ # (upsilon)
  ν # (nu)

# Forbidden characters/accents

## Characters:
 All characters not listed above are forbidden because some
 characters are visibly indistinguishable. Capital "a" and
 capital alpha are visibly indistinguishable, but are
 recognized as separate characters (e.g., search distinguishable).
 We must avoid problematic character-clashing like the example here:

  A = 1 # Capital "A"
  Α = 2 # Capital "Alpha"
  print(A) # prints 1
  print(Α) # prints 2

## Accents:
 For now, all accents (dot, hat, vec, etc.) are forbidden because
 of the listed reasons in the coding conventions. We must avoid
 unsafe variable names/combinations like the example here:

  ν⃗     (nu_vec)
  v⃗     (v_vec)

Also, these accents seem to have issues rendering in browsers. To
properly view this example, open this file in a text-editor.


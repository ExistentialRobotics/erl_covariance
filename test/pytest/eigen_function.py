import sympy as sp

sp.init_printing(wrap_line=False)

x, y = sp.symbols("x y")
L = sp.symbols("L", positive=True)
i, j = sp.symbols("i j", integer=True, positive=True)

phi_ij = sp.sin(sp.pi * i * (x + L) / (2 * L)) * sp.sin(sp.pi * j * (y + L) / (2 * L)) / L
print("Eigenfunction:")
sp.pretty_print(phi_ij)
neg_laplace_phi_ij = -sp.diff(phi_ij, x, 2) - sp.diff(phi_ij, y, 2)
neg_laplace_phi_ij = sp.simplify(neg_laplace_phi_ij)
print("-Laplacian:")
sp.pretty_print(neg_laplace_phi_ij)
lambda_ij = neg_laplace_phi_ij / phi_ij
print("Eigenvalue:")
sp.pretty_print(lambda_ij)  # eigenvalue
frequency_ij = sp.sqrt(lambda_ij)
print("Frequency:")
sp.pretty_print(frequency_ij)  # frequency

diff_phi_ij_x = sp.diff(phi_ij, x)
alpha_ij_x = diff_phi_ij_x / phi_ij
sp.pretty_print(diff_phi_ij_x)
sp.pretty_print(alpha_ij_x)

# print("===============================================")
# phi_ij = sp.sin(sp.pi * (i * (x + L) + j * (y + L)) / (2 * L)) / L
# print("Eigenfunction:")
# sp.pretty_print(phi_ij)
# neg_laplace_phi_ij = -sp.diff(phi_ij, x, 2) - sp.diff(phi_ij, y, 2)
# neg_laplace_phi_ij = sp.simplify(neg_laplace_phi_ij)
# print("-Laplacian:")
# sp.pretty_print(neg_laplace_phi_ij)
# lambda_ij = neg_laplace_phi_ij / phi_ij
# print("Eigenvalue:")
# sp.pretty_print(lambda_ij)  # eigenvalue
# frequency_ij = sp.sqrt(lambda_ij)
# print("Frequency:")
# sp.pretty_print(frequency_ij)  # frequency

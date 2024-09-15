import sympy as sp

alpha, l = sp.symbols("alpha l", positive=True)
x1, y1, x2, y2 = sp.symbols("x1 y1 x2 y2")

r2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
k = (1 + r2 / (2 * alpha * l * l)) ** (-alpha)

k = k.simplify()

diff_x = sp.Matrix(
    [
        sp.diff(k, x1),
        sp.diff(k, y1),
    ]
)
diff_x = sp.simplify(diff_x)

diff_x_prime = sp.Matrix(
    [
        sp.diff(k, x2),
        sp.diff(k, y2),
    ]
)
diff_x_prime = sp.simplify(diff_x_prime)

diff_x_x_prime = sp.Matrix(
    [
        [sp.diff(k, x1, x2), sp.diff(k, x1, y2)],
        [sp.diff(k, y1, x2), sp.diff(k, y1, y2)],
    ]
)
diff_x_x_prime = sp.simplify(diff_x_x_prime)

sp.init_printing(wrap_line=False)
print("grad_x:")
sp.pretty_print(sp.simplify(diff_x / k))

print("grad_x_prime:")
sp.pretty_print(sp.simplify(diff_x_prime / k))

print("grad_x_x_prime:")
sp.pretty_print(sp.simplify(diff_x_x_prime / k))

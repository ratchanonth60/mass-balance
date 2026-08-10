import sympy as sp

r, p, y, r_dot, p_dot, y_dot = sp.symbols('r p y r_dot p_dot y_dot', real=True)
d1, d2, d3, d4 = sp.symbols('d1 d2 d3 d4', real=True)
R0z = sp.symbols('R0z', real=True)

gi = sp.Matrix([0, 0, sp.Float(-9.81, 20)])
M = sp.Float(17.8, 20)
m = sp.Float(0.25, 20)
Mt = M + 4*m

Jx = sp.Float(1.84813686228, 20)
Jy = sp.Float(1.84813686228, 20)
Jz = sp.Float(2.99520457469, 20)
J = sp.diag(Jx, Jy, Jz)

cxy = sp.cos(sp.rad(60))*sp.cos(sp.rad(45))
cz = sp.sin(sp.rad(60))

c1 = sp.Matrix([-cxy, -cxy, cz]); c2 = sp.Matrix([-cxy, cxy, cz])
c3 = sp.Matrix([cxy, cxy, cz]);   c4 = sp.Matrix([cxy, -cxy, cz])

b1 = sp.Matrix([sp.Float(0.3,20),  sp.Float(0.3,20), 0])
b2 = sp.Matrix([sp.Float(0.3,20), -sp.Float(0.3,20), 0])
b3 = sp.Matrix([-sp.Float(0.3,20), -sp.Float(0.3,20), 0])
b4 = sp.Matrix([-sp.Float(0.3,20),  sp.Float(0.3,20), 0])

R0 = sp.Matrix([0, 0, R0z])
d0 = sp.Float(0.20, 20)

x_att = sp.Matrix([r, p, r_dot, p_dot, y_dot])
d = sp.Matrix([d1, d2, d3, d4])
w = sp.Matrix([r_dot, p_dot, y_dot])

y_val = 0
R_ib = sp.Matrix([
    [sp.cos(y_val)*sp.cos(p), sp.cos(y_val)*sp.sin(r)*sp.sin(p)-sp.cos(r)*sp.sin(y_val), sp.sin(r)*sp.sin(y_val)+sp.cos(r)*sp.cos(y_val)*sp.sin(p)],
    [sp.cos(p)*sp.sin(y_val), sp.cos(r)*sp.cos(y_val)+sp.sin(r)*sp.sin(y_val)*sp.sin(p), sp.cos(r)*sp.sin(y_val)*sp.sin(p)-sp.cos(y_val)*sp.sin(r)],
    [-sp.sin(p), sp.cos(p)*sp.sin(r), sp.cos(r)*sp.cos(p)],
])
gb = R_ib * gi

r1m = b1 + c1*(d1 - d0); r2m = b2 + c2*(d2 - d0)
r3m = b3 + c3*(d3 - d0); r4m = b4 + c4*(d4 - d0)
Rtot = (M*R0 + m*(r1m + r2m + r3m + r4m)) / Mt

Tq_inertia = -w.cross(J*w)
Tq_CoM = Mt * Rtot.cross(gb)
w_dot = J.inv() * (Tq_inertia + Tq_CoM)
F_att = sp.Matrix([r_dot, p_dot, w_dot[0], w_dot[1], w_dot[2]])

A = F_att.jacobian(x_att)
E = F_att.jacobian(d)

x_op = {r:0, p:0, r_dot:0, p_dot:0, y_dot:0}
A_op_expr = A.subs(x_op)
E_op_expr = E.subs(x_op)

def evalmat(mat, subs):
    return mat.subs(subs).evalf(20)

A_base = evalmat(A_op_expr, {d1:0,d2:0,d3:0,d4:0,R0z:0})
def unit_delta(sym):
    s = {d1:0,d2:0,d3:0,d4:0,R0z:0}
    s[sym] = 1
    return evalmat(A_op_expr, s) - A_base

A_d1 = unit_delta(d1); A_d2 = unit_delta(d2); A_d3 = unit_delta(d3); A_d4 = unit_delta(d4); A_r0z = unit_delta(R0z)
E_const = evalmat(E_op_expr, {})

def rust_array(mat, name):
    rows, cols = mat.shape
    lines = [f"pub const {name}: [[f64; {cols}]; {rows}] = ["]
    for i in range(rows):
        vals = ", ".join(f"{float(mat[i,j]):.17e}" for j in range(cols))
        lines.append(f"    [{vals}],")
    lines.append("];")
    return "\n".join(lines)

for name, mat in [("A_BASE", A_base), ("A_D1", A_d1), ("A_D2", A_d2), ("A_D3", A_d3), ("A_D4", A_d4), ("A_R0Z", A_r0z), ("E_CONST", E_const)]:
    print(rust_array(mat, name))
    print()

# residual check with high precision at a fresh random point
import random
random.seed(42)
dv = {d1: random.random(), d2: random.random(), d3: random.random(), d4: random.random(), R0z: random.random()*0.01}
A_direct = evalmat(A_op_expr, dv)
A_recon = A_base + dv[d1]*A_d1 + dv[d2]*A_d2 + dv[d3]*A_d3 + dv[d4]*A_d4 + dv[R0z]*A_r0z
resid = (A_direct - A_recon)
maxabs = max(abs(resid[i,j]) for i in range(5) for j in range(5))
print("max abs residual (affine check):", maxabs)

import sympy as sp

r, p, y, r_dot, p_dot, y_dot = sp.symbols('r p y r_dot p_dot y_dot', real=True)
ux, uy, uz = sp.symbols('ux uy uz', real=True)
rm1, rm2, rm3, rm4 = sp.symbols('rm1 rm2 rm3 rm4', real=True)

gi = sp.Matrix([0, 0, sp.Float(-9.81, 20)])
M = sp.Float(5.18, 20)
m = sp.Float(0.25, 20)
Mt = M + 4*m

Jx = sp.Float(0.184813686228, 20) * 10
Jy = sp.Float(0.184813686228, 20) * 10
Jz = sp.Float(0.299520457469, 20) * 10
J = sp.diag(Jx, Jy, Jz)

cxy = sp.cos(sp.rad(60))*sp.cos(sp.rad(45))
cz = sp.cos(sp.rad(60))  # InitLQR.m: cz = cosd(60)

dir1 = sp.Matrix([-cxy, -cxy, cz])
dir2 = sp.Matrix([ cxy, -cxy, cz])
dir3 = sp.Matrix([-cxy,  cxy, cz])
dir4 = sp.Matrix([ cxy,  cxy, cz])

R0 = sp.Matrix([0, 0, 0])

x = sp.Matrix([r, p, r_dot, p_dot, y_dot])
rm = sp.Matrix([rm1, rm2, rm3, rm4])
u = sp.Matrix([ux, uy, uz])

w = sp.Matrix([r_dot, p_dot, y_dot])
Tq_inertia = (-w).cross(J*w)
y_val = 0  # InitLQR.m:39 y = 0 shadows symbolic y
Rtot = (M*R0 + m*dir1*rm[0] + m*dir2*rm[1] + m*dir3*rm[2] + m*dir4*rm[3]) / Mt
R_ib = sp.Matrix([
    [sp.cos(y_val)*sp.cos(p), sp.cos(y_val)*sp.sin(r)*sp.sin(p)-sp.cos(r)*sp.sin(y_val), sp.sin(r)*sp.sin(y_val)+sp.cos(r)*sp.cos(y_val)*sp.sin(p)],
    [sp.cos(p)*sp.sin(y_val), sp.cos(r)*sp.cos(y_val)+sp.sin(r)*sp.sin(y_val)*sp.sin(p), sp.cos(r)*sp.sin(y_val)*sp.sin(p)-sp.cos(y_val)*sp.sin(r)],
    [-sp.sin(p), sp.cos(p)*sp.sin(r), sp.cos(r)*sp.cos(p)],
])
gb = R_ib * gi
Tq_CoM = Rtot.cross(Mt*gb)

Tq_tot = Tq_inertia + Tq_CoM + u
w_dot = J.inv() * Tq_tot
F = sp.Matrix([r_dot, p_dot, w_dot[0], w_dot[1], w_dot[2]])

A = F.jacobian(x)
B = F.jacobian(u)

op = {r:0,p:0,r_dot:0,p_dot:0,y_dot:0, rm1:0,rm2:0,rm3:0,rm4:0}
A_op = A.subs(op).evalf(20)
B_op = B.subs({ux:0,uy:0,uz:0}).evalf(20)   # B doesn't depend on u or rm/x since it's linear in u

print("A_op =")
sp.pprint(A_op)
print("B_op =")
sp.pprint(B_op)

def rust_array(mat, name):
    rows, cols = mat.shape
    lines = [f"pub const {name}: [[f64; {cols}]; {rows}] = ["]
    for i in range(rows):
        vals = ", ".join(f"{float(mat[i,j]):.17e}" for j in range(cols))
        lines.append(f"    [{vals}],")
    lines.append("];")
    return "\n".join(lines)

print(rust_array(A_op, "A_OP"))
print(rust_array(B_op, "B_OP"))

# --- ZOH discretization Ts=1, via matrix exponential of [[A,B],[0,0]] ---
import numpy as np
from scipy.linalg import expm

A_np = np.array(A_op.tolist(), dtype=float)
B_np = np.array(B_op.tolist(), dtype=float)
n = A_np.shape[0]; mdim = B_np.shape[1]
Ts = 1.0
M_aug = np.zeros((n+mdim, n+mdim))
M_aug[:n,:n] = A_np
M_aug[:n,n:] = B_np
Md = expm(M_aug*Ts)
Ad = Md[:n,:n]
Bd = Md[:n,n:]
np.set_printoptions(precision=17, suppress=False, floatmode='maxprec_equal')
print("Ad (discretized) =")
print(Ad)
print("Bd (discretized) =")
print(Bd)

def rust_array_np(mat, name):
    rows, cols = mat.shape
    lines = [f"pub const {name}: [[f64; {cols}]; {rows}] = ["]
    for i in range(rows):
        vals = ", ".join(f"{mat[i,j]:.17e}" for j in range(cols))
        lines.append(f"    [{vals}],")
    lines.append("];")
    return "\n".join(lines)

print(rust_array_np(Ad, "AD_OP"))
print(rust_array_np(Bd, "BD_OP"))

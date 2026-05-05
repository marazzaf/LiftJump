from firedrake import *
from lifting import *

#Mesh
N = 5
mesh = UnitSquareMesh(N, N, diagonal='crossed')
x, y = SpatialCoordinate(mesh)

#Function Spaces
k = 2
V = FunctionSpace(mesh, "DG", k)
W = VectorFunctionSpace(mesh, 'DG', k-1)

#test
u = Function(V)
op = LiftJump(u, function_space=W)
print(type(assemble(op)))
print(assemble(op).function_space())
op = derivative(LiftJump(u, function_space=W), u)
print(type(assemble(op)))
print(assemble(op).ufl_function_spaces())
v = TestFunction(V)
op = derivative(LiftJump(u, function_space=W), u, v)
print(type(assemble(op)))
print(assemble(op).ufl_function_spaces())

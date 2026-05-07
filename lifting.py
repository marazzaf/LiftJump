from firedrake import *
from firedrake.external_operators import AbstractExternalOperator, assemble_method


class LiftJump(AbstractExternalOperator):

    def __init__(self, *operands, function_space, **kwargs):
        AbstractExternalOperator.__init__(self, *operands, function_space=function_space, **kwargs)
    
    def assemble_matrix(self, w):
        Sigma = self.function_space() #W
        mesh = Sigma.mesh()
        tau = TestFunction(Sigma)
        n = FacetNormal(mesh)
        OG_space = w.function_space() #V
        u = TrialFunction(OG_space)
        return inner(jump(u), dot(avg(tau), n("+"))) * dS(mesh) + inner(u, dot(tau, n)) * ds(mesh)

    @assemble_method(0, (0,))
    def assemble_operator(self, *args, **kwargs):
        """Assemble LiftJump(u)."""
        (u,) = self.ufl_operands
        a = self.assemble_matrix(u)
        cofunc = assemble(action(a, u))
        return cofunc.riesz_representation() #function in primal space

    @assemble_method((1,), (0, 1))
    def assemble_Jacobian(self, *args, **kwargs):
        """Since the operator is linear this is similar to just the assembly."""
        (w,) = self.ufl_operands

        Sigma = self.function_space() #W
        mesh = Sigma.mesh()
        #tau = TestFunction(Sigma)
        tau = TestFunction(Sigma.dual())
        n = FacetNormal(mesh)
        OG_space = w.function_space() #V
        u = TrialFunction(OG_space)
        a = inner(jump(u), dot(avg(tau), n("+"))) * dS(mesh) + inner(u, dot(tau, n)) * ds(mesh)
        
        A = assemble(a)
        return A



import sympy as sp
from IPython.display import display, Markdown
from sympy.tensor.tensor import (
    TensorIndexType, TensorHead, TensorSymmetry,
    tensor_indices, canon_bp, contract_metric, TensExpr
)

sp.init_printing()


class StepStore(dict):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx

    def put(self, key, expr, simplify=True):
        self[key] = self.ctx.tsimplify(expr) if simplify else expr
        return self[key]

    def show(self, key, title=None):
        if title is None:
            title = key
        display(Markdown(f"#### {title}\nObjeto reutilizable: `S[{key!r}]`"))
        display(self[key])
        return self[key]

    def check_zero(self, key, expr, title=None):
        result = self.put(key, expr)
        if title is None:
            title = key
        display(Markdown(f"#### Verificación: {title}\nObjeto: `S[{key!r}]`"))
        display(result)
        if result != 0:
            raise AssertionError(f"La verificación {key} no dio cero.")
        return result


class FRContext:
    """Estado compartido del notebook.

    Contiene únicamente las definiciones algebraicas, símbolos, tensores,
    utilidades y plantillas reutilizadas entre etapas. Las deducciones
    concretas se ejecutan en los módulos de etapas.
    """

    def __init__(self):
        # ================================================================
        # 1. Geometría abstracta
        # ================================================================
        self.M = TensorIndexType(
            "M",
            dummy_name="0",
            metric_symmetry=1,
            metric_name=r"\mathrm{g}",
        )
        self.g = self.M.metric

        sym0 = TensorSymmetry.no_symmetry
        symS = TensorSymmetry.fully_symmetric
        symDP = TensorSymmetry.direct_product

        self.Riem = TensorHead(r"\mathrm{R}", [self.M]*4, TensorSymmetry.riemann())
        self.dRiem = TensorHead(r"\delta\mathrm{R}", [self.M]*4, TensorSymmetry.riemann())

        self.xi = TensorHead(r"\xi", [self.M], sym0(1))
        self.Dxi = TensorHead(r"\nabla\xi", [self.M]*2, sym0(2))

        self.DRiem = TensorHead(r"\nabla\mathrm{R}", [self.M]*5, sym0(5))

        self.h = TensorHead(r"\delta\mathrm{g}", [self.M]*2, symS(2))
        self.H = TensorHead(r"\mathrm{H}", [self.M]*2, symS(2))
        self.Dh = TensorHead(r"\nabla\delta\mathrm{g}", [self.M]*3, symDP(1, 2))
        self.DDh = TensorHead(r"\nabla\nabla\delta\mathrm{g}", [self.M]*4, symDP(1, 1, 2))

        self.DGamma = TensorHead(r"\nabla\delta\Gamma", [self.M]*4, symDP(1, 1, 2))

        self.DR = TensorHead(r"\nabla R", [self.M], sym0(1))
        self.DDR = TensorHead(r"\nabla\nabla R", [self.M]*2, symS(2))

        self.sqrtg = sp.Symbol("sqrt_minus_g", positive=True)

        # ================================================================
        # 2. Símbolos escalares permitidos en el input
        # ================================================================
        self.R = sp.Symbol("R", real=True)
        self.alpha, self.beta, self.gamma, self.Lambda, self.mu = sp.symbols(
            "alpha beta gamma Lambda mu", real=True
        )

        # Estado calculado en las etapas
        self.S = StepStore(self)
        self.L_input = None
        self.f1 = None
        self.f2 = None
        self.f3 = None

        self.P_TEMPLATE_INDICES = None
        self.P_PROJECTOR_TEMPLATE = None
        self.P_TEMPLATE = None

        self.P_METRIC_TEMPLATE_INDICES = None
        self.P_METRIC_TEMPLATE = None

    # ================================================================
    # 3. Utilidades de álgebra tensorial
    # ================================================================
    def tsimplify(self, expr, max_iter=8):
        """Canoniza índices mudos, contrae métricas y simplifica coeficientes."""
        if expr == 0:
            return sp.S.Zero

        if isinstance(expr, TensExpr):
            cur = sp.expand(expr)
            prev = None
            for _ in range(max_iter):
                cur = canon_bp(cur)
                cur = contract_metric(cur, self.g)
                cur = canon_bp(cur)
                if cur == prev:
                    break
                prev = cur
            return cur

        return sp.factor(sp.simplify(expr))

    @staticmethod
    def swap(expr, i, j):
        return expr.xreplace({i: j, j: i})

    def antisym(self, expr, i, j):
        return sp.Rational(1, 2) * (expr - self.swap(expr, i, j))

    @staticmethod
    def pair_exchange(expr, a, b, c, d):
        repl = {a: c, b: d, c: a, d: b}
        return expr.xreplace(repl)

    def pair_sym(self, expr, a, b, c, d):
        return sp.Rational(1, 2) * (
            expr + self.pair_exchange(expr, a, b, c, d)
        )

    def curvature_projector(self, a, b, c, d, return_steps=False):
        q0 = self.g(a, c) * self.g(b, d)
        q1 = self.tsimplify(self.antisym(q0, a, b))
        q2 = self.tsimplify(self.antisym(q1, c, d))
        q3 = self.tsimplify(self.pair_sym(q2, a, b, c, d))
        if return_steps:
            return q0, q1, q2, q3
        return q3

    def dginv_dgcov(self, x, y, p, q):
        return -sp.Rational(1, 2) * (
            self.g(x, p)*self.g(y, q) + self.g(x, q)*self.g(y, p)
        )

    def h_from_H(self, a, b):
        m, n = tensor_indices("hH_m hH_n", self.M)
        return -self.g(-a, -m) * self.g(-b, -n) * self.H(m, n)

    def scalar_covd(self, expr, a):
        return sp.diff(expr, self.R) * self.DR(-a)

    def scalar_hessian(self, expr, a, b):
        return (
            sp.diff(expr, self.R) * self.DDR(-a, -b)
            + sp.diff(expr, self.R, 2) * self.DR(-a) * self.DR(-b)
        )

    # ================================================================
    # Plantillas que aparecen progresivamente en la deducción
    # ================================================================
    @staticmethod
    def _reindex4(expr, old, new):
        return expr.xreplace(dict(zip(old, new)))

    def set_P_template(self, indices, projector):
        self.P_TEMPLATE_INDICES = indices
        self.P_PROJECTOR_TEMPLATE = projector
        self.P_TEMPLATE = self.f1 * projector

    def P_up(self, a, b, c, d):
        if self.P_TEMPLATE is None:
            raise RuntimeError("P^{abcd} todavía no ha sido construido.")
        return self._reindex4(
            self.P_TEMPLATE,
            self.P_TEMPLATE_INDICES,
            (a, b, c, d),
        )

    def set_P_metric_template(self, indices, expr):
        self.P_METRIC_TEMPLATE_INDICES = indices
        self.P_METRIC_TEMPLATE = expr

    def P_metric_up(self, a, b):
        if self.P_METRIC_TEMPLATE is None:
            raise RuntimeError("P^{ab} todavía no ha sido construido.")
        p, q = self.P_METRIC_TEMPLATE_INDICES
        return self.P_METRIC_TEMPLATE.xreplace({p: a, q: b})

    def DGamma_from_h(self, e, c, d, b):
        i = tensor_indices("DG_i", self.M)
        return sp.Rational(1, 2) * self.g(e, i) * (
            self.DDh(-c, -d, -b, -i)
            + self.DDh(-c, -b, -d, -i)
            - self.DDh(-c, -i, -d, -b)
        )

    def DP_up(self, e, a, b, c, d):
        """∇_e P^{abcd}, calculado desde el P del input."""
        if self.P_PROJECTOR_TEMPLATE is None:
            raise RuntimeError("P^{abcd} todavía no ha sido construido.")
        return self.scalar_covd(self.f1, e) * self.P_PROJECTOR_TEMPLATE.xreplace(
            dict(zip(self.P_TEMPLATE_INDICES, (a, b, c, d)))
        )

    def DDP_up(self, e, f, a, b, c, d):
        """∇_e∇_f P^{abcd}, calculado desde el P del input."""
        if self.P_PROJECTOR_TEMPLATE is None:
            raise RuntimeError("P^{abcd} todavía no ha sido construido.")
        return self.scalar_hessian(self.f1, e, f) * self.P_PROJECTOR_TEMPLATE.xreplace(
            dict(zip(self.P_TEMPLATE_INDICES, (a, b, c, d)))
        )

    def expose_user_symbols(self):
        """Devuelve los símbolos que el usuario necesita para definir L_input."""
        return (
            self.R,
            self.alpha,
            self.beta,
            self.gamma,
            self.Lambda,
            self.mu,
        )

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


class FRGBContext:
    """Estado compartido para Lagrangianos explícitos L(R, GB).

    GB representa el invariante de Gauss--Bonnet

        GB = R_{abcd}R^{abcd} - 4 R_{ab}R^{ab} + R^2.

    El motor conserva la misma lógica variacional del notebook f(R), pero
    construye P^{abcd} como suma de dos estructuras calculadas:

        P = L_R * dR/dRiemann + L_GB * dGB/dRiemann.
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

        # Derivadas abstractas de los invariantes escalares.
        self.DR = TensorHead(r"\nabla R", [self.M], sym0(1))
        self.DDR = TensorHead(r"\nabla\nabla R", [self.M]*2, symS(2))
        self.DGB = TensorHead(r"\nabla\mathcal{G}", [self.M], sym0(1))
        self.DDGB = TensorHead(r"\nabla\nabla\mathcal{G}", [self.M]*2, symS(2))

        self.sqrtg = sp.Symbol("sqrt_minus_g", positive=True)

        # ================================================================
        # 2. Símbolos escalares permitidos en el input
        # ================================================================
        self.R = sp.Symbol("R", real=True)
        self.GB = sp.Symbol("GB", real=True)
        self.alpha, self.beta, self.gamma, self.Lambda, self.mu = sp.symbols(
            "alpha beta gamma Lambda mu", real=True
        )

        self.scalar_invariants = (self.R, self.GB)
        self.scalar_gradients = {self.R: self.DR, self.GB: self.DGB}
        self.scalar_hessians = {self.R: self.DDR, self.GB: self.DDGB}

        # Estado calculado en las etapas
        self.S = StepStore(self)
        self.L_input = None
        self.L_R = None
        self.L_GB = None

        # Componentes P = sum_A coefficient_A * template_A
        self.P_COMPONENTS = []

    # ================================================================
    # 3. Utilidades de álgebra tensorial
    # ================================================================
    def tsimplify(self, expr, max_iter=8):
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

    def riemann_project(self, expr, a, b, c, d, return_steps=False):
        p1 = self.tsimplify(self.antisym(expr, a, b))
        p2 = self.tsimplify(self.antisym(p1, c, d))
        p3 = self.tsimplify(self.pair_sym(p2, a, b, c, d))
        if return_steps:
            return expr, p1, p2, p3
        return p3

    def dginv_dgcov(self, x, y, p, q):
        return -sp.Rational(1, 2) * (
            self.g(x, p)*self.g(y, q) + self.g(x, q)*self.g(y, p)
        )

    def h_from_H(self, a, b):
        m, n = tensor_indices("hH_m hH_n", self.M)
        return -self.g(-a, -m) * self.g(-b, -n) * self.H(m, n)

    def scalar_covd(self, expr, a):
        result = sp.S.Zero
        for scalar in self.scalar_invariants:
            result += sp.diff(expr, scalar) * self.scalar_gradients[scalar](-a)
        return self.tsimplify(result)

    def scalar_hessian(self, expr, a, b):
        result = sp.S.Zero

        # Términos lineales en los Hessianos de los invariantes.
        for scalar in self.scalar_invariants:
            result += (
                sp.diff(expr, scalar)
                * self.scalar_hessians[scalar](-a, -b)
            )

        # Términos cuadráticos en los gradientes.
        for scalar_i in self.scalar_invariants:
            for scalar_j in self.scalar_invariants:
                result += (
                    sp.diff(expr, scalar_i, scalar_j)
                    * self.scalar_gradients[scalar_i](-a)
                    * self.scalar_gradients[scalar_j](-b)
                )

        return self.tsimplify(result)

    # ================================================================
    # 4. Construcciones geométricas auxiliares
    # ================================================================
    def Ricci_down(self, a, b):
        i, j = tensor_indices("Ric_i Ric_j", self.M)
        return self.tsimplify(
            self.g(i, j) * self.Riem(-i, -a, -j, -b)
        )

    def Ricci_up(self, a, b):
        p, q = tensor_indices("Ric_p Ric_q", self.M)
        return self.tsimplify(
            self.g(a, p) * self.g(b, q) * self.Ricci_down(p, q)
        )

    @staticmethod
    def _reindex4(expr, old, new):
        return expr.xreplace(dict(zip(old, new)))

    # ================================================================
    # 5. Plantillas de P^{abcd}
    # ================================================================
    def set_P_components(self, components):
        """components: iterable de (nombre, coeficiente, indices, plantilla)."""
        self.P_COMPONENTS = list(components)

    def P_up(self, a, b, c, d):
        if not self.P_COMPONENTS:
            raise RuntimeError("P^{abcd} todavía no ha sido construido.")

        result = sp.S.Zero
        for _, coefficient, old_indices, template in self.P_COMPONENTS:
            result += coefficient * self._reindex4(
                template, old_indices, (a, b, c, d)
            )
        return self.tsimplify(result)

    def P_component_up(self, name, a, b, c, d):
        for component_name, _, old_indices, template in self.P_COMPONENTS:
            if component_name == name:
                return self._reindex4(template, old_indices, (a, b, c, d))
        raise KeyError(f"No existe la componente de P llamada {name!r}.")

    def P_metric_up(self, a, b):
        if "P_metric_ab" not in self.S:
            raise RuntimeError("P^{ab} todavía no ha sido construido.")
        p, q = self.S["P_metric_template_indices"]
        return self.S["P_metric_ab"].xreplace({p: a, q: b})

    # ================================================================
    # 6. Derivadas divergentes de P
    # ================================================================
    def DP_up(self, e, a, b, c, d):
        """Divergencia de P en uno de sus índices.

        En las etapas variacionales siempre se usa con e identificado con uno
        de (a,b,c,d). Las dos plantillas, dR/dRiemann y dGB/dRiemann, son
        divergencia-cero; por eso solo se derivan los coeficientes escalares.
        """
        if e not in (a, b, c, d):
            raise ValueError(
                "DP_up está definido para divergencias: el índice derivativo "
                "debe coincidir con uno de los índices de P."
            )

        result = sp.S.Zero
        for _, coefficient, old_indices, template in self.P_COMPONENTS:
            result += self.scalar_covd(coefficient, e) * self._reindex4(
                template, old_indices, (a, b, c, d)
            )
        return self.tsimplify(result)

    def DDP_up(self, e, f, a, b, c, d):
        """Doble divergencia en las combinaciones usadas por las dos IBP."""
        if e not in (a, b, c, d) or f not in (a, b, c, d):
            raise ValueError(
                "DDP_up está definido para dobles divergencias: cada índice "
                "derivativo debe coincidir con un índice de P."
            )

        result = sp.S.Zero
        for _, coefficient, old_indices, template in self.P_COMPONENTS:
            result += self.scalar_hessian(coefficient, e, f) * self._reindex4(
                template, old_indices, (a, b, c, d)
            )
        return self.tsimplify(result)

    def minus2_double_divergence_P_down(self, a, b):
        """Calcula -2 nabla^m nabla^n P_{amnb} desde las componentes de P."""
        m, n, p, q, r, s, t, u = tensor_indices(
            "DD_m DD_n DD_p DD_q DD_r DD_s DD_t DD_u", self.M
        )

        result = sp.S.Zero
        for _, coefficient, old_indices, template in self.P_COMPONENTS:
            template_rstu = self._reindex4(
                template, old_indices, (r, s, t, u)
            )
            result += (
                -2
                * self.g(m, p) * self.g(n, q)
                * self.g(-a, -r) * self.g(-m, -s)
                * self.g(-n, -t) * self.g(-b, -u)
                * self.scalar_hessian(coefficient, p, q)
                * template_rstu
            )

        return self.tsimplify(result)

    def DGamma_from_h(self, e, c, d, b):
        i = tensor_indices("DG_i", self.M)
        return sp.Rational(1, 2) * self.g(e, i) * (
            self.DDh(-c, -d, -b, -i)
            + self.DDh(-c, -b, -d, -i)
            - self.DDh(-c, -i, -d, -b)
        )

    def expose_user_symbols(self):
        return (
            self.R,
            self.GB,
            self.alpha,
            self.beta,
            self.gamma,
            self.Lambda,
            self.mu,
        )

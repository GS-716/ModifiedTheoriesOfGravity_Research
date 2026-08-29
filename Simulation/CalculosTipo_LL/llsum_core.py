import sympy as sp
from IPython.display import display, Markdown
from sympy.tensor.tensor import (
    TensorIndexType, TensorHead, TensorSymmetry,
    tensor_indices, canon_bp, contract_metric, TensExpr,
)

sp.init_printing()


# ============================================================================
# Objetos compactos para la combinatoria de Lanczos--Lovelock
# ============================================================================

class LLDensity(sp.AtomicExpr):
    """Densidad pura L_k escrita con delta generalizada, ya normalizada."""

    def __new__(cls, k):
        return sp.AtomicExpr.__new__(cls, sp.Integer(k))

    @property
    def k(self):
        return int(self.args[0])

    def _latex(self, printer):
        k = self.k
        if k == 0:
            return r"\frac{1}{16\pi}"
        return (
            r"\frac{1}{16\pi}\frac{1}{2^{%d}}"
            r"\,\delta^{a_1b_1\cdots a_%db_%d}_{c_1d_1\cdots c_%dd_%d}"
            r"\prod_{r=1}^{%d}R^{c_rd_r}{}_{a_rb_r}"
        ) % (k, k, k, k, k, k)


class GaussBonnetExpanded(sp.AtomicExpr):
    """Invariante de Gauss--Bonnet escrito explícitamente."""

    def _latex(self, printer):
        return (
            r"R_{abcd}R^{abcd}-4R_{ab}R^{ab}+R^2"
        )


class LLFieldExpanded(sp.AtomicExpr):
    """Tensor de campo mixto total, sin abreviaturas L_k."""

    def __new__(cls, mmax, *couplings):
        return sp.AtomicExpr.__new__(cls, sp.Integer(mmax), *couplings)

    @property
    def mmax(self):
        return int(self.args[0])

    @property
    def couplings(self):
        return self.args[1:]

    def _latex(self, printer):
        m = self.mmax
        c = [printer.doprint(x) for x in self.couplings]
        pieces = []

        if m >= 0:
            pieces.append(rf"-\frac{{{c[0]}}}{{32\pi}}\delta^i_j")
        if m >= 1:
            pieces.append(
                rf"+\frac{{{c[1]}}}{{16\pi}}"
                r"\left(R^i{}_j-\frac12\delta^i_jR\right)"
            )
        if m >= 2:
            pieces.append(
                rf"+\frac{{{c[2]}}}{{16\pi}}\Big["
                r"2RR^i{}_j-4R^{ik}R_{jk}-4R^{kl}R^i{}_{kjl}"
                r"+2R^{iklm}R_{jklm}"
                r"-\frac12\delta^i_j\left(R_{abcd}R^{abcd}-4R_{ab}R^{ab}+R^2\right)"
                r"\Big]"
            )
        if m >= 3:
            pieces.append(
                r"-\frac{1}{32\pi}\sum_{k=3}^{%d}\frac{c_k}{2^k}"
                r"\delta^{i a_1b_1\cdots a_kb_k}_{j c_1d_1\cdots c_kd_k}"
                r"\prod_{r=1}^{k}R^{c_rd_r}{}_{a_rb_r}"
                % m
            )

        return r"E^i{}_j=" + "".join(pieces)


class LLCurvatureContribution(sp.AtomicExpr):
    def __new__(cls, k, slot):
        return sp.AtomicExpr.__new__(cls, sp.Integer(k), sp.Integer(slot))

    @property
    def k(self):
        return int(self.args[0])

    @property
    def slot(self):
        return int(self.args[1])

    def _latex(self, printer):
        return rf"\Pi^{{abcd}}_{{({self.k};{self.slot})}}"


class LLCurvatureBase(sp.AtomicExpr):
    def __new__(cls, k):
        return sp.AtomicExpr.__new__(cls, sp.Integer(k))

    @property
    def k(self):
        return int(self.args[0])

    def _latex(self, printer):
        return rf"\Pi^{{abcd}}_{{({self.k};1)}}"


class LLMetricContribution(sp.AtomicExpr):
    def __new__(cls, k, slot, leg):
        return sp.AtomicExpr.__new__(
            cls, sp.Integer(k), sp.Integer(slot), sp.Integer(leg)
        )

    @property
    def k(self):
        return int(self.args[0])

    @property
    def slot(self):
        return int(self.args[1])

    @property
    def leg(self):
        return int(self.args[2])

    def _latex(self, printer):
        label = "c" if self.leg == 1 else "d"
        return rf"\mathsf{{M}}^{{ab}}_{{({self.k};{self.slot},{label})}}"


class LLRcalContribution(sp.AtomicExpr):
    def __new__(cls, k, slot):
        return sp.AtomicExpr.__new__(cls, sp.Integer(k), sp.Integer(slot))

    @property
    def k(self):
        return int(self.args[0])

    @property
    def slot(self):
        return int(self.args[1])

    def _latex(self, printer):
        return rf"\mathfrak{{R}}^{{ab}}_{{({self.k};{self.slot})}}"


class LLRcalBase(sp.AtomicExpr):
    def __new__(cls, k):
        return sp.AtomicExpr.__new__(cls, sp.Integer(k))

    @property
    def k(self):
        return int(self.args[0])

    def _latex(self, printer):
        return rf"\mathfrak{{R}}^{{ab}}_{{({self.k};1)}}"


class LLBianchiContribution(sp.AtomicExpr):
    def __new__(cls, k, slot):
        return sp.AtomicExpr.__new__(cls, sp.Integer(k), sp.Integer(slot))

    @property
    def k(self):
        return int(self.args[0])

    @property
    def slot(self):
        return int(self.args[1])

    def _latex(self, printer):
        return rf"\mathsf{{B}}^{{bcd}}_{{({self.k};{self.slot})}}"


class LLBianchiRepresentative(sp.AtomicExpr):
    def __new__(cls, k):
        return sp.AtomicExpr.__new__(cls, sp.Integer(k))

    @property
    def k(self):
        return int(self.args[0])

    def _latex(self, printer):
        k = self.k
        if k <= 1:
            return "0"
        rest = "" if k == 2 else rf"\prod_{{s=3}}^{{{k}}}R^{{c_sd_s}}{{}}_{{a_sb_s}}"
        return (
            rf"\delta^{{ab\,a_2b_2\cdots a_{k}b_{k}}}_{{cd\,c_2d_2\cdots c_{k}d_{k}}}"
            rf"\,\nabla_{{[a}}R^{{c_2d_2}}{{}}_{{a_2b_2]}}\,{rest}"
        )


def ll_compact_normal_form(expr):
    """Canoniza las contribuciones por slots y aplica Bianchi."""
    replacements = {}

    for atom in expr.atoms(LLCurvatureContribution):
        replacements[atom] = LLCurvatureBase(atom.k)

    for atom in expr.atoms(LLRcalContribution):
        replacements[atom] = LLRcalBase(atom.k)

    for atom in expr.atoms(LLMetricContribution):
        replacements[atom] = -LLRcalBase(atom.k)

    for atom in expr.atoms(LLBianchiContribution):
        replacements[atom] = sp.S.Zero

    return sp.factor(sp.simplify(expr.xreplace(replacements)))


# ============================================================================
# Almacén de resultados
# ============================================================================

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
            raise AssertionError(f"La verificación {key} no dio cero: {result}")
        return result


# ============================================================================
# Contexto compartido para la suma completa hasta m_max
# ============================================================================

class LLSumContext:
    """Estado compartido para L = sum_{k=0}^{m_max} c_k L_k."""

    def __init__(self):
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

        self.sqrtg = sp.Symbol("sqrt_minus_g", positive=True)
        self.D = sp.Symbol("D", integer=True, positive=True)
        self.R = sp.Symbol("R", real=True)
        self.Lambda = sp.Symbol(r"\Lambda", real=True)
        self.alpha = sp.Symbol(r"\alpha", real=True)

        self.S = StepStore(self)

        self.max_order = None
        self.couplings = None
        self.L_terms = None
        self.L_total_expr = None
        self.P_tensor = None
        self.Rcal_tensor = None

        self.metric_identity_derived = False
        self.divergence_free_derived = False

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

        compact_classes = (
            LLCurvatureContribution, LLCurvatureBase,
            LLMetricContribution, LLRcalContribution, LLRcalBase,
            LLBianchiContribution,
        )
        if any(expr.has(cls) for cls in compact_classes):
            return ll_compact_normal_form(expr)

        return sp.factor(sp.simplify(expr))

    @staticmethod
    def normalization(k):
        return sp.Rational(1, 16) / (sp.pi * 2**k)

    def exact_term(self, k):
        if k == 0:
            return sp.Rational(1, 16) / sp.pi
        if k == 1:
            return self.R / (16*sp.pi)
        if k == 2:
            return GaussBonnetExpanded() / (16*sp.pi)
        return LLDensity(k)

    def configure_max_order(self, mmax):
        if isinstance(mmax, bool) or not isinstance(mmax, (int, sp.Integer)):
            raise TypeError("m debe ser un entero no negativo.")
        mmax = int(mmax)
        if mmax < 0:
            raise ValueError("m debe satisfacer m >= 0.")

        self.max_order = mmax
        self.couplings = sp.symbols(f"c_0:{mmax+1}", real=True)
        self.L_terms = {k: self.exact_term(k) for k in range(mmax+1)}
        self.L_total_expr = sp.Add(*[
            self.couplings[k] * self.L_terms[k]
            for k in range(mmax+1)
        ])

        self.P_tensor = TensorHead(
            rf"\mathrm{{P}}_{{(\leq {mmax})}}",
            [self.M]*4,
            TensorSymmetry.riemann(),
        )
        self.Rcal_tensor = TensorHead(
            rf"\mathcal{{R}}_{{(\leq {mmax})}}",
            [self.M]*2,
            TensorSymmetry.fully_symmetric(2),
        )

    def P_up(self, a, b, c, d):
        if self.max_order is None:
            raise RuntimeError("Primero configure m.")
        if self.max_order == 0:
            return sp.S.Zero
        return self.P_tensor(a, b, c, d)

    def Rcal_contraction_raw(self, a, b):
        if self.max_order == 0:
            return sp.S.Zero
        i, j, k = tensor_indices("Rc_i Rc_j Rc_k", self.M)
        return self.tsimplify(
            self.P_up(a, i, j, k) * self.Riem(b, -i, -j, -k)
        )

    def Rcal_up(self, a, b):
        if self.max_order == 0:
            return sp.S.Zero
        return self.Rcal_tensor(a, b)

    def Rcal_down(self, a, b):
        if self.max_order == 0:
            return sp.S.Zero
        p, q = tensor_indices("Rcd_p Rcd_q", self.M)
        return self.tsimplify(
            self.g(-a,-p)*self.g(-b,-q)*self.Rcal_up(p,q)
        )

    def P_metric_up(self, a, b):
        if not self.metric_identity_derived:
            raise RuntimeError("P^{ab} aún no fue derivado independientemente.")
        return self.tsimplify(-2*self.Rcal_up(a, b))

    def h_from_H(self, a, b):
        m, n = tensor_indices("hH_m hH_n", self.M)
        return -self.g(-a, -m) * self.g(-b, -n) * self.H(m, n)

    def DGamma_from_h(self, e, c, d, b):
        i = tensor_indices("DG_i", self.M)
        return sp.Rational(1, 2) * self.g(e, i) * (
            self.DDh(-c, -d, -b, -i)
            + self.DDh(-c, -b, -d, -i)
            - self.DDh(-c, -i, -d, -b)
        )

    def DP_up(self, e, a, b, c, d):
        if not self.divergence_free_derived:
            raise RuntimeError("La divergencia nula de P todavía no ha sido demostrada.")
        return sp.S.Zero

    def DDP_up(self, e, f, a, b, c, d):
        if not self.divergence_free_derived:
            raise RuntimeError("La divergencia nula de P todavía no ha sido demostrada.")
        return sp.S.Zero

    def explicit_field_mixed(self):
        return LLFieldExpanded(self.max_order, *self.couplings)

    def expose_user_symbols(self):
        return self.D, self.R, self.Lambda, self.alpha

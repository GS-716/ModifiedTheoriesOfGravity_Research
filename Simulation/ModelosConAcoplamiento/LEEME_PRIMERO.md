# ModelosConAcoplamiento

## 1. Qué hace este proyecto

Este proyecto reconstruye de forma simbólica y auditable la variación de un
lagrangiano general:

```text
L = L(g^{ab}, R_{abcd}, phi, nabla_a phi)
```

Después lo especializa a los Casos 0, 1 y 2 de `Papers/Draft_4.pdf` y, sobre la
misma base, permite declarar una suma finita arbitraria de invariantes EQT:

```text
Caso 0: L_0 = R + 2/ell²
        alpha_1 = beta_0 = 0

Caso 1: L_1 = R + 2/ell² - alpha_1 (nabla phi)²
        beta_0 = 0

Caso 2: L_2 = R + 2/ell²
              + ell² beta_0 [3 R_ab u^a u^b - R u²]
        u_a = nabla_a phi,  alpha_1 = 0,  beta_0 != 0
```

La meta no es obtener únicamente `f(r)`. Se conserva la historia completa:
momentos, variación, términos de frontera, ecuaciones de Euler-Lagrange,
Bianchi-Noether, geometría del ansatz, ecuaciones radiales, soluciones e
invariantes.

El notebook `derivacion_modelos_con_acoplamiento_MAIN.ipynb` es solo el **main
narrativo**. La física y los algoritmos viven en archivos `.py`, para que puedan
probarse y reutilizarse sin depender de Jupyter.

### Convención de escritura de esta guía

Todas las fórmulas se muestran como texto monoespaciado. No se necesita MathJax,
KaTeX ni ninguna extensión de LaTeX para leerlas desde GitHub.

Los índices elevados se escriben con `^` y los inferiores con `_`. Por ejemplo:

```text
u^a       índice elevado
u_a       índice inferior
R_{abcd}  cuatro índices inferiores
ell²      ell al cuadrado
f'        primera derivada respecto de r
f''       segunda derivada respecto de r
```

---

## 2. La historia completa en una vista

```text
L(g, R, phi, nabla phi)
        |
        v
Definir los cuatro momentos P, M, J y F
        |
        v
Variar: medida + Palatini + integraciones por partes
        |
        v
Construir E_ab, E_phi, frontera y Bianchi-Noether
        |
        v
Especializar los Casos 0, 1 y 2
        |
        v
Construir la geometría común g[f]
        |
        v
Imponer phi = p varphi cuando corresponde
        |
        v
Evaluar todo manteniendo f(r) simbólica
        |
        v
Resolver las ecuaciones radiales
        |
        v
Sustituir f, f', f'', ... en todos los objetos
        |
        v
Verificar ecuaciones, Bianchi e invariantes
        |
        v
Notebook -> LaTeX -> PDF + JSON de residuos
```

Hay dos niveles deliberadamente separados:

1. **Tensorial, sin ansatz:** `g_ab` y `phi` son arbitrarios. Aquí se obtienen
   los momentos y las ecuaciones covariantes de cada teoría.
2. **Coordenado, con ansatz:** se fija la métrica circular y el perfil escalar,
   se calculan componentes, se resuelve `f(r)` y se sustituye la solución.

Así no se confunde una identidad general con una simplificación accidental de
una métrica particular.

---

## 3. Ruta de lectura recomendada

1. `mc_core.py`: entender qué es `ctx` y cómo se registra un cálculo.
2. `mc_general.py`: seguir la derivación física del lagrangiano general.
3. `mc_geometry.py`: ver cómo la geometría diferencial se vuelve álgebra SymPy.
4. `mc_case0.py`: estudiar Einstein-AdS y la geometría común.
5. `mc_case1.py`: añadir el escalar cinético y obtener la rama logarítmica.
6. `mc_case2.py`: estudiar la doble divergencia y la rama racional.
7. `mc_pipeline.py`: comprobar el orden de integración.
8. `test_symbolic.py`: conocer las propiedades protegidas por pruebas.
9. `mc_export.py`: entender la producción de LaTeX, PDF y JSON.
10. Notebook main: recorrer la misma secuencia de forma interactiva.

---

## 4. Diccionario físico-computacional

| Objeto físico | Objeto en el código |
|---|---|
| `g_ab`, `g^ab` | `geo.g`, `geo.g_inv` |
| `R^a_{bcd}`, `R_abcd` | `geo.Riemann_up`, `geo.Riemann_down` |
| `R_ab`, `R`, `G_ab` | `geo.Ricci`, `geo.Rscalar`, `geo.Einstein` |
| `u_a = nabla_a phi` | `u_cov` |
| `u^a` | `u_up` |
| `X = u^a u_a` | `X` |
| `P^abcd` | diccionario con claves `(a,b,c,d)` |
| `M_ab`, `Rcal_ab`, `E_ab` | matrices SymPy |
| `J^a` | vector columna SymPy |
| `F_phi`, `E_phi` | expresiones SymPy |
| `f(r)` aún no resuelta | `sp.Function("f")(r)` |

`Rcal_ab` representa el Ricci generalizado, escrito como `mathcal R_ab` en el
informe LaTeX.

La teoría tensorial abstracta se registra como cadenas legibles. Los cálculos
coordenados y las comprobaciones se conservan como objetos SymPy exactos. Esta
separación evita atribuir a SymPy una derivada funcional tensorial que requeriría
convenciones adicionales.

---

## 5. El hilo conductor: `mc_core.py`

### `Step`

Cada `Step` es una igualdad o verificación exportable:

- `key`: identificador único, por ejemplo `case2_final_M`;
- `title`: título humano;
- `lhs` y `rhs`: lados de la igualdad;
- `group`: sección y subsección de la historia;
- `note`: explicación adicional;
- `check`: residuo simbólico opcional.

### `CouplingContext`

Una corrida completa se guarda en un único objeto `ctx`:

- `ctx.steps`: historia ordenada para notebook y PDF;
- `ctx.objects`: objetos SymPy reutilizables;
- `ctx.checks`: residuos exactos de las verificaciones.

Sus operaciones centrales son:

- `ctx.add(...)`: agrega un paso visible. Si recibe `check`, simplifica el
  residuo y detiene la ejecución si no es cero.
- `ctx.put(...)`: guarda un objeto para otra etapa.
- `ctx.show(...)`: presenta pasos seleccionados en Jupyter.
- `latex_expr(...)`: convierte expresiones y matrices SymPy a LaTeX.

El ciclo normal de un resultado es:

```python
# 1. Construir el objeto simbólico.
E = (...).applyfunc(sp.simplify)

# 2. Guardarlo para cálculos posteriores.
ctx.put("E_case2_ansatz", E)

# 3. Agregarlo a la historia visible.
ctx.add("case2_ansatz_E", "Tensor métrico", lhs, latex_expr(E), group)

# 4. Si es una identidad, registrar también su residuo.
ctx.add(..., check=residuo)
```

El exportador no recalcula la física. Recorre una historia que ya fue construida
y, en los pasos con `check`, verificada.

---

## 6. Primer acto: `mc_general.py`

`build_general_theory(ctx)` crea el esqueleto covariante de los tres casos.

### Acción y momentos

```text
S = kappa integral[d^D x sqrt(-g) L(g^ab, R_abcd, phi, u_a)]
u_a = nabla_a phi
```

Se definen cuatro momentos distintos:

```text
P^abcd = partial L / partial R_abcd    momento de curvatura
M_ab   = partial L / partial g^ab      momento métrico
J^a    = partial L / partial u_a       momento del gradiente escalar
F_phi  = partial L / partial phi       dependencia escalar explícita
```

`P^abcd` hereda las simetrías de Riemann. Mantener nombres distintos para `P`
y `M` permite ver qué derivada parcial produce cada término.

### Variación

```text
delta L = M_ab delta g^ab
        + P^abcd delta R_abcd
        + F_phi delta phi
        + J^a nabla_a delta phi
```

Después se incorporan:

- la variación de `sqrt(-g)`;
- la identidad de Palatini;
- la variación de la conexión Levi-Civita;
- dos integraciones por partes en el sector de curvatura;
- una integración por partes en el sector escalar.

El Ricci generalizado se define como:

```text
Rcal_ab = P_a^{cde} R_bcde
```

La covariancia produce la identidad algebraica de momentos:

```text
M_ab = 2 Rcal_(ab) + (1/2) J_(a u_b)
```

Las ecuaciones de campo generales quedan:

```text
E_ab = Rcal_(ab) - (1/2) g_ab L
       - 2 nabla^m nabla^n P_(a|mn|b)
       + (1/2) J_(a u_b)

E_phi = F_phi - nabla_a J^a
```

El cierre estructural es Bianchi-Noether off-shell:

```text
2 nabla^a E_ab + E_phi u_b = 0
```

“Off-shell” significa que todavía no se impusieron `E_ab = 0` ni `E_phi = 0`.

---

## 7. Segundo acto: `mc_geometry.py`

`CoordinateGeometry(coordinates, metric)` convierte una métrica en objetos
coordenados. El constructor calcula, en orden:

1. métrica inversa;
2. símbolos de Christoffel;
3. Riemann con un índice elevado;
4. Riemann completamente covariante;
5. Ricci;
6. escalar de Ricci;
7. Einstein.

La convención de Riemann está implementada en `_riemann_up`. Un cambio de signo
allí se propagaría a todos los momentos, ecuaciones e invariantes.

Métodos principales:

- `_christoffel`, `_riemann_up`, `_riemann_down`, `_ricci`, `_einstein`:
  implementan las definiciones mediante sumas de índices.
- `divergence_cov2(T)`: calcula `nabla^a T_ab` para Bianchi.
- `scalar_gradient_cov(phi)`: construye `u_a`.
- `scalar_laplacian(phi)`: calcula `Box phi`.
- `nonzero_christoffel`, `independent_riemann` e
  `independent_einstein_hilbert_momentum`: eliminan ceros y redundancias de la
  presentación, sin modificar el tensor.
- `kretschmann`: implementa la contracción completa del Riemann.

Los tres casos comparten la métrica:

```text
ds² = -f(r) d tau² + dr²/f(r) + r² d varphi²
```

Se construye una vez en el Caso 0 y se guarda como
`ctx.objects["geometry"]`. Los Casos 1 y 2 reutilizan exactamente la misma
geometría y convención.

---

## 8. Tercer acto: `mc_case0.py`

### `build_case0(ctx)`

Registra Einstein-AdS antes de elegir coordenadas:

```text
P_0^abcd = (1/2) (g^ac g^bd - g^ad g^bc)
M_ab^(0) = 2 R_ab
J_0^a = 0
F_phi^(0) = 0
Rcal_ab^(0) = R_ab
```

Como `nabla P_0 = 0`, la doble divergencia desaparece:

```text
E_ab^(0) = G_ab - g_ab/ell²
```

También se registran Bianchi y el término Gibbons-Hawking-York. Este caso es el
control básico del motor.

### `evaluate_btz_ansatz(ctx)`

Introduce `tau`, `r`, `varphi`, `ell`, `lambda` y `f(r)`. Crea
`CoordinateGeometry` y guarda estos objetos en `ctx`.

Con `f(r)` arbitraria calcula momentos, Christoffel, Riemann, Ricci, `R`,
Einstein y `E_ab^(0)[f]`. Las componentes independientes dan:

```text
f'(r)  = 2r/ell²
f''(r) = 2/ell²

f_(0)(r) = r²/ell² - lambda
```

`subs_btz` sustituye simultáneamente `f`, `f'` y `f''`. Sustituir solo `f`
dejaría derivadas simbólicas antiguas dentro de los tensores.

Después se reconstruyen los momentos finales y se verifica:

```text
R_ab = -2 g_ab/ell²
R    = -6/ell²
G_ab = g_ab/ell²
E_ab^(0) = 0
R_abcd R^abcd = 12/ell⁴
```

También se comprueban curvatura constante y Bianchi.

---

## 9. Cuarto acto: `mc_case1.py`

### `build_case1(ctx)`

Para `L_1 = R + 2/ell² - alpha_1 X` obtiene:

```text
M_ab^(1) = 2 R_ab - alpha_1 u_a u_b
J_1^a    = -2 alpha_1 u^a
F_phi^(1) = 0
```

`F_phi = 0` expresa simetría de desplazamiento. El módulo forma el tensor
cinético, `E_ab^(1)`, `E_phi^(1) = 2 alpha_1 Box(phi)`, Bianchi y frontera.

### `evaluate_case1_ansatz(ctx)`

Recupera la geometría del Caso 0 e impone:

```text
phi = p varphi
```

Calcula `u_a`, `u^a`, `X`, el tensor cinético, `Box(phi)`, todos los momentos y
`E_ab^(1)[f]`. El perfil angular satisface `Box(phi) = 0` para cualquier `f(r)`
de este ansatz.

La ecuación radial es:

```text
f'(r) = 2r/ell² - alpha_1 p²/r
```

El código usa `sp.integrate` y verifica la primitiva antes de construir:

```text
f_(1)(r) = r²/ell² - lambda - alpha_1 p² log(r/r_0)
```

Luego sustituye `f`, `f'` y `f''` en momentos, curvatura y ecuaciones. Comprueba
la ecuación métrica, la escalar, Bianchi-Noether, invariantes y el flujo normal
al borde. También guarda `p` en `ctx`; el Caso 2 reutiliza el mismo símbolo.

---

## 10. Quinto acto: `mc_case2.py`

Es el módulo más técnico porque `P_2^abcd` no es covariantemente constante y su
doble divergencia contribuye realmente a la ecuación métrica.

### Funciones auxiliares

- `_curvature_momentum`: construye `P^abcd` para un sector `C^ab R_ab`.
- `_independent_rank4`: selecciona componentes independientes no nulas.
- `_lower_rank4`: baja los cuatro índices de `P`.
- `_generalized_ricci`: contrae `P_a^{cde} R_bcde`.
- `_double_divergence`: calcula dos derivadas covariantes sucesivas, con todas
  las correcciones de Christoffel.
- `_vector_divergence`: calcula `nabla_a J^a`.
- `_momentum_latex` y `_diagonal_tensor_latex`: hacen legible la salida.

`_double_divergence` debe leerse lentamente: baja índices, simetriza en `a,b`,
calcula una derivada covariante de rango cuatro, contrae y calcula la segunda
derivada covariante. No usa una simple derivada parcial.

### `build_case2(ctx)`

Organiza el acoplamiento mediante:

```text
H^ab = 3 u^a u^b - X g^ab
C^ab = g^ab + ell² beta_0 H^ab
```

Así el sector de curvatura es `C^ab R_ab`, y se obtiene:

```text
P_2^abcd = (1/4) [C^ac g^bd - C^ad g^bc
                   - C^bc g^ad + C^bd g^ac]
```

También registra `M_2`, `J_2`, `F_2 = 0`, `Rcal_2`, la divergencia de `P_2`,
las ecuaciones, Bianchi-Noether y la acción de frontera mejorada.

### `evaluate_case2_ansatz(ctx)`

Recupera `geometry`, coordenadas, `ell`, `lambda` y `f` del Caso 0, y `p` del
Caso 1. Con `phi = p varphi` calcula, en orden:

1. `u_a`, `u^a`, `X`, `R^ab` y `R_ab u^a u^b`;
2. `H^ab`, `C^ab` y el tensor completo `P^abcd`;
3. `Rcal_ab`, `J^a`, `F_phi` y `M_ab`;
4. la identidad algebraica entre los momentos;
5. la doble divergencia de `P`;
6. el lagrangiano, `E_phi` y `E_ab`.

En esta primera pasada `f(r)` sigue siendo arbitraria. Por eso el informe puede
mostrar todos los momentos después del ansatz pero antes de resolver la métrica.

### Reducción radial

```text
H(r) = 1 + beta_0 p² ell²/r²
N(r) = H(r) f(r)

N'(r)  = 2r/ell²
N''(r) = 2/ell²
```

La solución racional es:

```text
                    r²/ell² - lambda
f_(2)(r) = -----------------------------------
            1 + beta_0 p² ell²/r²
```

El código verifica exactamente:

```text
H(r) f_(2)(r) - [r²/ell² - lambda] = 0
```

### Momentos completamente explícitos

La doble divergencia puede generar hasta `f''''`. Por eso el diccionario final
incluye `f`, `f'`, `f''`, `f'''` y `f''''`.

Después se reconstruyen:

- `P_case2_final`;
- `M_case2_final`;
- `J_case2_final`;
- `F_case2_final`;
- `Rcal_case2_final`.

Ninguno conserva la función abstracta `f(r)`. Finalmente se verifican las
ecuaciones métricas y escalares, Bianchi, Bianchi-Noether, Ricci, `R`,
`R_ab R^ab`, Kretschmann y el flujo escalar en el borde.

---

## 11. Por qué el orden importa: `mc_pipeline.py`

`run_pipeline()` ejecuta:

```python
build_general_theory(ctx)
build_case0(ctx)
build_case1(ctx)
build_case2(ctx)
build_eqt_general(ctx, eqt_spec)
evaluate_btz_ansatz(ctx)
evaluate_case1_ansatz(ctx)
evaluate_case2_ansatz(ctx)
evaluate_eqt_general_ansatz(ctx, eqt_spec)
```

Los cinco primeros llamados registran teoría abstracta y reglas. Los cuatro
últimos hacen la evaluación coordenada. Las dependencias reales son:

```text
evaluate_btz_ansatz
  crea geometry, coordinates, ell, lambda y f
              |
              v
evaluate_case1_ansatz
  reutiliza geometry y crea p
              |
              v
evaluate_case2_ansatz
  reutiliza geometry, f, lambda, ell y p
              |
              v
evaluate_eqt_general_ansatz
  compone el modelo indicado por eqt_spec
```

El orden es parte de la arquitectura. Funciona como inyección explícita de
dependencias mediante `ctx`, sin variables globales ocultas.

---

## 12. El notebook como main

El notebook repite el pipeline por etapas para mostrar cada acto:

1. importa módulos y crea `ctx`;
2. construye y muestra la teoría general;
3. construye los Casos I sin ansatz;
4. registra las reglas del lagrangiano EQT configurable;
5. evalúa los Casos II y el modelo general con el ansatz;
6. muestra soluciones y residuos desde `ctx.objects`;
7. exige que todos los `ctx.checks` sean cero;
8. llama `export_results(ctx, compile_pdf=True)`.

Las listas `*_keys` filtran `ctx.steps` por grupo. El notebook no vuelve a hacer
la física: decide qué fragmento de la historia mostrar en cada sección.

---

## 13. Exportación: `mc_export.py`

`build_latex(ctx)` recorre los pasos en orden. Un grupo con formato
`Sección::Subsección` se convierte automáticamente en estructura LaTeX. Para
cada paso imprime título, clave, igualdad y nota.

`_equation` ajusta fórmulas extensas. Las matrices largas del Caso 2 llegan ya
separadas por componentes. Luego se agrega una tabla de residuos: `OK` significa
cero simbólico exacto, no una comprobación numérica en puntos elegidos.

`export_results` genera:

- `salidas/derivacion_modelos_con_acoplamiento_casos0_2.tex`;
- `salidas/derivacion_modelos_con_acoplamiento_casos0_2.pdf`;
- `salidas/verificaciones_simbolicas.json`.

Intenta usar `latexmk` si también existe Perl. Como respaldo ejecuta `pdflatex`
dos veces para resolver correctamente el índice y las referencias.

---

## 14. Pruebas: `test_symbolic.py`

La prueba reconstruye todo mediante `run_pipeline()` y exige:

- un mínimo de pasos y que todos los residuos sean cero;
- ecuaciones métricas finales nulas en los tres casos y el modelo compuesto;
- ecuaciones escalares correctas;
- `R_BTZ = -6/ell²` y la forma esperada de `J_1^varphi`;
- ausencia de `f(r)` en los momentos finales de los tres casos;
- `H(r) f_(2)(r) = r²/ell² - lambda`;
- reducción general `E_rr = [1/(2rf)] d[Hf-N]/dr`;
- ausencia de `f(r)` en los momentos finales EQT.

La prueba de ausencia de `f` distingue entre una fórmula parcialmente sustituida
y haber usado realmente la solución explícita en todos los momentos.

---

## 15. Cómo rastrear un resultado concreto

Ejemplo: momento métrico final del Caso 2.

1. `mc_general.py` define `M_ab = partial L / partial g^ab`.
2. `build_case2` registra su forma tensorial.
3. `evaluate_case2_ansatz` construye `M` con la geometría y `u_a`.
4. `M_case2_ansatz` lo conserva con `f(r)` arbitraria.
5. Se obtiene `f_case2_solution` mediante `N = H f`.
6. `M_final` sustituye la solución y todas sus derivadas.
7. `M_case2_final` conserva el objeto explícito.
8. `case2_final_M` lo incorpora a la narración por componentes.
9. `test_symbolic.py` verifica que ya no contiene `f`.
10. El notebook lo muestra y el exportador lo escribe en el PDF.

El patrón se repite para `P`, `J`, `F`, `Rcal`, ecuaciones e invariantes.

---

## 16. Cómo añadir un caso nuevo

1. Crear `build_caseN(ctx)` para el nivel tensorial.
2. Registrar truncamiento, lagrangiano, acción y cuatro momentos.
3. Construir `Rcal`, ecuaciones, Bianchi y frontera.
4. Crear `evaluate_caseN_ansatz(ctx)`.
5. Evaluar todos los momentos manteniendo `f(r)` simbólica.
6. Resolver las ecuaciones radiales.
7. Sustituir `f` y todas las derivadas necesarias.
8. Registrar momentos finales y checks exactos.
9. Integrar ambas funciones en pipeline y notebook.
10. Ampliar pruebas y resumen del exportador.

Debe conservarse la frontera `build_*` / `evaluate_*`: separa la teoría
covariante de la solución particular.

---

## 17. Cómo configurar un lagrangiano EQT arbitrario

La generalización nueva no reemplaza los Casos 0–2: los utiliza como límites de
regresión. La entrada del usuario es un `EQTModelSpec`, definido en
`mc_invariants.py`.

La configuración usada por defecto es:

```python
eqt_spec = symbolic_eqt_spec(
    alpha_orders=(1, 2),
    beta_orders=(1,),
)
```

Esto activa `alpha_1`, `alpha_2` y `beta_1` como coeficientes simbólicos. Para
otra suma finita basta cambiar las tuplas. Por ejemplo:

```python
# Einstein-AdS + alpha_1 + alpha_3 + beta_0 + beta_2
eqt_spec = symbolic_eqt_spec(
    alpha_orders=(1, 3),
    beta_orders=(0, 2),
)
```

También pueden suministrarse coeficientes que sean expresiones SymPy:

```python
from mc_invariants import EQTModelSpec

eqt_spec = EQTModelSpec(
    name="modelo personalizado",
    alpha={1: a, 2: 3*b},
    beta={0: c},
)
```

Las densidades soportadas son:

```text
Torre alpha:
  A_n = -alpha_n ell^[2(n-1)] X^n,       n >= 1

Torre beta:
  B_m = beta_m ell^[2(m+1)] X^m
        [(3+2m) R_ab u^a u^b - X R],    m >= 0
```

`mc_invariants.py` contiene las reglas locales de cada densidad: lagrangiano,
coeficiente de Ricci y corriente escalar. `mc_tensor.py` contiene las operaciones
independientes del modelo: momento de curvatura, contracciones, Ricci
generalizado, doble divergencia y simetrización. Finalmente, `mc_eqt.py` compone
las contribuciones y ejecuta:

```text
especificación
 -> densidad total
 -> P, J, F y reconstrucción de M
 -> E_ab y E_phi con f(r) arbitraria
 -> E_rr = [1/(2 r f)] d[H f - N]/dr
 -> f_EQT = N/H
 -> sustitución hasta f''''
 -> momentos finales, curvatura, Bianchi y checks
```

La rama analítica general implementada es:

```text
N(r) = r²/ell² - lambda - alpha_1 p² log(r/r_0)
       + SUM[n>=2] alpha_n ell^[2(n-1)] p^(2n)
                     / [2(n-1) r^[2(n-1)]]

H(r) = 1 + SUM[m>=0] beta_m (2m+1)
                     (p ell/r)^[2(m+1)]

f_EQT(r) = N(r)/H(r)
```

Las sumas son finitas: solo incluyen los órdenes presentes en `eqt_spec`. Al
activar muchos órdenes simultáneamente, las expresiones finales pueden crecer
con rapidez; el motor continúa siendo exacto, pero la simplificación simbólica
requiere más tiempo.

---

## 18. Ejecución reproducible en Windows/PowerShell

```powershell
.\.venv\Scripts\python.exe test_symbolic.py
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute derivacion_modelos_con_acoplamiento_MAIN.ipynb --output derivacion_modelos_con_acoplamiento_MAIN.ipynb --ExecutePreprocessor.cwd=. --ExecutePreprocessor.timeout=300
```

La primera orden reconstruye y prueba la cadena. La segunda ejecuta el notebook
y genera el informe. `requirements.txt` fija las dependencias Python; el PDF
requiere una instalación LaTeX con `latexmk` o `pdflatex`.

---

## 19. Mapa de archivos

```text
mc_core.py          Estado, pasos, objetos, checks y presentación
mc_general.py       Variación covariante de L(g,R,phi,nabla phi)
mc_geometry.py      Motor tensorial coordenado
mc_case0.py         Einstein-AdS y solución BTZ
mc_case1.py         Escalar mínimo y rama logarítmica
mc_case2.py         Acoplamiento con curvatura y rama racional
mc_invariants.py    Especificación EQT y reglas de cada invariante
mc_tensor.py        Operaciones tensoriales comunes
mc_eqt.py           Composición, reducción radial y rama general
mc_pipeline.py      Orden reproducible de integración
mc_export.py        LaTeX, PDF y JSON
test_symbolic.py    Pruebas exactas de regresión
*.ipynb             Main interactivo y narrativo
requirements.txt    Dependencias Python
salidas/            Artefactos finales
LEEME_PRIMERO.md    Este roadmap
```

## Idea final

```text
principio variacional
 -> identidades tensoriales
  -> objetos SymPy coordenados
   -> ecuaciones radiales
    -> soluciones explícitas
     -> residuos exactos
      -> narración LaTeX/PDF
```

La física determina qué objetos deben existir. La lógica computacional los
convierte en expresiones manipulables y verificables. `CouplingContext` conserva
la historia, el pipeline garantiza el orden, el notebook la revela al lector y
el exportador la transforma en la respuesta documental final.

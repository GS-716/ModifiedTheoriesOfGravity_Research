# ModelosConAcoplamiento

## 1. Qué hace este proyecto

Este proyecto reconstruye de forma simbólica y auditable la variación de

\[
L=L(g^{ab},R_{abcd},\phi,\nabla_a\phi),
\]

y luego la especializa a los Casos 0, 1 y 2 de `Papers/Draft_4.pdf`:

- **Caso 0:** \(L_0=R+2/\ell^2\), con \(\alpha_1=\beta_0=0\).
- **Caso 1:** \(L_1=R+2/\ell^2-\alpha_1(\nabla\phi)^2\), con \(\beta_0=0\).
- **Caso 2:** \(L_2=R+2/\ell^2+\ell^2\beta_0[3R_{ab}u^au^b-Ru^2]\),
  con \(u_a=\nabla_a\phi\), \(\alpha_1=0\) y \(\beta_0\neq0\).

La meta no es imprimir únicamente \(f(r)\). Se conserva la historia completa:
momentos, variación, frontera, ecuaciones de Euler-Lagrange, Bianchi-Noether,
geometría del ansatz, ecuaciones radiales, soluciones e invariantes.

El notebook `derivacion_modelos_con_acoplamiento_MAIN.ipynb` es solo el **main
narrativo**. La física y los algoritmos viven en archivos `.py`, para que puedan
probarse y reutilizarse sin depender de Jupyter.

---

## 2. La historia completa en una vista

```text
L(g, R, phi, nabla phi)
        |
        v
Definir P, M, J y F
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

1. **Tensorial, sin ansatz:** \(g_{ab}\) y \(\phi\) son arbitrarios. Aquí se
   obtienen los momentos y las ecuaciones covariantes de cada teoría.
2. **Coordenado, con ansatz:** se fija la métrica circular, el perfil escalar, se
   calculan componentes, se resuelve \(f(r)\) y se sustituye la solución.

Así no se confunde una identidad general con una simplificación accidental de
una métrica particular.

---

## 3. Ruta de lectura recomendada

1. `mc_core.py`: qué es `ctx` y cómo se registra un cálculo.
2. `mc_general.py`: derivación física para el lagrangiano general.
3. `mc_geometry.py`: traducción de geometría diferencial a álgebra SymPy.
4. `mc_case0.py`: línea base Einstein-AdS y construcción de la geometría común.
5. `mc_case1.py`: escalar cinético y rama logarítmica.
6. `mc_case2.py`: acoplamiento con curvatura, doble divergencia y rama racional.
7. `mc_pipeline.py`: orden exacto de integración.
8. `test_symbolic.py`: propiedades que el proyecto protege.
9. `mc_export.py`: transformación de la historia en LaTeX, PDF y JSON.
10. Notebook main: la misma secuencia presentada para lectura humana.

---

## 4. Diccionario físico-computacional

| Física | Código |
|---|---|
| \(g_{ab}\), \(g^{ab}\) | `geo.g`, `geo.g_inv` |
| \(R^a{}_{bcd}\), \(R_{abcd}\) | `geo.Riemann_up`, `geo.Riemann_down` |
| \(R_{ab}\), \(R\), \(G_{ab}\) | `geo.Ricci`, `geo.Rscalar`, `geo.Einstein` |
| \(u_a=\nabla_a\phi\), \(u^a\), \(X=u^au_a\) | `u_cov`, `u_up`, `X` |
| \(P^{abcd}\) | diccionario con claves `(a,b,c,d)` |
| \(M_{ab}\), \(\mathcal R_{ab}\), \(E_{ab}\) | matrices SymPy |
| \(J^a\) | vector columna SymPy |
| \(F_\phi\), \(E_\phi\) | expresiones SymPy |
| \(f(r)\) aún no resuelta | `sp.Function("f")(r)` |

La teoría tensorial abstracta se registra como LaTeX legible. Los cálculos
coordenados y las comprobaciones se conservan como objetos SymPy exactos. Esta
división evita atribuir a SymPy una derivada funcional tensorial que requeriría
convenciones adicionales.

---

## 5. El hilo conductor: `mc_core.py`

### `Step`

Cada `Step` es una igualdad o verificación exportable:

- `key`: identificador único, por ejemplo `case2_final_M`;
- `title`: título humano;
- `lhs`, `rhs`: igualdad escrita en LaTeX;
- `group`: sección y subsección de la historia;
- `note`: explicación adicional;
- `check`: residuo simbólico opcional.

### `CouplingContext`

Una corrida se guarda en un único `ctx` con tres almacenes:

- `ctx.steps`: historia ordenada para notebook y PDF;
- `ctx.objects`: objetos SymPy reutilizables por etapas posteriores;
- `ctx.checks`: residuos exactos de las verificaciones.

Las operaciones centrales son:

- `ctx.add(...)`: incorpora un paso visible. Si recibe `check`, simplifica el
  residuo y detiene la ejecución si no es cero.
- `ctx.put(...)`: conserva un objeto para otros módulos.
- `ctx.show(...)`: presenta pasos seleccionados en Jupyter.
- `latex_expr(...)`: convierte expresiones y matrices SymPy a LaTeX.

El ciclo normal es calcular un objeto, guardarlo con `put`, registrarlo con
`add` y, si representa una identidad, adjuntar un residuo. El exportador no
recalcula la física: recorre esa historia ya construida y verificada.

---

## 6. Primer acto: `mc_general.py`

`build_general_theory(ctx)` construye el esqueleto covariante heredado por los
tres casos. Parte de

\[
S=\kappa\int d^Dx\sqrt{-g}\,L(g^{ab},R_{abcd},\phi,u_a),
\qquad u_a=\nabla_a\phi.
\]

### Los cuatro momentos

\[
P^{abcd}=\frac{\partial L}{\partial R_{abcd}},\quad
M_{ab}=\frac{\partial L}{\partial g^{ab}},\quad
J^a=\frac{\partial L}{\partial u_a},\quad
F_\phi=\frac{\partial L}{\partial\phi}.
\]

Se mantienen nombres distintos para el momento de curvatura y el métrico. El
código registra también las simetrías de Riemann heredadas por \(P^{abcd}\).

### Variación

La regla de la cadena es

\[
\delta L=M_{ab}\delta g^{ab}+P^{abcd}\delta R_{abcd}
+F_\phi\delta\phi+J^a\nabla_a\delta\phi.
\]

Después se añaden la variación de \(\sqrt{-g}\), Palatini, la variación de la
conexión Levi-Civita, dos integraciones por partes en el sector de curvatura y
una en el sector escalar. Así se separan bulk y frontera.

Se define

\[
\mathcal R_{ab}=P_a{}^{cde}R_{bcde},
\]

y la covariancia da la identidad de momentos

\[
M_{ab}=2\mathcal R_{(ab)}+\frac12J_{(a}u_{b)}.
\]

La ecuación métrica reducida y la escalar quedan

\[
E_{ab}=\mathcal R_{(ab)}-\frac12g_{ab}L
-2\nabla^m\nabla^nP_{(a|mn|b)}+\frac12J_{(a}u_{b)},
\]

\[
E_\phi=F_\phi-\nabla_aJ^a.
\]

El cierre es la identidad off-shell

\[
2\nabla^aE_{ab}+E_\phi u_b\equiv0.
\]

“Off-shell” significa que todavía no se imponen las ecuaciones de campo. Esta
identidad es luego una prueba estructural de las especializaciones.

---

## 7. Segundo acto: `mc_geometry.py`

`CoordinateGeometry(coordinates, metric)` convierte una métrica en objetos
coordenados. El constructor calcula, en orden:

1. métrica inversa;
2. Christoffel;
3. Riemann con un índice elevado;
4. Riemann covariante;
5. Ricci;
6. escalar de Ricci;
7. Einstein.

La convención de Riemann está escrita en `_riemann_up`; un cambio de signo allí
se propagaría por todo el proyecto.

Métodos relevantes:

- `_christoffel`, `_riemann_up`, `_riemann_down`, `_ricci`, `_einstein`:
  implementan las definiciones por sumas de índices.
- `divergence_cov2(T)`: calcula \(\nabla^aT_{ab}\) para Bianchi.
- `scalar_gradient_cov(phi)`: construye \(u_a\).
- `scalar_laplacian(phi)`: calcula \(\Box\phi\).
- `nonzero_christoffel`, `independent_riemann` e
  `independent_einstein_hilbert_momentum`: eliminan ceros y redundancias de la
  salida, sin alterar el tensor.
- `kretschmann`: implementa la contracción completa del Riemann.

La geometría común es

\[
ds^2=-f(r)d\tau^2+\frac{dr^2}{f(r)}+r^2d\varphi^2.
\]

Se construye una vez en el Caso 0 y se guarda en `ctx.objects["geometry"]`.
Los Casos 1 y 2 la reutilizan, garantizando idénticas coordenadas y convención.

---

## 8. Tercer acto: `mc_case0.py`

### `build_case0(ctx)`

Mantiene todo tensorial y registra Einstein-AdS:

\[
P_0^{abcd}=\frac12(g^{ac}g^{bd}-g^{ad}g^{bc}),\qquad
M^{(0)}_{ab}=2R_{ab},\qquad J_0^a=F_\phi^{(0)}=0.
\]

Como \(\nabla P_0=0\), la doble divergencia desaparece y
\(E^{(0)}_{ab}=G_{ab}-\ell^{-2}g_{ab}\). También se registran Bianchi y el
término Gibbons-Hawking-York. Es el control básico del motor.

### `evaluate_btz_ansatz(ctx)`

Introduce coordenadas, `ell`, `lambda` y `f(r)`, crea `CoordinateGeometry` y
guarda todo en `ctx`. Con \(f\) arbitraria calcula momentos, Christoffel,
Riemann, Ricci, \(R\), Einstein y \(E^{(0)}_{ab}[f]\).

Las componentes dan

\[
f'=2r/\ell^2,\qquad f''=2/\ell^2,
\]

y por tanto \(f_{(0)}=r^2/\ell^2-\lambda\). `subs_btz` sustituye a la vez `f`,
`f'` y `f''`; sustituir solo `f` dejaría derivadas simbólicas antiguas dentro de
los tensores.

Después reconstruye todos los momentos finales y verifica Ricci, \(R\),
Einstein, ecuaciones de campo, curvatura constante, Kretschmann y Bianchi.

---

## 9. Cuarto acto: `mc_case1.py`

### `build_case1(ctx)`

Para \(L_1=R+2/\ell^2-\alpha_1X\) obtiene

\[
M^{(1)}_{ab}=2R_{ab}-\alpha_1u_au_b,qquad
J_1^a=-2\alpha_1u^a,qquad F_\phi^{(1)}=0.
\]

`F=0` expresa simetría de desplazamiento. Se forman el tensor cinético,
\(E^{(1)}_{ab}\), \(E_\phi^{(1)}=2\alpha_1\Box\phi\), Bianchi y frontera.

### `evaluate_case1_ansatz(ctx)`

Recupera la geometría del Caso 0 e impone \(\phi=p\varphi\). Calcula \(u_a\),
\(u^a\), \(X\), el tensor cinético, \(\Box\phi\), todos los momentos y
\(E^{(1)}_{ab}[f]\). El perfil angular satisface \(\Box\phi=0\) para cualquier
\(f(r)\) de este ansatz.

La ecuación radial es

\[
f'=\frac{2r}{\ell^2}-\frac{\alpha_1p^2}{r}.
\]

El código usa `sp.integrate` y verifica la primitiva antes de construir

\[
f_{(1)}=\frac{r^2}{\ell^2}-\lambda
-\alpha_1p^2\log(r/r_0).
\]

Luego sustituye `f`, `f'` y `f''` en momentos, curvatura y ecuaciones. Comprueba
ecuación métrica, escalar, Bianchi-Noether, invariantes y flujo normal al borde.
Además guarda `p` en `ctx`; el Caso 2 reutiliza el mismo símbolo.

---

## 10. Quinto acto: `mc_case2.py`

Es el módulo más técnico porque \(P_2^{abcd}\) no es covariantemente constante
y su doble divergencia contribuye a la ecuación métrica.

### Auxiliares

- `_curvature_momentum`: construye \(P^{abcd}\) para un sector \(C^{ab}R_{ab}\).
- `_independent_rank4`: selecciona componentes independientes no nulas.
- `_lower_rank4`: baja los cuatro índices de \(P\).
- `_generalized_ricci`: contrae \(P_a{}^{cde}R_{bcde}\).
- `_double_divergence`: calcula dos derivadas covariantes sucesivas, incluyendo
  todas las correcciones de Christoffel.
- `_vector_divergence`: calcula \(\nabla_aJ^a\).
- `_momentum_latex` y `_diagonal_tensor_latex`: hacen legible el resultado.

### `build_case2(ctx)`

Define

\[
H^{ab}=3u^au^b-Xg^{ab},\qquad
C^{ab}=g^{ab}+\ell^2\beta_0H^{ab},
\]

para escribir el sector de curvatura como \(C^{ab}R_{ab}\). De allí obtiene

\[
P_2^{abcd}=\frac14(C^{ac}g^{bd}-C^{ad}g^{bc}-C^{bc}g^{ad}+C^{bd}g^{ac}).
\]

También registra \(M_2\), \(J_2\), \(F_2=0\), \(\mathcal R_2\), la divergencia
de \(P_2\), ecuaciones, Bianchi-Noether y la acción de frontera mejorada.

### `evaluate_case2_ansatz(ctx)`

Recupera geometría, coordenadas, `ell`, `lambda` y `f` del Caso 0, y `p` del
Caso 1. Con \(\phi=p\varphi\) calcula, en orden:

1. \(u_a\), \(u^a\), \(X\), \(R^{ab}\) y \(R_{ab}u^au^b\);
2. \(H^{ab}\), \(C^{ab}\) y el tensor completo \(P^{abcd}\);
3. \(\mathcal R_{ab}\), \(J^a\), \(F_\phi\) y \(M_{ab}\);
4. la identidad algebraica entre los momentos;
5. la doble divergencia de \(P\);
6. el lagrangiano, \(E_\phi\) y \(E_{ab}\).

En esta primera pasada \(f(r)\) sigue siendo arbitraria. Por eso el informe
muestra todos los momentos después del ansatz y antes de resolver la métrica.

Para resolver se introduce

\[
H(r)=1+\frac{\beta_0p^2\ell^2}{r^2},\qquad N(r)=H(r)f(r),
\]

con \(N'=2r/\ell^2\), \(N''=2/\ell^2\). La solución es

\[
f_{(2)}(r)=\frac{r^2/\ell^2-\lambda}
{1+\beta_0p^2\ell^2/r^2}.
\]

Como la doble divergencia puede generar hasta `f''''`, la sustitución final
incluye `f`, `f'`, `f''`, `f'''` y `f''''`. Después se reconstruyen
`P_case2_final`, `M_case2_final`, `J_case2_final`, `F_case2_final` y
`Rcal_case2_final`. Ninguno conserva la función abstracta `f(r)`.

Finalmente se verifican ecuaciones métricas y escalares, Bianchi,
Bianchi-Noether, Ricci, \(R\), \(R_{ab}R^{ab}\), Kretschmann y el flujo escalar
en el borde.

---

## 11. Por qué el orden importa: `mc_pipeline.py`

`run_pipeline()` ejecuta:

```python
build_general_theory(ctx)
build_case0(ctx)
build_case1(ctx)
build_case2(ctx)
evaluate_btz_ansatz(ctx)
evaluate_case1_ansatz(ctx)
evaluate_case2_ansatz(ctx)
```

Los cuatro primeros llamados registran teoría abstracta. Los tres últimos hacen
la evaluación coordenada. Las dependencias son:

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
```

El orden es parte de la arquitectura, no una preferencia estética. Es una forma
explícita de inyección de dependencias mediante `ctx`, sin variables globales.

---

## 12. El notebook como main

El notebook repite el pipeline por etapas para poder mostrar cada acto:

1. importa módulos y crea `ctx`;
2. construye y muestra la teoría general;
3. construye los Casos I sin ansatz;
4. evalúa los Casos II con ansatz;
5. muestra soluciones y residuos desde `ctx.objects`;
6. exige que todos los `ctx.checks` sean cero;
7. llama `export_results(ctx, compile_pdf=True)`.

Las listas `*_keys` filtran `ctx.steps` por grupo. El notebook no vuelve a hacer
la física: decide qué fragmento de la historia mostrar en cada sección.

---

## 13. Exportación: `mc_export.py`

`build_latex(ctx)` recorre los pasos en orden. Un `group` de la forma
`Sección::Subsección` se convierte automáticamente en estructura LaTeX. Para
cada paso imprime título, clave, igualdad y nota.

`_equation` ajusta fórmulas extensas al ancho del texto. Las matrices largas del
Caso 2 llegan ya separadas por componentes. Luego se agrega una tabla con los
residuos de `ctx.checks`: `OK` significa cero exacto, no una prueba numérica.

`export_results` genera:

- `salidas/derivacion_modelos_con_acoplamiento_casos0_2.tex`;
- `salidas/derivacion_modelos_con_acoplamiento_casos0_2.pdf`;
- `salidas/verificaciones_simbolicas.json`.

Intenta `latexmk` si también existe Perl; en caso contrario usa `pdflatex` dos
veces para resolver índice y referencias.

---

## 14. Pruebas: `test_symbolic.py`

La prueba reconstruye todo mediante `run_pipeline()`. Exige:

- un mínimo de pasos y que todos los residuos sean cero;
- ecuaciones métricas finales nulas para los tres casos;
- ecuaciones escalares correctas;
- \(R_{BTZ}=-6/\ell^2\) y la forma esperada de \(J_1^\varphi\);
- ausencia de `f(r)` en los momentos finales de los tres casos;
- \(H(r)f_{(2)}(r)=r^2/\ell^2-\lambda\).

La prueba de ausencia de `f` distingue entre mostrar una fórmula parcialmente
sustituida y haber usado realmente la solución explícita en todos los momentos.

---

## 15. Cómo rastrear un resultado

Ejemplo: momento métrico final del Caso 2.

1. `mc_general.py` define \(M_{ab}=\partial L/\partial g^{ab}\).
2. `build_case2` registra su forma tensorial.
3. `evaluate_case2_ansatz` construye `M` con la geometría y \(u_a\).
4. `M_case2_ansatz` lo guarda con \(f(r)\) arbitraria.
5. Se obtiene `f_case2_solution` mediante \(N=Hf\).
6. `M_final` sustituye la solución y todas sus derivadas.
7. `M_case2_final` conserva el objeto explícito.
8. `case2_final_M` lo incorpora a la narración por componentes.
9. `test_symbolic.py` verifica que ya no contiene `f`.
10. El notebook lo muestra y el exportador lo escribe en el PDF.

El mismo patrón se repite para \(P\), \(J\), \(F\), \(\mathcal R\), ecuaciones
de campo e invariantes.

---

## 16. Cómo añadir un caso nuevo

1. Crear `build_caseN(ctx)` para el nivel tensorial.
2. Registrar truncamiento, lagrangiano, acción y cuatro momentos.
3. Construir \(\mathcal R\), ecuaciones, Bianchi y frontera.
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

## 17. Ejecución reproducible en Windows/PowerShell

```powershell
.\.venv\Scripts\python.exe test_symbolic.py
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute derivacion_modelos_con_acoplamiento_MAIN.ipynb --output derivacion_modelos_con_acoplamiento_MAIN.ipynb --ExecutePreprocessor.cwd=. --ExecutePreprocessor.timeout=300
```

La primera orden reconstruye y prueba la cadena. La segunda ejecuta el notebook
y genera el informe. `requirements.txt` fija las dependencias Python; para el
PDF se requiere una instalación LaTeX con `latexmk` o `pdflatex`.

---

## 18. Mapa de archivos

```text
mc_core.py          Estado, pasos, objetos, checks y presentación
mc_general.py       Variación covariante de L(g,R,phi,nabla phi)
mc_geometry.py      Motor tensorial coordenado
mc_case0.py         Einstein-AdS y solución BTZ
mc_case1.py         Escalar mínimo y rama logarítmica
mc_case2.py         Acoplamiento con curvatura y rama racional
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

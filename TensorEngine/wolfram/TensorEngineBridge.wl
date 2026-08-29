(* Puente local JSON para Wolfram Engine y xAct. *)

ClearAll[
  writeResponse, request, requestPath, responsePath, args,
  loadPackage, packageVersionData, componentData, runtimeData,
  makeCheck, zeroCheck, phaseFiveValidation, phaseSixValidation, phaseSevenValidation,
  genericModelValidation, setupGenericEnvironment, decodeGenericExpr,
  decodeGenericIndex, genericCanonicalResidual, validGenericNameQ, genericSetupStep,
  summarizeChecks,
  canonicalResidual, residualString, xTensorLoadedAtStartup,
  xPertLoadedAtStartup, xTrasLoadedAtStartup, xCobaLoadedAtStartup
];

args = Rest[System`$ScriptCommandLine];
If[Length[args] < 2, Exit[64]];
requestPath = args[[-2]];
responsePath = args[[-1]];

writeResponse[data_] := Export[responsePath, data, "RawJSON"];

request = Quiet@Check[Import[requestPath, "RawJSON"], $Failed];
If[request === $Failed,
  writeResponse[<|
    "schema_version" -> "1.1",
    "status" -> "failed",
    "diagnostic" -> "No se pudo importar request.json."
  |>];
  Exit[65]
];

loadPackage[context_String] := Module[{},
  Quiet@Check[Needs[context], Null];
  MemberQ[System`$Packages, context]
];

packageVersionData[context_String] := Module[{symbolName, value},
  symbolName = context <> "$Version";
  If[!NameQ[symbolName], Return[<|"version" -> Null, "release_date" -> Null|>]];
  value = Quiet@Check[ToExpression[symbolName], $Failed];
  Which[
    value === $Failed,
      <|"version" -> Null, "release_date" -> Null|>,
    MatchQ[value, {_String, {_Integer ..}}],
      <|"version" -> value[[1]], "release_date" -> value[[2]]|>,
    StringQ[value],
      <|"version" -> value, "release_date" -> Null|>,
    True,
      <|"version" -> ToString[value, InputForm], "release_date" -> Null|>
  ]
];

componentData[available_, context_String] := Join[
  <|"available" -> TrueQ[available]|>,
  If[TrueQ[available], packageVersionData[context], <|"version" -> Null, "release_date" -> Null|>]
];

runtimeData[] := <|
  "wolfram_version" -> System`$Version,
  "wolfram_version_number" -> N[System`$VersionNumber],
  "wolfram_release_number" -> System`$ReleaseNumber,
  "system_id" -> System`$SystemID
|>;

(* Estas cargas deben ocurrir antes de analizar las definiciones que siguen. *)
xTensorLoadedAtStartup = loadPackage["xAct`xTensor`"];
xPertLoadedAtStartup = If[
  TrueQ[xTensorLoadedAtStartup],
  loadPackage["xAct`xPert`"],
  False
];
xTrasLoadedAtStartup = If[
  TrueQ[xTensorLoadedAtStartup],
  loadPackage["xAct`xTras`"],
  False
];
xCobaLoadedAtStartup = If[
  TrueQ[xTensorLoadedAtStartup] && Lookup[request, "operation", ""] === "verify_phase7",
  loadPackage["xAct`xCoba`"],
  False
];

residualString[value_] := ToString[Short[value, 8], InputForm];

makeCheck[
  key_String,
  status_String,
  message_String,
  residual_: Null,
  strategy_: Null,
  adjudicates_: {}
] := <|
  "key" -> key,
  "status" -> status,
  "message" -> message,
  "residual" -> residual,
  "strategy" -> strategy,
  "adjudicates" -> adjudicates
|>;

canonicalResidual[expression_] := ToCanonical[ContractMetric[expression]];

SetAttributes[zeroCheck, HoldRest];
zeroCheck[key_String, message_String, expression_] := Module[{value, reduced},
  value = Quiet@Check[expression, $Failed];
  If[value === $Failed,
    Return[makeCheck[key, "undetermined", message <> " xAct no pudo construir la expresion.", "$Failed"]]
  ];
  reduced = Quiet@Check[canonicalResidual[value], $Failed];
  If[reduced === $Failed,
    Return[makeCheck[key, "undetermined", message <> " xAct no pudo canonizar el residual.", residualString[value]]]
  ];
  If[TrueQ[reduced === 0],
    makeCheck[key, "passed", message],
    makeCheck[key, "failed", message, residualString[reduced]]
  ]
];

summarizeChecks[checks_List] := Counts[Lookup[checks, "status"]];

phaseFiveValidation[xTensorLoaded_, xPertLoaded_, xTrasLoaded_] := Module[
  {
    checks = {}, pEH, curvatureAlgebraic, metricEuler, einsteinExpected,
    thetaMetric, thetaMetricExpected, scalarIBP, perturbationRiemann,
    deltaGamma, palatiniMixed, perturbationAllDown, expectedAllDown,
    conventions, summary
  },
  If[!TrueQ[xTensorLoaded],
    Return[<|
      "schema_version" -> "1.1",
      "status" -> "failed",
      "operation" -> "verify_phase5",
      "runtime" -> runtimeData[],
      "components" -> <|
        "xact_xtensor" -> componentData[False, "xAct`xTensor`"],
        "xact_xpert" -> componentData[False, "xAct`xPert`"],
        "xact_xtras" -> componentData[False, "xAct`xTras`"],
        "xact_xcoba" -> componentData[False, "xAct`xCoba`"]
      |>,
      "conventions" -> <||>,
      "checks" -> {},
      "summary" -> <|"passed" -> 0, "failed" -> 0, "undetermined" -> 0|>,
      "diagnostic" -> "xAct`xTensor` no pudo cargarse."
    |>]
  ];

  DefManifold[TEMPhase5, 4, {a, b, c, d, e, f, i, j, k, l, m, n, p, q, r, s}];
  DefMetric[-1, teMetric[-a, -b], teCD, PrintAs -> "g"];

  pEH[x_, y_, z_, w_] := (teMetric[x, z] teMetric[y, w] - teMetric[x, w] teMetric[y, z])/2;

  AppendTo[checks, zeroCheck[
    "riemann_antisymmetry_first_pair",
    "R_abcd es antisimetrico en el primer par.",
    RiemannteCD[-a, -b, -c, -d] + RiemannteCD[-b, -a, -c, -d]
  ]];
  AppendTo[checks, zeroCheck[
    "riemann_antisymmetry_second_pair",
    "R_abcd es antisimetrico en el segundo par.",
    RiemannteCD[-a, -b, -c, -d] + RiemannteCD[-a, -b, -d, -c]
  ]];
  AppendTo[checks, zeroCheck[
    "riemann_pair_exchange",
    "R_abcd es simetrico bajo intercambio de pares.",
    RiemannteCD[-a, -b, -c, -d] - RiemannteCD[-c, -d, -a, -b]
  ]];
  If[TrueQ[xTrasLoaded],
    AppendTo[checks, zeroCheck[
      "riemann_first_bianchi",
      "La identidad algebraica de Bianchi se anula tras la proyeccion de Young de xTras.",
      RiemannYoungProject[
        RiemannteCD[-a, -b, -c, -d] + RiemannteCD[-a, -c, -d, -b] + RiemannteCD[-a, -d, -b, -c],
        teCD
      ]
    ]],
    AppendTo[checks, makeCheck[
      "riemann_first_bianchi",
      "undetermined",
      "xAct`xTras` no pudo cargarse; no se aplico el proyector multitemino de Bianchi.",
      "xTras unavailable"
    ]]
  ];
  AppendTo[checks, zeroCheck[
    "metric_compatibility",
    "La derivada covariante de la metrica se anula.",
    teCD[-a][teMetric[-b, -c]]
  ]];

  AppendTo[checks, zeroCheck[
    "eh_momentum_first_pair",
    "P_EH^{abcd} es antisimetrico en el primer par.",
    pEH[a, b, c, d] + pEH[b, a, c, d]
  ]];
  AppendTo[checks, zeroCheck[
    "eh_momentum_second_pair",
    "P_EH^{abcd} es antisimetrico en el segundo par.",
    pEH[a, b, c, d] + pEH[a, b, d, c]
  ]];
  AppendTo[checks, zeroCheck[
    "eh_momentum_pair_exchange",
    "P_EH^{abcd} es simetrico bajo intercambio de pares.",
    pEH[a, b, c, d] - pEH[c, d, a, b]
  ]];
  AppendTo[checks, zeroCheck[
    "eh_momentum_first_bianchi",
    "P_EH^{abcd} satisface la identidad ciclica.",
    pEH[a, b, c, d] + pEH[a, c, d, b] + pEH[a, d, b, c]
  ]];
  AppendTo[checks, zeroCheck[
    "eh_momentum_covariantly_constant",
    "La compatibilidad metrica implica nabla P_EH=0.",
    teCD[-q][pEH[a, b, c, d]]
  ]];

  curvatureAlgebraic = -(
    teMetric[-a, -p] pEH[p, c, d, e] RiemannteCD[-b, -c, -d, -e] +
    teMetric[-b, -p] pEH[p, c, d, e] RiemannteCD[-a, -c, -d, -e]
  )/2;
  AppendTo[checks, zeroCheck[
    "eh_curvature_algebraic_term",
    "El termino -P_(a^{cde} R_b)cde reduce a -R_ab.",
    curvatureAlgebraic + RicciteCD[-a, -b]
  ]];
  AppendTo[checks, zeroCheck[
    "eh_curvature_derivative_term",
    "El termino -2 nabla^c nabla^d P_acdb se anula para Einstein-Hilbert.",
    -2 teMetric[-a, -p] teMetric[-b, -s] teCD[-q][teCD[-r][pEH[p, q, r, s]]]
  ]];

  metricEuler = 2 RicciteCD[-a, -b] + curvatureAlgebraic -
    teMetric[-a, -b] RicciScalarteCD[]/2;
  einsteinExpected = RicciteCD[-a, -b] - teMetric[-a, -b] RicciScalarteCD[]/2;
  AppendTo[checks, zeroCheck[
    "einstein_hilbert_metric_euler",
    "La formula universal de E_ab reproduce el tensor de Einstein.",
    metricEuler - einsteinExpected
  ]];

  DefTensor[teDeltaG[a, b], TEMPhase5, Symmetric[{a, b}], PrintAs -> "delta-g"];
  thetaMetric = -2 pEH[a, b, c, d] teMetric[-b, -m] teMetric[-c, -n]
    teCD[-d][teDeltaG[m, n]];
  thetaMetricExpected = -teCD[-m][teDeltaG[m, a]] +
    teCD[a][teMetric[-m, -n] teDeltaG[m, n]];
  AppendTo[checks, zeroCheck[
    "einstein_hilbert_metric_boundary",
    "Theta_g^a tiene el signo y la normalizacion esperados para variaciones de g^{ab}.",
    thetaMetric - thetaMetricExpected
  ]];

  DefTensor[teDeltaPhi[], TEMPhase5, PrintAs -> "delta-phi"];
  DefTensor[teJ[a], TEMPhase5, PrintAs -> "J"];
  scalarIBP = teJ[a] teCD[-a][teDeltaPhi[]] - (
    -teCD[-a][teJ[a]] teDeltaPhi[] + teCD[-a][teJ[a] teDeltaPhi[]]
  );
  AppendTo[checks, zeroCheck[
    "scalar_integration_by_parts",
    "La integracion por partes produce E_phi=F_phi-nabla_a J^a y Theta_phi^a=J^a delta phi.",
    scalarIBP
  ]];

  If[TrueQ[xPertLoaded],
    DefMetricPerturbation[teMetric, teH, teEpsilon];
    deltaGamma[x_, y_, z_] := teMetric[x, l] (
      teCD[-y][teH[LI[1], -l, -z]] +
      teCD[-z][teH[LI[1], -l, -y]] -
      teCD[-l][teH[LI[1], -y, -z]]
    )/2;
    palatiniMixed = teCD[-c][deltaGamma[a, d, b]] - teCD[-d][deltaGamma[a, c, b]];
    (* Mapa de convenciones: R_TE^a_bcd = -R_xAct_cd b^a. *)
    perturbationRiemann = -ExpandPerturbation[Perturbation[RiemannteCD[-c, -d, -b, a]]];
    AppendTo[checks, zeroCheck[
      "palatini_mixed_xpert",
      "xPert confirma el signo de delta R^a_bcd=nabla_c delta Gamma^a_db-nabla_d delta Gamma^a_cb.",
      perturbationRiemann - palatiniMixed
    ]];
    AppendTo[checks, zeroCheck[
      "connection_variation_lower_symmetry",
      "La conexion variada es simetrica en sus dos indices inferiores.",
      deltaGamma[a, b, c] - deltaGamma[a, c, b]
    ]];
    AppendTo[checks, zeroCheck[
      "palatini_derivative_antisymmetry",
      "La variacion mixta de Riemann es antisimetrica en c,d.",
      palatiniMixed + (teCD[-d][deltaGamma[a, c, b]] - teCD[-c][deltaGamma[a, d, b]])
    ]];
    perturbationAllDown = -ExpandPerturbation[
      Perturbation[teMetric[-a, -e] RiemannteCD[-c, -d, -b, e]]
    ];
    expectedAllDown = -teH[LI[1], -a, -e] RiemannteCD[-c, -d, -b, e] +
      teMetric[-a, -e] (palatiniMixed /. a -> e);
    AppendTo[checks, zeroCheck[
      "all_down_curvature_variation_xpert",
      "xPert confirma la contribucion por variar el indice bajado de R_abcd.",
      perturbationAllDown - expectedAllDown
    ]],
    AppendTo[checks, makeCheck[
      "palatini_mixed_xpert",
      "undetermined",
      "xAct`xPert` no pudo cargarse; no se comparo la variacion de Riemann.",
      "xPert unavailable"
    ]]
  ];

  conventions = <|
    "metric_signature" -> -1,
    "dimension" -> 4,
    "riemann_sign" -> If[NameQ["xAct`xTensor`$RiemannSign"], ToString[ToExpression["xAct`xTensor`$RiemannSign"], InputForm], Null],
    "ricci_sign" -> If[NameQ["xAct`xTensor`$RicciSign"], ToString[ToExpression["xAct`xTensor`$RicciSign"], InputForm], Null],
    "tensor_engine_riemann_map" -> "R_TE^a_bcd = -R_xAct_cd b^a"
  |>;
  summary = summarizeChecks[checks];
  <|
    "schema_version" -> "1.1",
    "status" -> If[Lookup[summary, "failed", 0] > 0, "failed", If[Lookup[summary, "undetermined", 0] > 0, "partial", "success"]],
    "operation" -> "verify_phase5",
    "runtime" -> runtimeData[],
    "components" -> <|
      "xact_xtensor" -> componentData[xTensorLoaded, "xAct`xTensor`"],
      "xact_xpert" -> componentData[xPertLoaded, "xAct`xPert`"],
      "xact_xtras" -> componentData[xTrasLoaded, "xAct`xTras`"],
      "xact_xcoba" -> componentData[xCobaLoadedAtStartup, "xAct`xCoba`"]
    |>,
    "conventions" -> conventions,
    "checks" -> checks,
    "summary" -> <|
      "passed" -> Lookup[summary, "passed", 0],
      "failed" -> Lookup[summary, "failed", 0],
      "undetermined" -> Lookup[summary, "undetermined", 0]
    |>
  |>
];

phaseSixValidation[xTensorLoaded_, xPertLoaded_, xTrasLoaded_] := Module[
  {
    checks = {}, pEH, deltaInverseMetric, chargeEH, komarExpected,
    thetaEH, currentEH, constraintEH, divergenceChargeEH,
    scalarCurrent, scalarConstraint, scalarEulerMetric,
    generalCharge, pNonminimal, thetaNonminimal, currentNonminimal,
    eulerNonminimal, constraintNonminimal, chargeNonminimal,
    conventions, summary
  },
  If[!TrueQ[xTensorLoaded],
    Return[<|
      "schema_version" -> "1.1",
      "status" -> "failed",
      "operation" -> "verify_phase6",
      "runtime" -> runtimeData[],
      "components" -> <|
        "xact_xtensor" -> componentData[False, "xAct`xTensor`"],
        "xact_xpert" -> componentData[False, "xAct`xPert`"],
        "xact_xtras" -> componentData[False, "xAct`xTras`"],
        "xact_xcoba" -> componentData[False, "xAct`xCoba`"]
      |>,
      "conventions" -> <||>,
      "checks" -> {},
      "summary" -> <|"passed" -> 0, "failed" -> 0, "undetermined" -> 0|>,
      "diagnostic" -> "xAct`xTensor` no pudo cargarse."
    |>]
  ];

  DefManifold[TEMPhase6, 4, {a6, b6, c6, d6, e6, m6, n6, p6, q6}];
  DefMetric[-1, teMetric6[-a6, -b6], teCD6, PrintAs -> "g6"];
  DefTensor[teXi6[a6], TEMPhase6, PrintAs -> "xi"];
  DefTensor[tePhi6[], TEMPhase6, PrintAs -> "phi"];

  pEH[x_, y_, z_, w_] := (teMetric6[x, z] teMetric6[y, w] - teMetric6[x, w] teMetric6[y, z])/2;
  deltaInverseMetric[x_, y_] := -(teCD6[x][teXi6[y]] + teCD6[y][teXi6[x]]);

  AppendTo[checks, zeroCheck[
    "diffeomorphism_inverse_metric",
    "Lie_xi g^{ab}=-2 nabla^{(a}xi^{b)}.",
    LieD[teXi6[q6], teCD6][teMetric6[a6, b6]] - deltaInverseMetric[a6, b6]
  ]];
  AppendTo[checks, zeroCheck[
    "diffeomorphism_scalar",
    "Lie_xi phi=xi^a nabla_a phi.",
    LieD[teXi6[q6], teCD6][tePhi6[]] - teXi6[a6] teCD6[-a6][tePhi6[]]
  ]];

  chargeEH = -2 pEH[a6, b6, c6, d6] teCD6[-c6][teXi6[-d6]];
  komarExpected = -teCD6[a6][teXi6[b6]] + teCD6[b6][teXi6[a6]];
  AppendTo[checks, zeroCheck[
    "einstein_hilbert_wald_charge",
    "La carga de Wald de Einstein-Hilbert reduce al potencial de Komar.",
    chargeEH - komarExpected
  ]];
  AppendTo[checks, zeroCheck[
    "wald_charge_antisymmetry",
    "Q_xi^{ab} es antisimetrico.",
    chargeEH + (chargeEH /. {a6 -> b6, b6 -> a6})
  ]];

  thetaEH = -2 pEH[a6, b6, c6, d6] teMetric6[-b6, -m6] teMetric6[-c6, -n6]
    teCD6[-d6][deltaInverseMetric[m6, n6]];
  currentEH = thetaEH - teXi6[a6] RicciScalarteCD6[];
  constraintEH = 2 EinsteinteCD6[a6, -b6] teXi6[b6];
  divergenceChargeEH = teCD6[-b6][chargeEH];
  AppendTo[checks, zeroCheck[
    "einstein_hilbert_current_decomposition",
    "J_xi^a=2E^a_b xi^b+nabla_b Q_xi^{ab} para Einstein-Hilbert.",
    SortCovDs[
      ContractMetric[EinsteinToRicci[currentEH - constraintEH - divergenceChargeEH, teCD6]],
      teCD6
    ]
  ]];
  AppendTo[checks, zeroCheck[
    "einstein_hilbert_noether_identity",
    "La identidad de Noether reduce a la Bianchi contraida.",
    2 teCD6[-a6][EinsteinteCD6[a6, -b6]]
  ]];

  DefTensor[teP6[a6, b6, c6, d6], TEMPhase6, RiemannSymmetric[{a6, b6, c6, d6}], PrintAs -> "P"];
  generalCharge = -2 teP6[a6, b6, c6, d6] teCD6[-c6][teXi6[-d6]] +
    4 teXi6[-d6] teCD6[-c6][teP6[a6, b6, c6, d6]];
  AppendTo[checks, zeroCheck[
    "general_wald_charge_antisymmetry",
    "La formula general de Q_xi^{ab} hereda la antisimetria de P^{abcd}.",
    generalCharge + (generalCharge /. {a6 -> b6, b6 -> a6})
  ]];

  DefTensor[teF6[], TEMPhase6, PrintAs -> "f"];
  pNonminimal = teF6[] pEH[a6, b6, c6, d6];
  thetaNonminimal = -2 pNonminimal teMetric6[-b6, -m6] teMetric6[-c6, -n6]
      teCD6[-d6][deltaInverseMetric[m6, n6]] +
    2 teCD6[-d6][pNonminimal] teMetric6[-b6, -m6] teMetric6[-c6, -n6]
      deltaInverseMetric[m6, n6];
  currentNonminimal = thetaNonminimal - teXi6[a6] teF6[] RicciScalarteCD6[];
  eulerNonminimal = teF6[] EinsteinteCD6[-c6, -b6] +
    teMetric6[-c6, -b6] teCD6[-m6][teCD6[m6][teF6[]]] -
    teCD6[-c6][teCD6[-b6][teF6[]]];
  constraintNonminimal = 2 teMetric6[a6, c6] eulerNonminimal teXi6[b6];
  chargeNonminimal = -2 pNonminimal teCD6[-c6][teXi6[-d6]] +
    4 teXi6[-d6] teCD6[-c6][pNonminimal];
  AppendTo[checks, zeroCheck[
    "nonminimal_fR_current_decomposition",
    "El acoplamiento f(phi)R confirma el signo del termino 4 xi_d nabla_c P^{abcd}.",
    SortCovDs[
      ContractMetric[
        EinsteinToRicci[
          currentNonminimal - constraintNonminimal - teCD6[-b6][chargeNonminimal],
          teCD6
        ]
      ],
      teCD6
    ]
  ]];

  DefTensor[teU6[-a6], TEMPhase6, PrintAs -> "u"];
  DefTensor[teLScalar6[], TEMPhase6, PrintAs -> "Lphi"];
  scalarCurrent = -teU6[a6] teXi6[b6] teU6[-b6] - teXi6[a6] teLScalar6[];
  scalarEulerMetric = -teU6[-a6] teU6[-b6]/2 - teMetric6[-a6, -b6] teLScalar6[]/2;
  scalarConstraint = 2 teMetric6[a6, c6] (
    -teU6[-c6] teU6[-b6]/2 - teMetric6[-c6, -b6] teLScalar6[]/2
  ) teXi6[b6];
  AppendTo[checks, zeroCheck[
    "scalar_current_has_no_wald_charge",
    "El sector L(phi,nabla phi) satisface J_xi^a=2E^a_b xi^b sin carga Q adicional.",
    scalarCurrent - scalarConstraint
  ]];

  conventions = <|
    "metric_signature" -> -1,
    "dimension" -> 4,
    "current_definition" -> "J_xi^a = Theta^a(Lie_xi fields) - xi^a L",
    "charge_definition" -> "Q_xi^ab = -2 P^abcd nabla_c xi_d + 4 xi_d nabla_c P^abcd",
    "decomposition" -> "J_xi^a = 2 E^a_b xi^b + nabla_b Q_xi^ab"
  |>;
  summary = summarizeChecks[checks];
  <|
    "schema_version" -> "1.1",
    "status" -> If[Lookup[summary, "failed", 0] > 0, "failed", If[Lookup[summary, "undetermined", 0] > 0, "partial", "success"]],
    "operation" -> "verify_phase6",
    "runtime" -> runtimeData[],
    "components" -> <|
      "xact_xtensor" -> componentData[xTensorLoaded, "xAct`xTensor`"],
      "xact_xpert" -> componentData[xPertLoaded, "xAct`xPert`"],
      "xact_xtras" -> componentData[xTrasLoaded, "xAct`xTras`"],
      "xact_xcoba" -> componentData[xCobaLoadedAtStartup, "xAct`xCoba`"]
    |>,
    "conventions" -> conventions,
    "checks" -> checks,
    "summary" -> <|
      "passed" -> Lookup[summary, "passed", 0],
      "failed" -> Lookup[summary, "failed", 0],
      "undetermined" -> Lookup[summary, "undetermined", 0]
    |>
  |>
];

phaseSevenValidation[xTensorLoaded_, xCobaLoaded_] := Module[
  {
    checks = {}, metric7, inverse7, covd7, pd7, gamma7, ricci7,
    ricciScalar7, einstein7, scale7, scalar7, gradient7, box7,
    conventions, summary, appendCheck, reduced, zeroShape, ctensorData
  },
  If[!TrueQ[xTensorLoaded] || !TrueQ[xCobaLoaded],
    Return[<|
      "schema_version" -> "1.1",
      "status" -> "failed",
      "operation" -> "verify_phase7",
      "runtime" -> runtimeData[],
      "components" -> <|
        "xact_xtensor" -> componentData[xTensorLoaded, "xAct`xTensor`"],
        "xact_xpert" -> componentData[xPertLoadedAtStartup, "xAct`xPert`"],
        "xact_xtras" -> componentData[xTrasLoadedAtStartup, "xAct`xTras`"],
        "xact_xcoba" -> componentData[xCobaLoaded, "xAct`xCoba`"]
      |>,
      "conventions" -> <||>,
      "checks" -> {},
      "summary" -> <|"passed" -> 0, "failed" -> 0, "undetermined" -> 0|>,
      "diagnostic" -> "xAct`xTensor` o xAct`xCoba` no pudo cargarse."
    |>]
  ];

  DefManifold[TEMPhase7, 4, {a7, b7, c7, d7}];
  DefChart[teFLRW7, TEMPhase7, {0, 1, 2, 3}, {teT7[], teX7[], teY7[], teZ7[]}];
  DefScalarFunction[teScale7];
  DefScalarFunction[tePhi7];

  scale7 = teScale7[teT7[]];
  scalar7 = tePhi7[teT7[]];
  metric7 = CTensor[
    DiagonalMatrix[{-1, scale7^2, scale7^2, scale7^2}],
    {-teFLRW7, -teFLRW7}
  ];
  SetCMetric[metric7, teFLRW7, SignatureOfMetric -> {3, 1, 0}];
  MetricCompute[
    metric7,
    teFLRW7,
    {
      "Christoffel"[1, -1, -1], "Ricci"[-1, -1],
      "RicciScalar"[], "Einstein"[-1, -1]
    },
    CVSimplify -> Simplify,
    Verbose -> False
  ];

  ctensorData[value_] := value /. CTensor[array_, ___] :> array;
  inverse7 = Simplify[Inverse[ctensorData[metric7]]];
  covd7 = CovDOfMetric[metric7];
  pd7 = PDOfBasis[teFLRW7];
  gamma7 = ctensorData[Christoffel[covd7, pd7]];
  ricci7 = ctensorData[Ricci[covd7]];
  ricciScalar7 = ctensorData[RicciScalar[covd7]];
  einstein7 = ctensorData[Einstein[covd7]];

  appendCheck[key_String, message_String, expression_] := Module[{value},
    value = Quiet@Check[Simplify[Together[expression]], $Failed];
    If[value === $Failed,
      AppendTo[checks, makeCheck[key, "undetermined", message, "$Failed"]],
      zeroShape = ConstantArray[0, Dimensions[value]];
      If[TrueQ[value === zeroShape],
        AppendTo[checks, makeCheck[key, "passed", message]],
        AppendTo[checks, makeCheck[key, "failed", message, residualString[value]]]
      ]
    ]
  ];

  appendCheck[
    "flrw_metric_inverse",
    "xCoba confirma que la metrica FLRW y su inversa producen la identidad.",
    ctensorData[metric7].inverse7 - IdentityMatrix[4]
  ];
  appendCheck[
    "flrw_christoffel_time_space_space",
    "xCoba obtiene Gamma^0_ii=a a' para las tres direcciones espaciales.",
    Table[gamma7[[1, i, i]] - scale7 Derivative[1][teScale7][teT7[]], {i, 2, 4}]
  ];
  appendCheck[
    "flrw_christoffel_space_time_space",
    "xCoba obtiene Gamma^i_0i=Gamma^i_i0=a'/a.",
    Flatten@Table[
      {
        gamma7[[i, 1, i]] - Derivative[1][teScale7][teT7[]]/scale7,
        gamma7[[i, i, 1]] - Derivative[1][teScale7][teT7[]]/scale7
      },
      {i, 2, 4}
    ]
  ];
  appendCheck[
    "flrw_ricci_components",
    "xCoba reproduce R_00=-3 a''/a y R_ii=a a''+2(a')^2.",
    Join[
      {ricci7[[1, 1]] + 3 Derivative[2][teScale7][teT7[]]/scale7},
      Table[
        ricci7[[i, i]] - scale7 Derivative[2][teScale7][teT7[]] -
          2 Derivative[1][teScale7][teT7[]]^2,
        {i, 2, 4}
      ]
    ]
  ];
  appendCheck[
    "flrw_ricci_scalar",
    "xCoba reproduce R=6(a a''+(a')^2)/a^2.",
    ricciScalar7 - 6 (
      scale7 Derivative[2][teScale7][teT7[]] +
      Derivative[1][teScale7][teT7[]]^2
    )/scale7^2
  ];
  appendCheck[
    "flrw_einstein_components",
    "xCoba reproduce G_00=3(a'/a)^2 y G_ii=-(a')^2-2 a a''.",
    Join[
      {einstein7[[1, 1]] - 3 Derivative[1][teScale7][teT7[]]^2/scale7^2},
      Table[
        einstein7[[i, i]] + Derivative[1][teScale7][teT7[]]^2 +
          2 scale7 Derivative[2][teScale7][teT7[]],
        {i, 2, 4}
      ]
    ]
  ];
  appendCheck[
    "flrw_off_diagonal_vanish",
    "Las componentes fuera de la diagonal de Ricci y Einstein se anulan.",
    Flatten@Table[
      If[i == j, 0, {ricci7[[i, j]], einstein7[[i, j]]}],
      {i, 1, 4}, {j, 1, 4}
    ]
  ];

  gradient7 = Table[pd7[{i, -teFLRW7}][scalar7], {i, 0, 3}];
  box7 = Sum[
    inverse7[[i + 1, j + 1]] (
      pd7[{i, -teFLRW7}][gradient7[[j + 1]]] -
      Sum[gamma7[[k + 1, i + 1, j + 1]] gradient7[[k + 1]], {k, 0, 3}]
    ),
    {i, 0, 3}, {j, 0, 3}
  ];
  appendCheck[
    "flrw_homogeneous_scalar_box",
    "xCoba confirma Box(phi)=-phi''-3(a'/a)phi' para phi(t).",
    box7 + Derivative[2][tePhi7][teT7[]] +
      3 Derivative[1][teScale7][teT7[]] Derivative[1][tePhi7][teT7[]]/scale7
  ];

  conventions = <|
    "metric_signature" -> -1,
    "dimension" -> 4,
    "ansatz" -> "ds^2=-dt^2+a(t)^2(dx^2+dy^2+dz^2), phi=phi(t)",
    "curvature_convention" -> "TensorEngine phase0 mapped to xCoba Levi-Civita"
  |>;
  summary = summarizeChecks[checks];
  <|
    "schema_version" -> "1.1",
    "status" -> If[Lookup[summary, "failed", 0] > 0, "failed", If[Lookup[summary, "undetermined", 0] > 0, "partial", "success"]],
    "operation" -> "verify_phase7",
    "runtime" -> runtimeData[],
    "components" -> <|
      "xact_xtensor" -> componentData[xTensorLoaded, "xAct`xTensor`"],
      "xact_xpert" -> componentData[xPertLoadedAtStartup, "xAct`xPert`"],
      "xact_xtras" -> componentData[xTrasLoadedAtStartup, "xAct`xTras`"],
      "xact_xcoba" -> componentData[xCobaLoaded, "xAct`xCoba`"]
    |>,
    "conventions" -> conventions,
    "checks" -> checks,
    "summary" -> <|
      "passed" -> Lookup[summary, "passed", 0],
      "failed" -> Lookup[summary, "failed", 0],
      "undetermined" -> Lookup[summary, "undetermined", 0]
    |>
  |>
];

(* Fases 11-12: transporte IR enumerado y validación ligada algebraico-diferencial. *)
validGenericNameQ[value_] := StringQ[value] &&
  StringMatchQ[value, RegularExpression["^[A-Za-z][A-Za-z0-9_]*$"]];

SetAttributes[genericSetupStep, HoldRest];
genericSetupStep[label_String, expression_] := Module[{value},
  $te11Progress = "setup:" <> label;
  value = Catch[
    Quiet@Check[expression; "success", $Failed],
    _,
    Function[{thrown, tag}, $Failed]
  ];
  If[value === $Failed, $te11SetupDiagnostic = label];
  value
];

setupGenericEnvironment[model_Association, indexNames_List] := Module[
  {dimensionData, dimension, names, indexSymbols, functionData, parameterData, symbols, head},
  If[!AllTrue[indexNames, validGenericNameQ], Return[$Failed]];
  names = DeleteDuplicates@Join[indexNames, {"teFallbackA", "teFallbackB", "teFallbackC"}];
  indexSymbols = Symbol["TensorEngineGeneric`i$" <> #] & /@ names;
  $te11IndexMap = AssociationThread[names, indexSymbols];
  $te11Manifold = Symbol["TensorEngineGeneric`M"];
  $te11Metric = Symbol["TensorEngineGeneric`g"];
  $te11CD = Symbol["TensorEngineGeneric`CD"];
  $te11Phi = Symbol["TensorEngineGeneric`phi"];
  $te11Xi = Symbol["TensorEngineGeneric`xi"];
  $te11Volume = Symbol["TensorEngineGeneric`volume"];
  $te11Delta = Symbol["TensorEngineGeneric`delta"];
  $te11DeltaGamma = Symbol["TensorEngineGeneric`deltaGamma"];
  $te11SetupDiagnostic = "unknown";

  dimensionData = Lookup[model, "dimension", <|"value" -> 4|>];
  dimension = Lookup[dimensionData, "value", 4];
  If[StringQ[dimension],
    If[!validGenericNameQ[dimension], Return[$Failed]];
    dimension = Symbol["TensorEngineGeneric`constant$" <> dimension];
    If[genericSetupStep["DefConstantSymbol dimension", DefConstantSymbol[Evaluate[dimension]]] === $Failed, Return[$Failed]]
  ];
  symbols = Lookup[model, "symbols", <||>];
  $te11MetricName = Lookup[symbols, "metric", "g"];
  $te11CurvatureName = Lookup[symbols, "curvature", "Riemann"];
  $te11ScalarName = Lookup[symbols, "scalar", "phi"];
  $te11GradientName = Lookup[symbols, "scalar_gradient", "u"];
  If[!AllTrue[
      {$te11MetricName, $te11CurvatureName, $te11ScalarName, $te11GradientName},
      validGenericNameQ
    ], Return[$Failed]];

  If[genericSetupStep[
      "DefManifold",
      DefManifold[Evaluate[$te11Manifold], dimension, Evaluate[indexSymbols]]
    ] === $Failed, Return[$Failed]];
  If[genericSetupStep[
      "DefMetric",
      DefMetric[
        -1,
        Evaluate[$te11Metric[-indexSymbols[[1]], -indexSymbols[[2]]]],
        Evaluate[$te11CD],
        PrintAs -> "g"
      ]
    ] === $Failed, Return[$Failed]];
  If[genericSetupStep[
      "DefTensor phi",
      DefTensor[Evaluate[$te11Phi[]], Evaluate[$te11Manifold], PrintAs -> "phi"]
    ] === $Failed, Return[$Failed]];
  If[genericSetupStep[
      "DefTensor xi",
      DefTensor[Evaluate[$te11Xi[indexSymbols[[1]]]], Evaluate[$te11Manifold], PrintAs -> "xi"]
    ] === $Failed, Return[$Failed]];
  If[genericSetupStep[
      "DefTensor volume",
      DefTensor[Evaluate[$te11Volume[]], Evaluate[$te11Manifold], PrintAs -> "sqrt(-g)"]
    ] === $Failed, Return[$Failed]];
  If[genericSetupStep[
      "DefTensor delta",
      DefTensor[
        Evaluate[$te11Delta[-indexSymbols[[1]], indexSymbols[[2]]]],
        Evaluate[$te11Manifold],
        PrintAs -> "delta"
      ]
    ] === $Failed, Return[$Failed]];
  If[genericSetupStep[
      "DefTensor deltaGamma",
      DefTensor[
        Evaluate[$te11DeltaGamma[indexSymbols[[1]], -indexSymbols[[2]], -indexSymbols[[3]]]],
        Evaluate[$te11Manifold],
        PrintAs -> "deltaGamma"
      ]
    ] === $Failed, Return[$Failed]];
  $te11Riemann = Riemann[$te11CD];

  $te11FunctionMap = <||>;
  functionData = Lookup[model, "functions", {}];
  Do[
    If[!AssociationQ[item] || !validGenericNameQ[Lookup[item, "name", ""]], Return[$Failed]];
    head = Symbol["TensorEngineGeneric`function$" <> item["name"]];
    If[genericSetupStep["DefScalarFunction " <> item["name"], DefScalarFunction[Evaluate[head]]] === $Failed, Return[$Failed]];
    AssociateTo[$te11FunctionMap, item["name"] -> head],
    {item, functionData}
  ];

  $te11ScalarMap = <||>;
  parameterData = Lookup[model, "parameters", {}];
  Do[
    If[!AssociationQ[item] || !validGenericNameQ[Lookup[item, "name", ""]], Return[$Failed]];
    head = Symbol["TensorEngineGeneric`constant$" <> item["name"]];
    If[genericSetupStep["DefConstantSymbol " <> item["name"], DefConstantSymbol[Evaluate[head]]] === $Failed, Return[$Failed]];
    AssociateTo[$te11ScalarMap, item["name"] -> head],
    {item, parameterData}
  ];
  If[StringQ[Lookup[dimensionData, "value", 4]],
    AssociateTo[$te11ScalarMap, Lookup[dimensionData, "value"] -> dimension]
  ];
  True
];

decodeGenericIndex[data_Association] := Module[{name, symbol, variance},
  name = Lookup[data, "name", ""];
  variance = Lookup[data, "variance", ""];
  If[!KeyExistsQ[$te11IndexMap, name], Return[$Failed]];
  symbol = $te11IndexMap[name];
  Which[variance === "up", symbol, variance === "down", -symbol, True, $Failed]
];

decodeGenericExpr[data_Association] := Module[
  {type, children, indices, name, head, orders, derivative},
  type = Lookup[data, "type", ""];
  Switch[type,
    "number",
      Lookup[data, "numerator", 0]/Lookup[data, "denominator", 1],
    "scalar",
      name = Lookup[data, "name", ""];
      Which[
        name === $te11ScalarName, $te11Phi[],
        KeyExistsQ[$te11ScalarMap, name], $te11ScalarMap[name],
        True, $Failed
      ],
    "tensor",
      name = Lookup[data, "name", ""];
      indices = decodeGenericIndex /@ Lookup[data, "indices", {}];
      If[MemberQ[indices, $Failed], Return[$Failed]];
      Switch[name,
        $te11MetricName, Apply[$te11Metric, indices],
        $te11CurvatureName, If[Length[indices] === 4, -Apply[$te11Riemann, indices[[{3, 4, 2, 1}]]], $Failed],
        $te11GradientName, If[Length[indices] === 1, Apply[$te11CD, indices][$te11Phi[]], $Failed],
        "xi", If[Length[indices] === 1, Apply[$te11Xi, indices], $Failed],
        "delta", Apply[$te11Metric, indices],
        "delta_Gamma", Apply[$te11DeltaGamma, indices],
        _, $Failed
      ],
    "add",
      children = decodeGenericExpr /@ Lookup[data, "terms", {}];
      If[MemberQ[children, $Failed], $Failed, Total[children]],
    "mul",
      children = decodeGenericExpr /@ Lookup[data, "factors", {}];
      If[MemberQ[children, $Failed], $Failed, Times @@ children],
    "power",
      children = decodeGenericExpr /@ {Lookup[data, "base", <||>], Lookup[data, "exponent", <||>]};
      If[MemberQ[children, $Failed], $Failed, children[[1]]^children[[2]]],
    "function",
      name = Lookup[data, "name", ""];
      If[!KeyExistsQ[$te11FunctionMap, name], Return[$Failed]];
      children = decodeGenericExpr /@ Lookup[data, "arguments", {}];
      If[MemberQ[children, $Failed], $Failed, Apply[$te11FunctionMap[name], children]],
    "function_derivative",
      name = Lookup[data, "name", ""];
      If[!KeyExistsQ[$te11FunctionMap, name], Return[$Failed]];
      children = decodeGenericExpr /@ Lookup[data, "arguments", {}];
      orders = Lookup[data, "derivative_orders", {}];
      If[MemberQ[children, $Failed], Return[$Failed]];
      derivative = Apply[Derivative, orders][$te11FunctionMap[name]];
      Apply[derivative, children],
    "covariant_derivative",
      indices = decodeGenericIndex[Lookup[data, "index", <||>]];
      children = decodeGenericExpr[Lookup[data, "operand", <||>]];
      If[indices === $Failed || children === $Failed, $Failed, Apply[$te11CD, {indices}][children]],
    "variation",
      children = decodeGenericExpr[Lookup[data, "operand", <||>]];
      If[children === $Failed, $Failed, Perturbation[children]],
    "volume_element",
      $te11Volume[],
    _,
      $Failed
  ]
];

genericCanonicalResidual[expression_, xTrasLoaded_, strategy_] := Module[{value, candidate},
  value = expression;
  value = Quiet@Check[ToCanonical[ContractMetric[value]], $Failed];
  If[value === $Failed, Return[$Failed]];
  Which[
    strategy === "riemann_bianchi",
      If[TrueQ[xTrasLoaded],
        value = Quiet@Check[RiemannYoungProject[value, $te11CD], value]
      ];
      Quiet@Check[ToCanonical[ContractMetric[value]], $Failed],
    strategy === "differential",
      value = Quiet@Check[SortCovDs[value, $te11CD], $Failed];
      If[value === $Failed, Return[$Failed]];
      value = Quiet@Check[ToCanonical[ContractMetric[value]], $Failed];
      If[value === $Failed, Return[$Failed]];
      If[TrueQ[xTrasLoaded],
        candidate = Quiet@Check[FullSimplification[value], $Failed];
        If[candidate =!= $Failed && TrueQ[candidate === 0], Return[0]];
        candidate = Quiet@Check[
          RicciToEinsteinCC[0][value, $te11CD],
          $Failed
        ];
        If[candidate =!= $Failed,
          candidate = Quiet@Check[SortCovDs[candidate, $te11CD], $Failed];
          If[candidate =!= $Failed,
            candidate = Quiet@Check[ToCanonical[ContractMetric[candidate]], $Failed]
          ];
          If[candidate =!= $Failed && TrueQ[candidate === 0], Return[0]]
        ]
      ];
      value,
    True,
      value
  ]
];

genericModelValidation[xTensorLoaded_, xPertLoaded_, xTrasLoaded_] := Module[
  {options, subject, model, indexNames, checkData, checks = {}, decoded, reduced,
   status, summary, conventions, strategy, adjudicates},
  $te11Progress = "generic:start";
  options = Lookup[request, "options", <||>];
  subject = Lookup[options, "subject", <||>];
  model = Lookup[options, "model", <||>];
  indexNames = Lookup[options, "indices", {}];
  checkData = Lookup[options, "checks", {}];
  If[!TrueQ[xTensorLoaded] || !AssociationQ[subject] || !AssociationQ[model] ||
      !ListQ[indexNames] || !ListQ[checkData] ||
      setupGenericEnvironment[model, indexNames] === $Failed,
    Return[<|
      "schema_version" -> "1.3",
      "status" -> "failed",
      "operation" -> "verify_model",
      "subject" -> subject,
      "runtime" -> runtimeData[],
      "components" -> <|
        "xact_xtensor" -> componentData[xTensorLoaded, "xAct`xTensor`"],
        "xact_xpert" -> componentData[xPertLoaded, "xAct`xPert`"],
        "xact_xtras" -> componentData[xTrasLoaded, "xAct`xTras`"],
        "xact_xcoba" -> componentData[False, "xAct`xCoba`"]
      |>,
      "conventions" -> <||>,
      "checks" -> {},
      "summary" -> <|"passed" -> 0, "failed" -> 0, "undetermined" -> 0|>,
      "diagnostic" -> "No se pudo inicializar el entorno IR-xAct del modelo: " <> ToString[$te11SetupDiagnostic]
    |>]
  ];

  $te11Progress = "generic:checks";
  Do[
    $te11Progress = "check:" <> ToString[Lookup[item, "key", "invalid_check"]];
    strategy = ToString[Lookup[item, "strategy", "algebraic"]];
    adjudicates = Lookup[item, "adjudicates", {}];
    decoded = Catch[
      Quiet@Check[decodeGenericExpr[Lookup[item, "residual", <||>]], $Failed],
      _,
      Function[{thrown, tag}, $Failed]
    ];
    If[decoded === $Failed,
      AppendTo[checks, makeCheck[
        ToString[Lookup[item, "key", "invalid_check"]],
        "undetermined",
        ToString[Lookup[item, "message", ""]] <> " El residual IR no pudo transportarse.",
        "IR decode failed",
        strategy,
        adjudicates
      ]],
      reduced = Catch[
        genericCanonicalResidual[decoded, xTrasLoaded, strategy],
        _,
        Function[{thrown, tag}, $Failed]
      ];
      If[reduced === $Failed,
        AppendTo[checks, makeCheck[
          ToString[Lookup[item, "key", "invalid_check"]],
          "undetermined",
          ToString[Lookup[item, "message", ""]] <> " xAct no pudo canonizar el residual.",
          residualString[decoded],
          strategy,
          adjudicates
        ]],
        If[TrueQ[reduced === 0],
          AppendTo[checks, makeCheck[
            ToString[Lookup[item, "key", "invalid_check"]],
            "passed",
            ToString[Lookup[item, "message", ""]],
            Null,
            strategy,
            adjudicates
          ]],
          status = If[Lookup[item, "on_nonzero", "failed"] === "undetermined", "undetermined", "failed"];
          AppendTo[checks, makeCheck[
            ToString[Lookup[item, "key", "invalid_check"]],
            status,
            ToString[Lookup[item, "message", ""]],
            residualString[reduced],
            strategy,
            adjudicates
          ]]
        ]
      ]
    ],
    {item, checkData}
  ];
  summary = summarizeChecks[checks];
  conventions = <|
    "tensor_engine_convention_id" -> Lookup[Lookup[model, "conventions", <||>], "convention_id", Null],
    "tensor_engine_riemann_map" -> "R_TE^a_bcd = -R_xAct_cd b^a",
    "dimension" -> Lookup[Lookup[model, "dimension", <||>], "value", Null]
  |>;
  <|
    "schema_version" -> "1.3",
    "status" -> If[Lookup[summary, "failed", 0] > 0, "failed", If[Lookup[summary, "undetermined", 0] > 0, "partial", "success"]],
    "operation" -> "verify_model",
    "subject" -> subject,
    "runtime" -> runtimeData[],
    "components" -> <|
      "xact_xtensor" -> componentData[xTensorLoaded, "xAct`xTensor`"],
      "xact_xpert" -> componentData[xPertLoaded, "xAct`xPert`"],
      "xact_xtras" -> componentData[xTrasLoaded, "xAct`xTras`"],
      "xact_xcoba" -> componentData[False, "xAct`xCoba`"]
    |>,
    "conventions" -> conventions,
    "checks" -> checks,
    "summary" -> <|
      "passed" -> Lookup[summary, "passed", 0],
      "failed" -> Lookup[summary, "failed", 0],
      "undetermined" -> Lookup[summary, "undetermined", 0]
    |>
  |>
];

Catch[Module[{operation, response},
  operation = Lookup[request, "operation", Missing["operation"]];
  response = Switch[
    operation,
    "ping",
      <|
        "schema_version" -> "1.1",
        "status" -> "success",
        "wolfram_version" -> System`$Version,
        "xact_available" -> TrueQ[xTensorLoadedAtStartup],
        "xact_version" -> Lookup[packageVersionData["xAct`xTensor`"], "version", Null],
        "runtime" -> runtimeData[],
        "components" -> <|
          "xact_xtensor" -> componentData[xTensorLoadedAtStartup, "xAct`xTensor`"],
          "xact_xpert" -> componentData[xPertLoadedAtStartup, "xAct`xPert`"],
          "xact_xtras" -> componentData[xTrasLoadedAtStartup, "xAct`xTras`"],
          "xact_xcoba" -> componentData[xCobaLoadedAtStartup, "xAct`xCoba`"]
        |>
      |>,
    "verify_phase5",
      phaseFiveValidation[xTensorLoadedAtStartup, xPertLoadedAtStartup, xTrasLoadedAtStartup],
    "verify_phase6",
      phaseSixValidation[xTensorLoadedAtStartup, xPertLoadedAtStartup, xTrasLoadedAtStartup],
    "verify_phase7",
      phaseSevenValidation[xTensorLoadedAtStartup, xCobaLoadedAtStartup],
    "verify_model",
      Catch[
        genericModelValidation[xTensorLoadedAtStartup, xPertLoadedAtStartup, xTrasLoadedAtStartup],
        _,
        Function[{thrown, tag}, <|
          "schema_version" -> "1.3",
          "status" -> "failed",
          "operation" -> "verify_model",
          "subject" -> Lookup[Lookup[request, "options", <||>], "subject", <||>],
          "runtime" -> runtimeData[],
          "components" -> <|
            "xact_xtensor" -> componentData[xTensorLoadedAtStartup, "xAct`xTensor`"],
            "xact_xpert" -> componentData[xPertLoadedAtStartup, "xAct`xPert`"],
            "xact_xtras" -> componentData[xTrasLoadedAtStartup, "xAct`xTras`"],
            "xact_xcoba" -> componentData[False, "xAct`xCoba`"]
          |>,
          "conventions" -> <||>,
          "checks" -> {},
          "summary" -> <|"passed" -> 0, "failed" -> 0, "undetermined" -> 0|>,
          "diagnostic" -> "Throw no controlado durante verify_model."
        |>]
      ],
    _,
      <|
        "schema_version" -> "1.1",
        "status" -> "unsupported",
        "diagnostic" -> "La operacion aun no tiene traductor IR-xAct registrado."
      |>
  ];
  writeResponse[response]
],
  _,
  Function[{thrown, tag},
    writeResponse[<|
      "schema_version" -> "1.3",
      "status" -> "failed",
      "operation" -> "verify_model",
      "subject" -> Lookup[Lookup[request, "options", <||>], "subject", <||>],
      "runtime" -> runtimeData[],
      "components" -> <|
        "xact_xtensor" -> componentData[xTensorLoadedAtStartup, "xAct`xTensor`"],
        "xact_xpert" -> componentData[xPertLoadedAtStartup, "xAct`xPert`"],
        "xact_xtras" -> componentData[xTrasLoadedAtStartup, "xAct`xTras`"],
        "xact_xcoba" -> componentData[False, "xAct`xCoba`"]
      |>,
      "conventions" -> <||>,
      "checks" -> {},
      "summary" -> <|"passed" -> 0, "failed" -> 0, "undetermined" -> 0|>,
      "diagnostic" -> "Throw capturado en " <> ToString[$te11Progress]
    |>]
  ]
];

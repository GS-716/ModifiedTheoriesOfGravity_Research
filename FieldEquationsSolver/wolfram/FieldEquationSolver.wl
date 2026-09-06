(* Coordinate-scalar adapter for the existing IR, isolated from xTensor symbols.
   No ToExpression of input strings, and no numerical solvers. *)
ClearAll[fsSymbol, fsDecode, fsEncode, fsText, fsTimed, fsSubstitution, fieldEquationSolve];
fsSymbol[name_String] := If[KeyExistsQ[$fsNames, name], $fsNames[name],
  With[{s = Symbol["TEFieldSolve`v" <> ToString[Length[$fsNames] + 1]]},
    AssociateTo[$fsNames, name -> s]; s]];
fsDecode[d_Association] := Switch[Lookup[d, "type", ""],
  "number", d["numerator"]/d["denominator"],
  "scalar", fsSymbol[d["name"]],
  "add", Total[fsDecode /@ d["terms"]],
  "mul", Times @@ (fsDecode /@ d["factors"]),
  "power", fsDecode[d["base"]]^fsDecode[d["exponent"]],
  "function", With[{h = Lookup[<|"sin" -> Sin, "cos" -> Cos, "exp" -> Exp,
      "log" -> Log, "tan" -> Tan, "sinh" -> Sinh, "cosh" -> Cosh|>, d["name"], fsSymbol[d["name"]]]},
      h @@ (fsDecode /@ d["arguments"])],
  "function_derivative", (Derivative @@ d["derivative_orders"])[fsSymbol[d["name"]]] @@ (fsDecode /@ d["arguments"]),
  _, Throw[<|"reason" -> "Nodo no escalar en resolución", "node" -> d|>, "fsDecode"]
];
fsDecode[d_] := Throw[<|"reason" -> "Nodo escalar malformado", "node" -> d|>, "fsDecode"];
fsText[e_] := ToString[e /. Normal[AssociationThread[Values[$fsNames], Keys[$fsNames]]], InputForm];
fsEncode[e_] := Module[{h = Head[e], name, args, orders},
  Which[
    IntegerQ[e] || Head[e] === Rational, <|"type" -> "number", "numerator" -> Numerator[e], "denominator" -> Denominator[e]|>,
    h === Symbol && MemberQ[Values[$fsNames], e],
      <|"type" -> "scalar", "name" -> First[Keys[Select[$fsNames, # === e &]]]|>,
    MatchQ[e, C[_Integer]], <|"type" -> "scalar", "name" -> ($fsConstantPrefix <> ToString[e[[1]]])|>,
    h === Plus, <|"type" -> "add", "terms" -> (fsEncode /@ List @@ e)|>,
    h === Times, <|"type" -> "mul", "factors" -> (fsEncode /@ List @@ e)|>,
    h === Power && e[[1]] === System`E,
      <|"type" -> "function", "name" -> "exp", "arguments" -> {fsEncode[e[[2]]]}|>,
    h === Power, <|"type" -> "power", "base" -> fsEncode[e[[1]]], "exponent" -> fsEncode[e[[2]]]|>,
    Head[h] === Derivative || (Head[Head[h]] === Derivative),
      Throw["Derivada de salida no representable: " <> fsText[e], "fsEncode"],
    MemberQ[Values[$fsNames], h],
      name = First[Keys[Select[$fsNames, # === h &]]];
      <|"type" -> "function", "name" -> name, "arguments" -> (fsEncode /@ List @@ e)|>,
    MemberQ[{Log, Exp, Sin, Cos, Tan, Sinh, Cosh}, h],
      name = Lookup[<|Log -> "log", Exp -> "exp", Sin -> "sin", Cos -> "cos", Tan -> "tan", Sinh -> "sinh", Cosh -> "cosh"|>, h];
      <|"type" -> "function", "name" -> name, "arguments" -> (fsEncode /@ List @@ e)|>,
    True, Throw["Salida formal conservada en Wolfram: " <> fsText[e], "fsEncode"]
  ]
];
SetAttributes[fsTimed, HoldAll];
fsTimed[e_] := Quiet[TimeConstrained[CheckAbort[e, $Aborted], $fsLimit, $Aborted]];
fsSubstitution[rules_] := rules /. Rule[lhs_, rhs_] /; Head[lhs] =!= Symbol :>
  With[{h = Head[lhs], arguments = List @@ lhs, value = rhs}, h -> Function[Evaluate[arguments], Evaluate[value]]];

fieldEquationSolve[options_Association] := Catch[Module[
  {eq, original, unknowns, funcs, parameters, active, vars, guards, equations,
   jets, jetSymbols, jetRules, lifted, reduced, eliminated, solved, formal = {},
   operations = {}, candidates = {}, diagnostics = {}, elimination, output, record,
   rules, substitution, residuals, encoded, localVars, attempts, assumptions, constants, fits, fitted = {},
   reducedBranches, branchRules, entry},
  $fsNames = <||>;
  $fsLimit = Min[30, Max[1, Lookup[options, "time_limit", 15]]];
  eq = fsDecode /@ Lookup[options, "equations", {}];
  original = fsDecode /@ Lookup[options, "original_equations", {}];
  unknowns = fsDecode /@ Lookup[options, "unknowns", {}];
  guards = fsDecode /@ Lookup[options, "nonzero", {}];
  assumptions = Function[d, With[{lhs = fsDecode[d["lhs"]], rhs = fsDecode[d["rhs"]]},
    Switch[d["op"], "eq", lhs == rhs, "ne", lhs != rhs, "gt", lhs > rhs,
      "ge", lhs >= rhs, "lt", lhs < rhs, "le", lhs <= rhs]]] /@ Lookup[options, "assumptions", {}];
  elimination = fsDecode /@ Lookup[options, "eliminate", {}];
  $fsConstantPrefix = "integrationConstant";
  While[AnyTrue[Keys[$fsNames], StringStartsQ[#, $fsConstantPrefix] &], $fsConstantPrefix = $fsConstantPrefix <> "X"];
  funcs = Select[unknowns, Head[#] =!= Symbol &];
  parameters = Complement[unknowns, funcs];
  active = Select[funcs, Function[u, !FreeQ[eq, u] || With[{h = Head[u]}, !FreeQ[eq, Derivative[__][h][__]]]]];
  vars = DeleteDuplicates[Flatten[List @@@ active]];
  equations = Thread[eq == ConstantArray[0, Length[eq]]];
  (* Algebraic reduction of jets is a necessary relation, not integration. *)
  jets = DeleteDuplicates[Join[Cases[eq, Derivative[__][_][__], Infinity], active]];
  jetSymbols = Table[Unique["TEFieldSolve`jet"], {Length[jets]}];
  jetRules = Thread[jets -> jetSymbols];
  lifted = equations /. jetRules;
  record[name_, value_, note_] := <|"operation" -> name,
    "status" -> If[value === $Aborted, "timeout", If[!FreeQ[value, Solve | Reduce | Eliminate | DSolve], "unevaluated", "evaluated"]],
    "expression" -> fsText[value], "note" -> note|>;
  If[Length[Join[jetSymbols, parameters]] > 0 && Length[eq] > 0,
    reduced = fsTimed[Reduce[And @@ Join[lifted, assumptions /. jetRules, Thread[(guards /. jetRules) != 0]], Join[jetSymbols, parameters], Reals]];
    AppendTo[operations, record["Reduce", reduced, "Relación algebraica de jets; no certifica integrabilidad ni enumera todas las soluciones diferenciales."]];
    If[jets === {} && reduced =!= $Aborted && reduced =!= False,
      reducedBranches = With[{expanded = LogicalExpand[reduced]}, If[Head[expanded] === Or, List @@ expanded, {expanded}]];
      Do[
        branchRules = Quiet@Check[ToRules[branch], {}];
        If[MatchQ[branchRules, {(_Rule)..}],
          AppendTo[formal, <|"rules" -> branchRules, "origin" -> "Wolfram Reduce"|>]],
        {branch, reducedBranches}]
    ];
    solved = fsTimed[Solve[lifted, Join[jetSymbols, parameters]]];
    AppendTo[operations, record["Solve", solved, "Las ramas con jets requieren integración posterior."]];
    If[jets === {} && MatchQ[solved, {___List}],
      formal = Join[formal, (<|"rules" -> #, "origin" -> "Wolfram Solve"|> & /@ solved)]];
  ];
  If[elimination =!= {},
    eliminated = fsTimed[Eliminate[lifted, elimination /. jetRules]];
    AppendTo[operations, record["Eliminate", eliminated, "Consecuencia proyectada; las ecuaciones originales se conservan para verificar."]],
    AppendTo[operations, <|"operation" -> "Eliminate", "status" -> "not_requested", "note" -> "No se solicitaron incógnitas a eliminar."|>]
  ];
  If[active =!= {} && Length[eq] > 0,
    solved = fsTimed[DSolve[equations, active, vars]];
    AppendTo[operations, record["DSolve", solved, "Sin condiciones iniciales ni de frontera."]];
    If[MatchQ[solved, {___List}],
      formal = Join[formal, (<|"rules" -> #, "origin" -> "Wolfram DSolve"|> & /@ solved)]];
    (* A single equation may yield candidate families for an overdetermined
       system. Every candidate is checked against ALL original equations. *)
    If[formal === {} && Length[active] === 1 && Length[vars] === 1,
      attempts = Take[SortBy[Select[equations, Function[e,
        !FreeQ[e, First[active]] || With[{h = Head[First[active]]}, !FreeQ[e, Derivative[__][h][__]]]]], LeafCount], UpTo[3]];
      Do[
        solved = fsTimed[DSolve[one, active, vars]];
        AppendTo[operations, record["DSolve", solved, "Candidato de una ecuación; requiere sustitución en todo el sistema."]];
        If[MatchQ[solved, {___List}],
          formal = Join[formal, (<|"rules" -> #, "origin" -> "Wolfram DSolve (ecuación individual)"|> & /@ solved)]],
        {one, attempts}]
    ],
    AppendTo[operations, <|"operation" -> "DSolve", "status" -> "not_applicable", "note" -> "No hay funciones activas para integrar."|>]
  ];
  (* Enforce the remaining equations on integration constants without any
     initial/boundary data. Reject coordinate-dependent "constant" rules. *)
  Do[
    rules = entry["rules"];
    If[!MatchQ[rules, {(_Rule)..}], Continue[]];
    constants = DeleteDuplicates[Cases[rules, System`C[_Integer], Infinity]];
    If[constants === {}, Continue[]];
    residuals = fsTimed[FullSimplify[original /. fsSubstitution[rules]]];
    If[!ListQ[residuals] || AllTrue[residuals, # === 0 &], Continue[]];
    fits = fsTimed[Solve[Thread[residuals == ConstantArray[0, Length[residuals]]], constants]];
    AppendTo[operations, record["Solve", fits, "Restricciones sobre constantes de integración desde todas las ecuaciones originales."]];
    If[MatchQ[fits, {___List}],
      Do[If[AllTrue[fit, Function[rule, AllTrue[vars, FreeQ[Last[rule], #] &]]],
        AppendTo[fitted, <|"rules" -> (rules /. fit),
          "origin" -> (entry["origin"] <> " + Solve de constantes")|>]], {fit, fits}]],
    {entry, DeleteDuplicates[formal]}];
  formal = Join[formal, fitted];
  Do[
    rules = entry["rules"];
    If[!MatchQ[rules, {(_Rule)..}], Continue[]];
    rules = With[{simplified = fsTimed[FullSimplify[rules, And @@ Join[assumptions, Thread[guards != 0]]]]},
      If[MatchQ[simplified, {(_Rule)..}], simplified, rules]];
    encoded = Catch[(Function[rule, {fsEncode[rule[[1]]], fsEncode[rule[[2]]]}] /@ rules), "fsEncode"];
    If[!ListQ[encoded], AppendTo[diagnostics, ToString[encoded]]; Continue[]];
    substitution = fsSubstitution[rules];
    residuals = fsTimed[FullSimplify[original /. substitution, And @@ Thread[(guards /. substitution) != 0]]];
    AppendTo[candidates, <|"rules" -> encoded, "origin" -> entry["origin"],
      "integration_constants" -> (fsEncode /@ DeleteDuplicates[Cases[rules, System`C[_Integer], Infinity]]),
      "verification" -> If[original =!= {} && ListQ[residuals] && Length[residuals] === Length[original] && AllTrue[residuals, # === 0 &], "verified", "undetermined"],
      "residuals" -> If[ListQ[residuals], fsText /@ residuals, fsText[residuals]]|>],
    {entry, DeleteDuplicates[formal]}];
  <|"status" -> "evaluated", "wolfram_version" -> System`$Version,
    "xact_validation" -> "inherited_from_source_run_only",
    "operations" -> operations, "candidates" -> candidates, "diagnostics" -> diagnostics,
    "jet_mapping" -> MapThread[<|"symbol" -> ToString[#1, InputForm], "expression" -> fsText[#2]|> &, {jetSymbols, jets}],
    "unresolved" -> "Completitud y ramas singulares no certificadas; comprobar el dominio de cada familia."|>
], "fsDecode", Function[{data, tag}, <|"status" -> "unavailable", "diagnostics" -> {ToString[data, InputForm]}, "failed_node" -> data|>]];

(* Standalone JSON entry point used by FieldEquationWolframBridge. *)
Module[{args, requestPath, responsePath, request, response},
  args = Rest[System`$ScriptCommandLine];
  If[Length[args] < 2, Exit[64]];
  requestPath = args[[-2]];
  responsePath = args[[-1]];
  request = Quiet@Check[Import[requestPath, "RawJSON"], $Failed];
  If[request === $Failed,
    Export[responsePath, <|"status" -> "unavailable", "diagnostics" -> {"No se pudo importar request.json."}|>, "RawJSON"];
    Exit[65]
  ];
  response = If[
    Lookup[request, "operation", ""] === "solve_field_equations",
    fieldEquationSolve[Lookup[request, "options", <||>]],
    <|"status" -> "unsupported", "diagnostics" -> {"Operación no soportada por FieldEquationsSolver."}|>
  ];
  Export[responsePath, response, "RawJSON"];
];

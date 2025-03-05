BeginPackage["doubleQWExperimentalModel`"];

(* Unprotect and clear all package symbols *)
Unprotect @@ Names["doubleQWExperimentalModel`*"];
ClearAll @@ Names["doubleQWExperimentalModel`*"];

(* define public symbols *)
experimentalDoubleQwEvolution;


Begin["`Private`"];

(* WE ONLY USE THE LR BASIS IN THIS HOUSE *)

walkerSpaceDimension = 5;
maxOccupationNumber = (walkerSpaceDimension - 1)/2;
(* these are the indices corresponding to the walker starting in position "0" *)
inputStateIndices = {walkerSpaceDimension, walkerSpaceDimension + 1};

(* the weird way to chose the element with 1 is to convert between notation with (-2, -1, 0, 1, 2) as indices to the natural Mathematica one with (1, 2, 3, 4, 5) *)
(* polarization convention is here R->{1, 0}, L->{0, 1} *)
base["R", n_Integer] := Normal @ Flatten @ KroneckerProduct[SparseArray[{n + maxOccupationNumber + 1 -> 1}, walkerSpaceDimension], {1, 0}];
base["L", n_Integer] := Normal @ Flatten @ KroneckerProduct[SparseArray[{n + maxOccupationNumber + 1 -> 1}, walkerSpaceDimension], {0, 1}];


(* define qplate unitary *)
QP[alpha_, delta_] := Plus[
    Cos[delta / 2] * KroneckerProduct[IdentityMatrix @ walkerSpaceDimension, IdentityMatrix @ 2],
    I * Sin[delta / 2] * Sum[
        (# + ConjugateTranspose @ #) &@ (Exp[2 I alpha] KroneckerProduct[base["L", n], Conjugate @ base["R", n + 1] ]),
        {n, -maxOccupationNumber, maxOccupationNumber - 1}
    ]
];

(* lambda mezzi *)
HWP[theta_] := KroneckerProduct[
    IdentityMatrix @ walkerSpaceDimension,
    {
        {0, Exp[2 I theta]},
        {Exp[-2 I theta], 0}
    }
];

QWP[varphi_] := KroneckerProduct[
    IdentityMatrix @ walkerSpaceDimension, 
    {
        {1 + I, (1 - I) Exp[2 I varphi]},
        {(1 - I) Exp[-2 I varphi], 1 + I}
    } / 2
];

(* define the unitary describing A SINGLE quantum walk (without specifying input state or final projection) *)
qwUnitary[{alpha2_, delta2_ : Pi}, {varphi2_, theta2_, zeta2_}, {alpha1_, delta1_}] := Dot[
    QP[alpha2, delta2]  (* second qplate) *),
    QWP[varphi2] . HWP[theta2] . QWP[zeta2], (* coin operation *)
    QP[alpha1, delta1] (* first qplate *)
];


projectUnitaryOnState[unitary_, polarizationState_] := Dot[
    (* first space is walker, second space is polarization *)
    KroneckerProduct[IdentityMatrix @ walkerSpaceDimension, ConjugateTranspose @ polarizationState],
    unitary
];

(* project the polarization on the specified polarization state (input space is left unchanged) *)
qwUnitaryAfterPolarizationProjection[{varphip_, thetap_}, {alpha2_, delta2_ : Pi}, {varphi2_, theta2_, zeta2_}, {alpha1_, delta1_}] := projectUnitaryOnState[
    qwUnitary[{alpha2, delta2}, {varphi2, theta2, zeta2}, {alpha1, delta1}],
    {Cos[thetap], Sin[thetap] * Exp[I * varphip]}
];

(* project also on the possible input states *)
qwProjectedIsometry[{varphip_, thetap_}, {alpha2_, delta2_ : Pi}, {varphi2_, theta2_, zeta2_}, {alpha1_, delta1_}] := qwUnitaryAfterPolarizationProjection[
    {varphip, thetap},
    {alpha2, delta2},
    {varphi2, theta2, zeta2},
    {alpha1, delta1}
][[All, inputStateIndices]];



(* THIS IS THE "OLD" MODEL, WITH zeta2 and phi2 switched (wrt the optimal one) *)
experimentalDoubleQwEvolutionBad = KroneckerProduct[
    With[{
            alpha1 = 19*Pi/180, delta1 = Pi/2,
            alpha2 = 77*Pi/180, delta2 = Pi,
            zeta2 = 2.908160255954825,
            varphi2 = 1.0070977275570985,
            theta2 = 1.5570696348907915,
            thetap = 0.7853981644358207, varphip = 0.6904379401618504
        },
        qwProjectedIsometry[{varphip, thetap}, {alpha2, delta2}, {varphi2, theta2, zeta2}, {alpha1, delta1}]
    ],
    With[{
            alpha1 = 336*Pi/180, delta1 = Pi/2,
            alpha2 = 163*Pi/180, delta2 = Pi,
            zeta2 = 2.89289820797498,
            varphi2 = 1.095755783171672,
            theta2 = 1.5937676381596888,
            thetap = 0.7853981643272621, varphip = 0.6567152340642829
        },
        qwProjectedIsometry[{varphip, thetap}, {alpha2, delta2}, {varphi2, theta2, zeta2}, {alpha1, delta1}]
    ]
] // Chop;


experimentalDoubleQwEvolution = KroneckerProduct[
    With[{
            alpha1 = 19*Pi/180, delta1 = Pi/2,
            alpha2 = 77*Pi/180, delta2 = Pi,
            zeta2 = 1.0070977275570985,
            varphi2 = 2.908160255954825,
            theta2 = 1.5570696348907915,
            thetap = 0.7853981644358207, varphip = 0.6904379401618504
        },
        qwProjectedIsometry[{varphip, thetap}, {alpha2, delta2}, {varphi2, theta2, zeta2}, {alpha1, delta1}]
    ],
    With[{
            alpha1 = 336*Pi/180, delta1 = Pi/2,
            alpha2 = 163*Pi/180, delta2 = Pi,
            zeta2 = 1.095755783171672,
            varphi2 = 2.89289820797498,
            theta2 = 1.5937676381596888,
            thetap = 0.7853981643272621, varphip = 0.6567152340642829
        },
        qwProjectedIsometry[{varphip, thetap}, {alpha2, delta2}, {varphi2, theta2, zeta2}, {alpha1, delta1}]
    ]
] // Chop;



experimentalDoubleQwEvolutionOrthogonalProjection = KroneckerProduct[
    With[{
            alpha1 = 19*Pi/180, delta1 = Pi/2,
            alpha2 = 77*Pi/180, delta2 = Pi,
            zeta2 = 2.908160255954825,
            varphi2 = 1.0070977275570985,
            theta2 = 1.5570696348907915,
            thetap = Pi/2 - 0.7853981644358207, varphip = -0.6904379401618504
        },
        qwProjectedIsometry[{varphip, thetap}, {alpha2, delta2}, {varphi2, theta2, zeta2}, {alpha1, delta1}]
    ],
    With[{
            alpha1 = 336*Pi/180, delta1 = Pi/2,
            alpha2 = 163*Pi/180, delta2 = Pi,
            zeta2 = 2.89289820797498,
            varphi2 = 1.095755783171672,
            theta2 = 1.5937676381596888,
            thetap = Pi/2 - 0.7853981643272621, varphip = -0.6567152340642829
        },
        qwProjectedIsometry[{varphip, thetap}, {alpha2, delta2}, {varphi2, theta2, zeta2}, {alpha1, delta1}]
    ]
] // Chop;


experimentalDoubleQwEvolutionInvertedAlphas = KroneckerProduct[
    With[{
            alpha1 = 19*Pi/180, delta1 = Pi/2,
            alpha2 = 336*Pi/180, delta2 = Pi,
            zeta2 = 2.908160255954825,
            varphi2 = 1.0070977275570985,
            theta2 = 1.5570696348907915,
            thetap = Pi/2 - 0.7853981644358207, varphip = -0.6904379401618504
        },
        qwProjectedIsometry[{varphip, thetap}, {alpha2, delta2}, {varphi2, theta2, zeta2}, {alpha1, delta1}]
    ],
    With[{
            alpha1 = 77*Pi/180, delta1 = Pi/2,
            alpha2 = 163*Pi/180, delta2 = Pi,
            zeta2 = 2.89289820797498,
            varphi2 = 1.095755783171672,
            theta2 = 1.5937676381596888,
            thetap = Pi/2 - 0.7853981643272621, varphip = -0.6567152340642829
        },
        qwProjectedIsometry[{varphip, thetap}, {alpha2, delta2}, {varphi2, theta2, zeta2}, {alpha1, delta1}]
    ]
] // Chop;


(* renormalised result is

{{0.128929, 0, 0, 0}, {0.121485 - 0.0291699 I, 
  0.0958133 - 0.0862707 I, 0, 
  0}, {0.11745 - 0.042603 I, -0.0707627 + 0.102967 I, 0, 
  0}, {-0.107588 + 0.0710466 I, 0.0587755 - 0.11025 I, 0, 0}, {0, 
  0.0324141 - 0.124788 I, 0, 0}, {-0.0243756 - 0.122537 I, 
  0, -0.0793769 - 0.101598 I, 
  0}, {-0.0506918 - 0.109947 I, -0.100108 - 0.0747525 I, -0.09778 - 
   0.0777729 I, -0.126971 - 0.0223884 I}, {-0.062696 - 0.103573 I, 
  0.11124 + 0.0477873 I, -0.105881 - 0.066323 I, 
  0.124705 - 0.00763103 I}, {0.0878649 + 0.0888218 I, -0.115896 - 
   0.0350176 I, 
  0.122223 + 0.0410399 I, -0.123064 + 0.0215607 I}, {0, -0.12473 - 
   0.0072144 I, 0, -0.118291 + 0.0512847 I}, {0.114273 - 0.0505094 I, 
  0, 0.0815536 - 0.0946497 I, 0}, {0.0962476 - 0.073447 I, 
  0.0511241 - 0.113999 I, 
  0.0554306 - 0.107636 I, -0.00272686 - 0.124908 I}, {0.0874086 - 
   0.0837723 I, -0.0223803 + 0.118984 I, 0.0430167 - 0.113171 I, 
  0.0308296 + 0.117079 I}, {-0.0675246 + 0.105119 I, 
  0.00890266 - 0.120743 I, -0.0158975 + 0.123923 I, -0.0437584 - 
   0.112886 I}, {0, -0.0201578 - 0.123301 I, 
  0, -0.0711063 - 0.10273 I}, {0.0741284 + 0.105488 I, 
  0, -0.110156 - 0.0589518 I, 0}, {0.0937147 + 0.0826263 I, 
  0.125674 + 0.0287916 I, -0.117133 - 0.0306256 I, -0.121308 + 
   0.0298988 I}, {0.102386 + 0.0716014 I, -0.124931 + 
   0.00130408 I, -0.119828 - 0.0173036 I, 
  0.107539 - 0.055618 I}, {-0.119987 - 0.0471787 I, 
  0.123998 - 0.015299 I, 
  0.124407 - 0.0115076 I, -0.100628 + 0.0673213 I}, {0, 
  0.120737 - 0.0452267 I, 0, -0.0847524 + 0.0917964 I}, {0, 
  0, -0.037488 + 0.123359 I, 0}, {0, 0, -0.00741387 + 0.124718 I, 
  0.0546843 + 0.116758 I}, {0, 0, 
  0.00661212 + 0.124763 I, -0.0779431 - 0.0976444 I}, {0, 
  0, -0.0366943 - 0.123597 I, 0.0883966 + 0.0882927 I}, {0, 0, 0, 
  0.109972 + 0.0672975 I}}


 *)



(* Protect all package symbols *)
With[{syms = Names["doubleQWExperimentalModel`*"]},
  SetAttributes[syms, {Protected, ReadProtected}]
];

End[];  (* End `Private` *)

EndPackage[];

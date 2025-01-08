BeginPackage["experimentalDataLoader`", {"QELM`"}];

loadExperimentalData::usage = "experimentalDataLoader[] loads the experimental data from the data files.";
trainQELMFromExperimentalData;


Begin["`Private`"];

(* using the convention in the experiment (I THINK THIS IS IN THE FUCKING LR BASIS) *)
localPolarizationUnitary[θ_, φ_] := With[
    {
        QWP = (1/2) * {
            {1 + I, (1 - I) E^(2 I θ)},
            {(1 - I) E^(-2 I θ), 1 + I}
        },
        HWP = {
            {0, E^(2 I φ)},
            {E^(-2 I φ), 0}
        }
    },
    Dot[QWP, HWP]
];

(* this produces tensors of Dimension num_files x num_states x 2 x 2 (for each state there's two angles per qubit) *)
loadAnglesData3009[path_, filenames_List] := (Pi/180) * Import[FileNameJoin[{path, #}], "CSV"][[1]] & /@ filenames // Transpose;
loadAnglesData3009[path_, filename_String] := loadAnglesData3009[path, {filename}];

loadAnglesData2009[path_, filenames_List] := (Pi/180) * Flatten@Import[FileNameJoin[{path, #}], "CSV"] & /@ filenames // Transpose;
loadAnglesData2009[path_, filename_String] := loadAnglesData2009[path, {filename}];


(* this produces tensors of Dimension num_files x num_outcomes x num_states *)
loadCountsData[path_, filenames_List] := Partition[Flatten@Import[FileNameJoin[{path, #}], "Table"], 25] & /@ filenames // Transpose[#, {1, 3, 2}]&;
loadCountsData[path_, filename_String] := loadCountsData[path, {filename}];

(* this produces tensors of Dimension num_files x num_states *)

(* Load the 30-09 experimental data *)
Options[loadExperimentalData] = {"referenceInputStates" -> None};
loadExperimentalData["30-09", opts : OptionsPattern[] ] := loadExperimentalData["30-09", opts] = Module[
    {
        dataSeptemberDir,
        anglesEntStates, anglesSepStates,
        referenceStateEnt, referenceStateSep, referenceStateSep2,
        inputUnitariesEnt, inputUnitariesSep,
        inputStatesEnt, inputStatesSep, inputStatesSep2,
        countsVectorsEntAllReps, countsVectorsSepAllReps, countsVectorsSepAllReps2
    },
    
    dataSeptemberDir = FileNameJoin[{NotebookDirectory[], "experimental data", "dati 30-09"}];
    
    anglesEntStates = Transpose@{
        loadAnglesData3009[dataSeptemberDir, {"Angoli_QWP1_ent.txt", "Angoli_HWP1_ent.txt"}],
        loadAnglesData3009[dataSeptemberDir, {"Angoli_QWP2_ent.txt", "Angoli_HWP2_ent.txt"}]
    };
    
    anglesSepStates = Transpose[{
        loadAnglesData3009[dataSeptemberDir, {"Angoli_QWP1_sep.txt", "Angoli_HWP1_sep.txt"}],
        loadAnglesData3009[dataSeptemberDir, {"Angoli_QWP2_sep.txt", "Angoli_HWP2_sep.txt"}]
    }][[;; ;; 2]];

    inputUnitariesEnt = Table[
        KroneckerProduct[
            localPolarizationUnitary[angles[[1, 1]], angles[[1, 2]]],
            localPolarizationUnitary[angles[[2, 1]], angles[[2, 2]]]
        ],
        {angles, anglesEntStates}
    ];
    
    inputUnitariesSep = Table[
        KroneckerProduct[
            localPolarizationUnitary[angles[[1, 1]], angles[[1, 2]]],
            localPolarizationUnitary[angles[[2, 1]], angles[[2, 2]]]
        ],
        {angles, anglesSepStates}
    ];
    
    (* in this case we have two sep references b/c each angle is used for both an HV and a VH input reference *)
    If[OptionValue @ "referenceInputStates" =!= None,
        {referenceStateEnt, referenceStateSep, referenceStateSep2} = {#["ent"], #["sep1"], #["sep2"]}& @ OptionValue @ "referenceInputStates",
        (* default ref states are \Phi^-, HV, and VH, respectively (in the LR basis) *)
        {referenceStateEnt, referenceStateSep, referenceStateSep2} = {Normalize@{0, 1, -1, 0}, Normalize@{1, 1, -1, -1}, Normalize@{1, -1, 1, -1}}
    ];
    (* input states as kets *)
    inputStatesEnt = Dot[#, referenceStateEnt] & /@ inputUnitariesEnt;
    inputStatesSep = Dot[#, referenceStateSep] & /@ inputUnitariesSep;
    inputStatesSep2 = Dot[#, referenceStateSep2] & /@ inputUnitariesSep;
    
    countsVectorsEntAllReps = loadCountsData[dataSeptemberDir, {"ccENT_rep_1.txt", "ccENT_rep_2.txt", "ccENT_rep_3.txt", "ccENT_rep_4.txt"}];
    (* extract even and odd elements, for each repetition, for the separable data (even corresponds to HV inputs, odd to VH inputs) *)
    {countsVectorsSepAllReps, countsVectorsSepAllReps2} = {#[[All, All, ;; ;; 2]], #[[All, All, 2;; ;; 2]]} & @ loadCountsData[dataSeptemberDir, {"ccSEP_rep_1.txt", "ccSEP_rep_2.txt", "ccSEP_rep_3.txt", "ccSEP_rep_4.txt"}];
    
    experimentalData @ <|
        "dataPath" -> dataSeptemberDir,
        "dataFiles" -> FileNameTake /@ FileNames[All, dataSeptemberDir],
        "statesAsKets" -> True, (* flag to mark whether states are stored as kets, rather than as dms *)
        "angles" -> <|
            "ent" -> anglesEntStates,
            "sep1" -> anglesSepStates,
            "sep2" -> anglesSepStates
        |>,
        "referenceStates" -> <|
            "ent" -> referenceStateEnt,
            "sep1" -> referenceStateSep,
            "sep2" -> referenceStateSep2
        |>,
        "inputStates" -> <|
            "ent" -> inputStatesEnt,
            "sep1" -> inputStatesSep,
            "sep2" -> inputStatesSep2
        |>,
        "counts" -> <|
            "ent" -> countsVectorsEntAllReps,
            "sep1" -> countsVectorsSepAllReps,
            "sep2" -> countsVectorsSepAllReps2
        |>
    |>
];
loadExperimentalData["20-09", opts : OptionsPattern[] ] := loadExperimentalData["20-09", opts] = Module[
    {
        data2009Dir,
        anglesEntStates, anglesSepStates,
        referenceStateEnt, referenceStateSep, referenceStateSep2,
        inputUnitariesEnt, inputUnitariesSep,
        inputStatesEnt, inputStatesSep, inputStatesSep2,
        countsVectorsEntAllReps, countsVectorsSepAllReps, countsVectorsSepAllReps2
    },
    
    data2009Dir = FileNameJoin[{NotebookDirectory[], "experimental data", "dati 20-09"}];
    
    anglesEntStates = Transpose@{
        loadAnglesData2009[data2009Dir, {"Angoli_QWP_ent.txt", "Angoli_HWP_ent.txt"}],
        loadAnglesData2009[data2009Dir, {"Angoli_QWP_ent.txt", "Angoli_HWP_ent.txt"}]
    };
    
    anglesSepStates = Transpose[{
        loadAnglesData2009[data2009Dir, {"Angoli_QWP_sep.txt", "Angoli_HWP_sep.txt"}],
        loadAnglesData2009[data2009Dir, {"Angoli_QWP_sep.txt", "Angoli_HWP_sep.txt"}]
    }];

    inputUnitariesEnt = Table[
        KroneckerProduct[
            localPolarizationUnitary[angles[[1, 1]], angles[[1, 2]]],
            localPolarizationUnitary[angles[[2, 1]], angles[[2, 2]]]
        ],
        {angles, anglesEntStates}
    ];
    
    inputUnitariesSep = Table[
        KroneckerProduct[
            localPolarizationUnitary[angles[[1, 1]], angles[[1, 2]]],
            localPolarizationUnitary[angles[[2, 1]], angles[[2, 2]]]
        ],
        {angles, anglesSepStates}
    ];
    
    If[OptionValue @ "referenceInputStates" =!= None,
        {referenceStateEnt, referenceStateSep} = {#["ent"], #["sep"]}& @ OptionValue @ "referenceInputStates",
        {referenceStateEnt, referenceStateSep} = {Normalize@{1, 0, 0, -1}, Normalize@{1, 1, -1, -1}}
    ];
    (* input states as kets *)
    inputStatesEnt = Dot[#, referenceStateEnt] & /@ inputUnitariesEnt // Chop;
    inputStatesSep = Dot[#, referenceStateSep] & /@ inputUnitariesSep // Chop;  
    
    countsVectorsEntAllReps = loadCountsData[data2009Dir, {"ccENT_rep_1.txt", "ccENT_rep_2.txt", "ccENT_rep_3.txt", "ccENT_rep_4.txt"}];
    countsVectorsSepAllReps = loadCountsData[data2009Dir, {"ccSEP_rep_1.txt", "ccSEP_rep_2.txt", "ccSEP_rep_3.txt", "ccSEP_rep_4.txt"}];
    
    experimentalData @ <|
        "dataPath" -> data2009Dir,
        "dataFiles" -> FileNameTake /@ FileNames[All, data2009Dir],
        "statesAsKets" -> True, (* flag to mark whether states are stored as kets, rather than as dms *)
        "angles" -> <|
            "ent" -> anglesEntStates,
            "sep" -> anglesSepStates
        |>,
        "referenceStates" -> <|
            "ent" -> referenceStateEnt,
            "sep" -> referenceStateSep
        |>,
        "inputStates" -> <|
            "ent" -> inputStatesEnt,
            "sep" -> inputStatesSep
        |>,
        "counts" -> <|
            "ent" -> countsVectorsEntAllReps,
            "sep" -> countsVectorsSepAllReps
        |>
    |>
];
loadExperimentalData["02-09", opts : OptionsPattern[] ] := loadExperimentalData["02-09", opts] = Module[
    {
        data2009Dir,
        anglesEntStates, anglesSepStates,
        referenceStateEnt, referenceStateSep, referenceStateSep2,
        inputUnitariesEnt, inputUnitariesSep,
        inputStatesEnt, inputStatesSep, inputStatesSep2,
        countsVectorsEntAllReps, countsVectorsSepAllReps, countsVectorsSepAllReps2
    },
    
    data0209Dir = FileNameJoin[{NotebookDirectory[], "experimental data", "dati 02-09"}];
    
    anglesEntStates = Transpose@{
        loadAnglesData2009[data0209Dir, {"Angoli_QWP_ent.txt", "Angoli_HWP_ent.txt"}],
        loadAnglesData2009[data0209Dir, {"Angoli_QWP_ent.txt", "Angoli_HWP_ent.txt"}]
    };
    
    anglesSepStates = Transpose[{
        loadAnglesData2009[data0209Dir, {"Angoli_QWP_sep.txt", "Angoli_HWP_sep.txt"}],
        loadAnglesData2009[data0209Dir, {"Angoli_QWP_sep.txt", "Angoli_HWP_sep.txt"}]
    }];

    inputUnitariesEnt = Table[
        KroneckerProduct[
            localPolarizationUnitary[angles[[1, 1]], angles[[1, 2]]],
            localPolarizationUnitary[angles[[2, 1]], angles[[2, 2]]]
        ],
        {angles, anglesEntStates}
    ];
    
    inputUnitariesSep = Table[
        KroneckerProduct[
            localPolarizationUnitary[angles[[1, 1]], angles[[1, 2]]],
            localPolarizationUnitary[angles[[2, 1]], angles[[2, 2]]]
        ],
        {angles, anglesSepStates}
    ];
    
    If[OptionValue @ "referenceInputStates" =!= None,
        {referenceStateEnt, referenceStateSep} = {#["ent"], #["sep"]}& @ OptionValue @ "referenceInputStates",
        {referenceStateEnt, referenceStateSep} = {Normalize@{1, 0, 0, -1}, Normalize@{1, 1, -1, -1}}
    ];
    (* input states as kets *)
    inputStatesEnt = Dot[#, referenceStateEnt] & /@ inputUnitariesEnt // Chop;
    inputStatesSep = Dot[#, referenceStateSep] & /@ inputUnitariesSep // Chop;  
    
    countsVectorsEntAllReps = loadCountsData[data0209Dir, {"ccENT_rep_1.txt", "ccENT_rep_2.txt", "ccENT_rep_3.txt"}];
    countsVectorsSepAllReps = loadCountsData[data0209Dir, {"ccSEP_rep_1.txt", "ccSEP_rep_2.txt", "ccSEP_rep_3.txt"}];
    
    experimentalData @ <|
        "dataPath" -> data0209Dir,
        "dataFiles" -> FileNameTake /@ FileNames[All, data0209Dir],
        "statesAsKets" -> True, (* flag to mark whether states are stored as kets, rather than as dms *)
        "angles" -> <|
            "ent" -> anglesEntStates,
            "sep" -> anglesSepStates
        |>,
        "referenceStates" -> <|
            "ent" -> referenceStateEnt,
            "sep" -> referenceStateSep
        |>,
        "inputStates" -> <|
            "ent" -> inputStatesEnt,
            "sep" -> inputStatesSep
        |>,
        "counts" -> <|
            "ent" -> countsVectorsEntAllReps,
            "sep" -> countsVectorsSepAllReps
        |>
    |>
];


(* experimentalData /: Part[experimentalData[data_Association], spec__] := data[label]; *)

experimentalData[data_Association]["filter_labels", pattern_] := experimentalData @ Append[data, Table[
    keyToUpdate -> KeySelect[data @ keyToUpdate, StringMatchQ @ pattern],
    {keyToUpdate, {"angles", "referenceStates", "inputStates", "counts"}}
] ];

(* should work like <| "ent" -> {"ent1" -> spec1_, "ent2" -> spec2_} |> *)
splittingFunction[ass_Association, basekey_String -> newkeysSpecs_List, dimensionToSplit_Integer : 1] := Append[ass,
    Table[
        newkeySpecs[[1]] -> ass[basekey][[ Sequence @@ PadLeft[{newkeySpecs[[2]]}, dimensionToSplit, All] ]],
        {newkeySpecs, newkeysSpecs}
    ]
];

experimentalData[data_Association]["split_dataset", splittingRule_] := experimentalData @ Append[data, {
        "angles" -> splittingFunction[data @ "angles", splittingRule],
        "inputStates" -> splittingFunction[data @ "inputStates", splittingRule],
        "counts" -> splittingFunction[data @ "counts", splittingRule, 3] (* the 3 is b/c the THIRD dimension of counts corresponds to the states*)
    }
];
(* this is losing the rest of the labels, TO COMPLETE *)


experimentalData[data_Association]["merge_reps"] := If[
    Length @ Dimensions @ First @ Values @ data["counts"] == 3,
    (* only do something if counts matrices have 3 sizes, which indicates they haven't been flattened before *)
    experimentalData @ Append[data, {
        (* for each label associated to a dataset in "counts", sum over the first axis, which represents repetitions *)
        "counts" -> Association @@ Table[
            datakey -> Mean[ data["counts"][datakey] ],
            {datakey, Keys @ data["counts"]}
        ]
    }],
    (* otherwise do nothing; this is to prevent running "merge_reps" twice and fucking shit up *)
    experimentalData @ data
];
experimentalData[data_Association]["filter_reps", spec__] := experimentalData @ Append[data, {
    "counts" -> Association @@ Table[
        datakey -> data["counts"][datakey][[spec]],
        {datakey, Keys @ data["counts"]}
    ]
}];
experimentalData[data_Association][label_String] := data[label];


(* experimentalData[data_Association]["normalize_counts"] := experimentalData @ Append[data, {
    "counts" -> Association @@ Table[
        countsKey -> data["counts"][countsKey],
        {countsKey, Keys @ data["counts"]}
    ]
}]; *)


(* mergeLabels is to take labels corresponding to different datasetes, eg "ent" and "sep1", and put all data together (consistently between states and counts) *)
(* inputStates are stored num_states x state_dimension, while counts are stored num_outcomes x num_states, hence the different ways to merge them *)
mergeLabels[data_Association] := Append[data, {
    "train" -> Append[data["train"], {
        "inputStates" -> Flatten[#, 1]& @ Values @ data["train"]["inputStates"],
        "counts" -> Join[Sequence @@ #, 2] & @ Values @ data["train"]["counts"]
    }],
    "test" -> Append[data["test"], {
        "inputStates" -> Flatten[#, 1]& @ Values @ data["test"]["inputStates"],
        "counts" -> Join[Sequence @@ #, 2] & @ Values @ data["test"]["counts"]
    }]
}];
(* this function is used to train the QELM from the experimental data *)
(* repetitions are flattened here: only a single dataset for train and one for test *)
(* also labels are joined: only a single set of states for train and one for test *)
Options[trainQELMFromExperimentalData] = {
    "trainLabels" -> None,
    "testLabels" -> None,
    "targetObservables" -> None,
    "numSamples" -> None
};
trainQELMFromExperimentalData[data_experimentalData, opts : OptionsPattern[] ] := With[{
        (* sum counts for different repetitions (if not done before) *)
        flattenedData = data["merge_reps"][[1]]
    },
    (* we first extract train and test datasets, and afterwards with mergeLabels put them together. Finally we compute the expected observable expvals *)
    (* this should result in "inputStates" and "counts" now being straight up data matrices *)
    trainQELM[
        qelmData[mergeLabels @ Association[
            "train" -> Association @@ Table[
                label -> KeySelect[flattenedData[label], StringMatchQ @ (Alternatives @@ OptionValue["trainLabels"]) ],
                {label, {"inputStates", "counts"}}
            ],
            "test" -> Association @@ Table[
                label -> KeySelect[flattenedData[label], StringMatchQ @ (Alternatives @@ OptionValue["testLabels"]) ],
                {label, {"inputStates", "counts"}}
            ],
            "statesAsKets" -> flattenedData["statesAsKets"],
            "targetObservables" -> OptionValue["targetObservables"]
        ] ]["computeTrueExpvals"],
        "numSamples" -> OptionValue["numSamples"]
    ]
];






End[];  (* End `Private` *)

EndPackage[];

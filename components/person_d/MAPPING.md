# Person D vocabulary to diagnosticCriteria mapping

Generated from `mapping_acceptance.json` by `python -m components.person_d.mapping --write`; do not
edit by hand. `tests/test_mapping.py` keeps the two files, the mapping rule and the reference
assets in step.

| Item | Value |
|---|---|
| diagnosticCriteria version | `1.2.1` |
| Prompt asset (vocabulary) | `1.2.0`, SHA-256 `9474dcb890dc2f99c74c817242a048f4de59ec7ab425f76005a041ad8120ffa6` |
| Label map | `1.0.0`, SHA-256 `e1f59e667358fbca6f6bfe70cbfc140cd6c0a4ed8f6661920b5562a042f46e38` |
| Attributes | 24 |
| Vocabulary options | 123 |
| Criteria options | 138 |
| Unmapped vocabulary options | 0 |
| Vocabulary options mapped to several criteria options | 1 |
| Unreachable criteria options | 14 |

## Rules

1. A vocabulary label reaches criteria options through, in order: the label map attribute alias,
   the label map global alias, an exact match, a case-insensitive match (`LabelMap.resolve`).
2. An **unmapped** vocabulary label is dropped. When it was a model's only label, that model does
   not provide informative evidence for the attribute (`unmapped_label`).
3. A vocabulary label **mapped to several** criteria options counts once when those options carry
   the same condition in the query's criteria; when their conditions differ the label is ambiguous
   and dropped (`ambiguous_label`).
4. An **unreachable** criteria option is one no vocabulary label reaches, so it cannot itself
   produce a condition hit. Under the production `forward_all_informative` policy it does not
   block other observed evidence: PLIP or CONCH hitting any reachable `Must_True`, `Must_False`,
   `High_Possibly_True`, or `Low_Possibly_True` option forwards the ROI. Under the retained
   `strict_match` compatibility policy, unreachable Must options remain unverified and block
   selection according to `unverified_must_true`.

## Unreachable criteria options by category

| Category | Options | Meaning |
|---|---:|---|
| `unable_to_confirm` | 14 | Not a scored prediction target. The attribute has no N/A label, so this option cannot itself produce a condition hit; production forwarding may still use an informative prediction from either model. |

| Attribute | Criteria option | Category | Note |
|---|---|---|---|
| `Cellular_and_Nuclear.Cell_Pleomorphism` | Unable to confirm | `unable_to_confirm` |  |
| `Cellular_and_Nuclear.Cytological_Atypia` | Unable to confirm | `unable_to_confirm` |  |
| `Cellular_and_Nuclear.Chromatin` | Unable to confirm | `unable_to_confirm` |  |
| `Cellular_and_Nuclear.Nuclear_Size` | Unable to confirm | `unable_to_confirm` |  |
| `Cellular_and_Nuclear.Nuclear_Shape` | Unable to confirm | `unable_to_confirm` |  |
| `Cellular_and_Nuclear.Prominent_Nucleoli` | Unable to confirm | `unable_to_confirm` |  |
| `Cellular_and_Nuclear.Epithelial_Cell_Shape` | Unable to confirm | `unable_to_confirm` |  |
| `Cellular_and_Nuclear.Polarity` | Unable to confirm | `unable_to_confirm` |  |
| `Architecture.Architectural_Atypia` | Unable to confirm | `unable_to_confirm` |  |
| `Special_Features.Apocrine_Differentiation` | Unable to confirm | `unable_to_confirm` |  |
| `Special_Features.Apical_Snout` | Unable to confirm | `unable_to_confirm` |  |
| `Special_Features.Endoluminal_Secretion` | Unable to confirm | `unable_to_confirm` |  |
| `Special_Features.Necrosis` | Unable to confirm | `unable_to_confirm` |  |
| `Special_Features.Calcifications` | Unable to confirm | `unable_to_confirm` |  |

## Attribute-level mapping

Each row is one vocabulary label, the criteria option or options it reaches, and the rule that
applied.

### Cellular_and_Nuclear.Cell_Pleomorphism

Criteria options: Monomorphic, Mild, Moderate, Severe, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Monomorphic | Monomorphic | exact |
| Mild | Mild | exact |
| Moderate | Moderate | exact |
| Severe | Severe | exact |

Unreachable: Unable to confirm

### Cellular_and_Nuclear.Cytological_Atypia

Criteria options: Absent, Low-grade/Mild, Intermediate-grade/Moderate, High-grade/Severe, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Absent | Absent | exact |
| Low-grade/Mild | Low-grade/Mild | exact |
| Intermediate-grade/Moderate | Intermediate-grade/Moderate | exact |
| High-grade/Severe | High-grade/Severe | exact |

Unreachable: Unable to confirm

### Cellular_and_Nuclear.Chromatin

Criteria options: Regular, Vesicular, Coarsely clumped, Hyperchromatic, Salt-and-pepper, Smudged, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Regular | Regular | exact |
| Vesicular | Vesicular | exact |
| Coarsely clumped | Coarsely clumped | exact |
| Hyperchromatic | Hyperchromatic | exact |
| Salt-and-pepper | Salt-and-pepper | exact |
| Smudged | Smudged | exact |

Unreachable: Unable to confirm

### Cellular_and_Nuclear.Nuclear_Size

Criteria options: Uniform, Enlargement, Variable, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Uniform | Uniform | exact |
| Enlargement | Enlargement | exact |
| Variable | Variable | exact |

Unreachable: Unable to confirm

### Cellular_and_Nuclear.Nuclear_Shape

Criteria options: Uniform, Spindle, Irregular/Pleomorphic, Vesicular, Variable, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Uniform | Uniform | exact |
| Spindle | Spindle | exact |
| Irregular/Pleomorphic | Irregular/Pleomorphic | exact |
| Vesicular | Vesicular | exact |
| Variable | Variable | exact |

Unreachable: Unable to confirm

### Cellular_and_Nuclear.Prominent_Nucleoli

Criteria options: Absent, Present, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Absent | Absent | exact |
| Present | Present | exact |

Unreachable: Unable to confirm

### Cellular_and_Nuclear.Epithelial_Cell_Shape

Criteria options: Bilayered, Cuboidal, Columnar, Squamous, Polygonal, Plasmacytoid, Spindle, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Bilayered | Bilayered | exact |
| Cuboidal/Columnar | Cuboidal, Columnar | attribute alias |
| Squamous | Squamous | exact |
| Polygonal | Polygonal | exact |
| Plasmacytoid | Plasmacytoid | exact |
| Spindle | Spindle | exact |

`Cuboidal/Columnar`: Maps to both 'Cuboidal' and 'Columnar'. When the two options carry different conditions in a criteria the label is ambiguous and dropped; when they carry the same condition it counts once with that condition.

Unreachable: Unable to confirm

### Cellular_and_Nuclear.Polarity

Criteria options: Maintained, Partially lost, Completely lost, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Maintained | Maintained | exact |
| Partially lost | Partially lost | exact |
| Completely lost | Completely lost | exact |

Unreachable: Unable to confirm

### Cellular_and_Nuclear.Cytoplasmic_Features

Criteria options: Amphophilic to Pale Eosinophilic, Eosinophilic/Granular, Clear/Glycogenated, Vacuolated/Foamy/Lipid-rich, Signet-ring, Hobnail, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Amphophilic to Pale Eosinophilic | Amphophilic to Pale Eosinophilic | exact |
| Eosinophilic/Granular | Eosinophilic/Granular | exact |
| Clear/Glycogenated | Clear/Glycogenated | exact |
| Vacuolated/Foamy/Lipid-rich | Vacuolated/Foamy/Lipid-rich | exact |
| Signet-ring | Signet-ring | exact |
| Hobnail | Hobnail | exact |
| N/A | Unable to confirm | global alias |

### Cellular_and_Nuclear.Mitotic_Activity

Criteria options: Inconspicuous, Present/Brisk, Atypical figures present, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Inconspicuous | Inconspicuous | exact |
| Present/Brisk | Present/Brisk | exact |
| Atypical figures present | Atypical figures present | exact |
| N/A | Unable to confirm | global alias |

### Architecture.Pattern

Criteria options: Absent, No special pattern, Simple Tubular/Branching Structures, Solid, Comedo, Cribriform, Papillary, Trabecular/Cords, Micropapillary, Single-file/Linear cords, Fenestrated, Tubular/Glandular, Cystic/Microcystic, Syncytial, Nests/Morules, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| N/A | Unable to confirm | global alias |
| Absent, No special pattern | Absent, No special pattern | exact |
| Simple Tubular/Branching Structures | Simple Tubular/Branching Structures | exact |
| Comedo | Comedo | exact |
| Solid | Solid | exact |
| Cribriform | Cribriform | exact |
| Papillary | Papillary | exact |
| Trabecular/Cords | Trabecular/Cords | exact |
| Single file | Single-file/Linear cords | attribute alias |
| fenestrated | Fenestrated | case-insensitive |
| micropapillary | Micropapillary | case-insensitive |
| Tubular/Glandular | Tubular/Glandular | exact |
| Cystic/Microcystic | Cystic/Microcystic | exact |
| Syncytial | Syncytial | exact |
| Nests/Morules | Nests/Morules | exact |

### Architecture.Specific_Sub_Architectures

Criteria options: Absent, No special pattern, Rigid bridges, Arches, Bulbous, Thin/Stretched bridge, Bars, Microrosettes, One to several layers, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| N/A | Unable to confirm | global alias |
| Absent, No special pattern | Absent, No special pattern | exact |
| Rigid bridges | Rigid bridges | exact |
| Arches | Arches | exact |
| Bulbous | Bulbous | exact |
| thin/stretched bridge | Thin/Stretched bridge | case-insensitive |
| bars | Bars | case-insensitive |
| microrosettes | Microrosettes | case-insensitive |
| One to several layers | One to several layers | exact |

### Architecture.Architectural_Atypia

Criteria options: Absent, Mild/Moderate, Severe, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Absent | Absent | exact |
| Mild/Moderate | Mild/Moderate | exact |
| Severe | Severe | exact |

Unreachable: Unable to confirm

### Architecture.Lumina_Shape

Criteria options: Round/Oval, Slit-like/Compressed, Irregular/Angulated, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Round/Oval | Round/Oval | exact |
| Slit-like/Compressed | Slit-like/Compressed | exact |
| Irregular/Angulated | Irregular/Angulated | exact |
| N/A | Unable to confirm | global alias |

### Architecture.Tumour_Border

Criteria options: Absent, No tumour, Well-circumscribed/Pushing, Irregular/Infiltrative, Pseudoinfiltrative, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Absent, No tumour | Absent, No tumour | exact |
| Well-circumscribed/Pushing | Well-circumscribed/Pushing | exact |
| Irregular/Infiltrative | Irregular/Infiltrative | exact |
| Pseudoinfiltrative | Pseudoinfiltrative | exact |
| N/A | Unable to confirm | global alias |

### Architecture.Myoepithelial_Cell_Layer

Criteria options: Visibly Intact/Prominent, Attenuated/Indiscernible, Visibly Absent, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Visibly Intact/Prominent | Visibly Intact/Prominent | exact |
| Attenuated/Indiscernible | Attenuated/Indiscernible | exact |
| Visibly Absent | Visibly Absent | exact |
| N/A | Unable to confirm | global alias |

### Special_Features.Apocrine_Differentiation

Criteria options: Absent, Present, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Absent | Absent | exact |
| Present | Present | exact |

Unreachable: Unable to confirm

### Special_Features.Apical_Snout

Criteria options: Absent, Present, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Absent | Absent | exact |
| Present | Present | exact |

Unreachable: Unable to confirm

### Special_Features.Endoluminal_Secretion

Criteria options: Absent, Physiological, Eosinophilic/Proteinaceous, Mucinous, Cellular debris, Mucin pools/Extracellular mucin, Present, Not Otherwise Specified, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Absent | Absent | exact |
| Physiological | Physiological | exact |
| Present | Present, Not Otherwise Specified | attribute alias |
| Eosinophilic/Proteinaceous | Eosinophilic/Proteinaceous | exact |
| Mucinous | Mucinous | exact |
| Cellular debris | Cellular debris | exact |
| Mucin pools/Extracellular mucin | Mucin pools/Extracellular mucin | exact |

Unreachable: Unable to confirm

### Special_Features.Necrosis

Criteria options: Absent, Focal/Punctate, Comedonecrosis, Present, Not Otherwise Specified, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Absent | Absent | exact |
| Focal/Punctate | Focal/Punctate | exact |
| Comedonecrosis | Comedonecrosis | exact |
| Present, Not Otherwise Specified | Present, Not Otherwise Specified | exact |

Unreachable: Unable to confirm

### Special_Features.Calcifications

Criteria options: Absent, Coarse/Popcorn, Psammomatous, Microcalcifications, Linear, Branching, Fine pleomorphic/Clustered, Amorphous, Present, Not Otherwise Specified, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Absent | Absent | exact |
| Coarse/Popcorn | Coarse/Popcorn | exact |
| Psammomatous | Psammomatous | exact |
| Fine pleomorphic/Clustered | Fine pleomorphic/Clustered | exact |
| microcalcifications | Microcalcifications | case-insensitive |
| amorphous | Amorphous | case-insensitive |
| linear | Linear | case-insensitive |
| branching | Branching | case-insensitive |
| Present, Not Otherwise Specified | Present, Not Otherwise Specified | exact |

Unreachable: Unable to confirm

### Special_Features.Squamous_Sebaceous_Differentiation

Criteria options: Absent, Squamous, Sebaceous, Present, Not Otherwise Specified, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Absent | Absent | exact |
| Squamous | Squamous | exact |
| Sebaceous | Sebaceous | exact |
| Present, Not Otherwise Specified | Present, Not Otherwise Specified | exact |
| N/A | Unable to confirm | global alias |

### Special_Features.Stromal_Characteristics

Criteria options: Fibrofatty/Loose Intralobular, Desmoplastic/Fibrotic, Fibroelastotic, Myxoid/Chondromyxoid, Oedematous, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Fibrofatty/Loose Intralobular | Fibrofatty/Loose Intralobular | exact |
| Desmoplastic/Fibrotic | Desmoplastic/Fibrotic | exact |
| Fibroelastotic | Fibroelastotic | exact |
| Myxoid/Chondromyxoid | Myxoid/Chondromyxoid | exact |
| Oedematous | Oedematous | exact |
| N/A | Unable to confirm | global alias |

### Special_Features.Tumour_Infiltrating_Lymphocytes

Criteria options: Absent/Sparse, Mild/Moderate, Prominent/Rich, Unable to confirm

| Vocabulary label | Criteria option | Rule |
|---|---|---|
| Absent/Sparse | Absent/Sparse | exact |
| Mild/Moderate | Mild/Moderate | exact |
| Prominent/Rich | Prominent/Rich | exact |
| N/A | Unable to confirm | global alias |

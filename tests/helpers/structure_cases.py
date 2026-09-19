"""Name compatibility datasets explicitly so adding a file cannot grow test work silently.

The quick reader contracts use 1DAW. The broader panel exercises deposited files
across crystal systems and file encodings in the slow tier. Pair-based pipeline
checks use trigonal 2DQ6 and body-centred tetragonal 3A5V in addition to their
separate 1DAW checks.
"""

MODEL_CODES = (
    "1DAW",  # C-centred monoclinic; quick reference structure.
    "2DQ6",  # Trigonal.
    "3A5V",  # Body-centred tetragonal.
    "3E98",  # Monoclinic screw axis.
    "3GR5",  # Hexagonal screw axis.
    "3K7M",  # Cubic.
    "3VRJ",  # Additional monoclinic deposition.
    "4BX9",  # Tetragonal screw axis.
    "5BOV",  # Triclinic P1.
    "6G9X",  # Orthorhombic.
)

MTZ_CODES = MODEL_CODES + ("1AK5", "1BYW", "1VER", "6JZA", "6SXW", "6VHI")
SF_CIF_CODES = MODEL_CODES + ("7L84",)
EXTENDED_PAIR_CODES = ("2DQ6", "3A5V")
MODEL_CIF_FILES = tuple(f"{code}.cif" for code in MODEL_CODES) + (
    "test_ihm_ensemble.cif",
)

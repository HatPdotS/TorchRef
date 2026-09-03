"""Regression tests for the refinement output header.

The writer used to carry an input PDB's header through verbatim and then append
its own ``REMARK 3``, so a refined file asserted two different refinements at
once -- the previous program's R-factors first, ours several hundred lines
later. The selection was also inverted: TITLE, AUTHOR and REMARK were kept
(including the superseded statistics) while SEQRES, SSBOND, LINK, CISPEP,
EXPDTA, COMPND, DBREF and FORMUL were dropped, i.e. the chemistry refinement
does not invalidate.

What is asserted here is the resulting contract: exactly one refinement block
and it is ours; records that describe the crystal survive; records that describe
a superseded refinement do not; and the chain of programs applied to the model
is preserved through mmCIF's ``_software`` loop rather than by hoarding the
previous program's output.
"""

import pandas as pd
import pytest

from torchref.io import cif, pdb
from torchref.io.metadata import RefinementMetadata

# 3GR5 was refined with REFMAC 5.1.24 and carries a full deposition header:
# 420 lines including REMARK 2/3/500, JRNL, AUTHOR, SEQRES, SSBOND and SITE.
INPUT_PDB = "tests/files/pdb/3GR5.pdb"

#: PDB record order, abridged to the records this writer can emit. The format
#: mandates this sequence; TITLE used to be written *after* REMARK 900.
RECORD_ORDER = [
    "TITLE", "COMPND", "SOURCE", "KEYWDS", "EXPDTA", "MDLTYP", "AUTHOR",
    "REMARK", "DBREF", "DBREF1", "DBREF2", "SEQADV", "SEQRES", "MODRES",
    "HET", "HETNAM", "HETSYN", "FORMUL", "SSBOND", "LINK", "CISPEP", "SITE",
]


def _atom_df():
    """Two atoms whose coordinates exercise trailing-zero formatting.

    ``18.3`` and ``31.51`` are the values that exposed the writer using
    ``str()`` semantics on a rounded float instead of ``%8.3f``.
    """
    df = pd.DataFrame(
        {
            "ATOM": ["ATOM", "HETATM"],
            "serial": [1, 2],
            "name": ["CA", "O"],
            "altloc": ["", ""],
            "resname": ["LEU", "HOH"],
            "chainid": ["A", "A"],
            "resseq": [1, 2],
            "icode": ["", ""],
            "x": [-7.223, -9.22],
            "y": [29.982, 31.51],
            "z": [18.3, 20.364],
            "occupancy": [1.0, 1.0],
            "tempfactor": [95.4, 30.0],
            "element": ["C", "O"],
            "charge": [0, 0],
            "anisou_flag": [False, False],
            "u11": [0.0, 0.0],
            "u22": [0.0, 0.0],
            "u33": [0.0, 0.0],
            "u12": [0.0, 0.0],
            "u13": [0.0, 0.0],
            "u23": [0.0, 0.0],
        }
    )
    df.attrs["cell"] = [90.645, 90.645, 133.422, 90.0, 90.0, 120.0]
    df.attrs["spacegroup"] = "P 65 2 2"
    return df


def _refined_metadata():
    """Input header plus this refinement's statistics, as the writer builds it."""
    meta = RefinementMetadata.from_pdb_file(INPUT_PDB)
    ours = RefinementMetadata(
        program_version="0.7.0",
        target_function="ML",
        optimizer="1 MACROCYCLE, SEPARATE, ISOTROPIC ADP, SCALE TARGET NLL",
        r_work=0.2083,
        r_free=0.2471,
        percent_free=9.9,
        n_reflections_all=20942,
        n_reflections_test=2063,
        resolution_high=2.05,
        resolution_low=18.69,
        starting_model=INPUT_PDB,
        rfree_selection="MTZReader FreeR",
    )
    return meta.merge(ours)


def _header_lines():
    return _refined_metadata().render_pdb_header().splitlines()


def _remark_number(line):
    try:
        return int(line[7:10])
    except ValueError:
        return None


# ====================================================================== #
#  One refinement block, and it is ours
# ====================================================================== #


@pytest.mark.unit
def test_exactly_one_refinement_block():
    header = "\n".join(_header_lines())
    assert header.count("REMARK   3 REFINEMENT.") == 1
    assert "TORCHREF" in header


@pytest.mark.unit
@pytest.mark.parametrize("program", ["REFMAC", "PHENIX", "BUSTER", "CNS"])
def test_no_foreign_program_is_credited(program):
    """The input was refined by REFMAC; the output must not say so anywhere."""
    assert program not in "\n".join(_header_lines())


@pytest.mark.unit
def test_superseded_statistics_are_dropped():
    """REMARK 2, 3 and 500 describe a resolution and a model we replaced."""
    numbers = {_remark_number(line) for line in _header_lines()
               if line.startswith("REMARK")}
    assert 2 not in numbers            # resolution: regenerated
    assert 500 not in numbers          # geometry outliers of old coordinates
    assert 3 in numbers                # ours, and the only one


@pytest.mark.unit
def test_input_remark_3_is_not_carried_through():
    """The specific inversion this module exists to prevent."""
    meta = RefinementMetadata.from_pdb_file(INPUT_PDB)
    assert not any(_remark_number(r) == 3 for r in meta.passthrough_pdb_remarks)
    # ... while the input genuinely has one, so the assertion has teeth.
    with open(INPUT_PDB) as handle:
        assert any(line.startswith("REMARK   3") for line in handle)


# ====================================================================== #
#  Attribution
# ====================================================================== #


@pytest.mark.unit
def test_authors_and_journal_are_not_inherited():
    """Both credit the deposition, not this refinement."""
    meta = RefinementMetadata.from_pdb_file(INPUT_PDB)
    assert meta.authors == []
    lines = _header_lines()
    assert not any(line.startswith(("AUTHOR", "JRNL")) for line in lines)


@pytest.mark.unit
def test_explicitly_set_authors_are_written():
    """Not inheriting authors must not stop --authors from working."""
    meta = _refined_metadata()
    meta.authors = ["A.Person", "B.Other"]
    lines = meta.render_pdb_header().splitlines()
    author = [line for line in lines if line.startswith("AUTHOR")]
    assert author and "A.Person" in author[0]


# ====================================================================== #
#  Records that survive
# ====================================================================== #


@pytest.mark.unit
@pytest.mark.parametrize(
    "record", ["COMPND", "SOURCE", "KEYWDS", "EXPDTA", "DBREF", "SEQADV",
               "SEQRES", "HETNAM", "FORMUL", "SSBOND", "SITE"]
)
def test_structural_records_are_carried_through(record):
    """Refinement moves atoms; it does not change the sequence or chemistry."""
    with open(INPUT_PDB) as handle:
        expected = sum(1 for line in handle if line.startswith(record))
    assert expected > 0, f"{record} absent from the fixture"
    written = sum(1 for line in _header_lines() if line.startswith(record))
    assert written == expected


@pytest.mark.unit
def test_secondary_structure_is_not_emitted():
    """Nothing here computes HELIX/SHEET, so carrying them can only mislead."""
    lines = _header_lines()
    assert not any(line.startswith(("HELIX", "SHEET")) for line in lines)


@pytest.mark.unit
def test_records_are_in_mandated_order():
    seen = []
    for line in _header_lines():
        record = line[:6].strip()
        if record and (not seen or seen[-1] != record):
            seen.append(record)
    assert all(r in RECORD_ORDER for r in seen), [
        r for r in seen if r not in RECORD_ORDER
    ]
    positions = [RECORD_ORDER.index(r) for r in seen]
    assert positions == sorted(positions), seen


@pytest.mark.unit
def test_remarks_ascend_with_ours_at_three():
    numbers = [
        _remark_number(line) for line in _header_lines()
        if line.startswith("REMARK")
    ]
    assert numbers == sorted(numbers)
    assert 3 in numbers


# ====================================================================== #
#  Free text is the author's, never generated
# ====================================================================== #


@pytest.mark.unit
def test_no_remarks_section_without_author_text():
    assert "OTHER REFINEMENT REMARKS" not in "\n".join(_header_lines())


@pytest.mark.unit
def test_author_text_is_written_when_supplied():
    meta = _refined_metadata()
    meta.output_remarks = "Re-refined for the benchmark.\n\nSecond paragraph."
    header = meta.render_pdb_header()
    assert "REMARK   3  OTHER REFINEMENT REMARKS:" in header
    assert "Re-refined for the benchmark." in header
    assert "Second paragraph." in header


@pytest.mark.unit
def test_free_set_provenance_is_reported():
    """R-free from a different test set is not comparable; say which one."""
    header = "\n".join(_header_lines())
    assert "FREE R VALUE TEST SET SELECTION" in header
    assert "MTZReader FreeR" in header


@pytest.mark.unit
def test_remark_3_colons_line_up_within_each_block():
    """Colons used to be ragged, because alignment relied on caller padding.

    Two columns exist by design, which is what REFMAC does too: a narrow one
    for the identification lines (``PROGRAM     :``) and a wide one for the
    statistics (``RESOLUTION RANGE HIGH (ANGSTROMS) :``). Each must be
    internally consistent, including on wrapped continuation lines.
    """
    columns = [
        line.index(" : ") for line in _header_lines()
        if line.startswith("REMARK   3   ") and " : " in line
    ]
    assert set(columns) == {24, 46}, sorted(set(columns))


@pytest.mark.unit
def test_header_lines_fit_the_format():
    assert [line for line in _header_lines() if len(line) > 80] == []


@pytest.mark.unit
def test_long_identification_values_wrap_rather_than_overflow():
    """Naming cycles, mode, ADP model and scale target overruns column 80."""
    meta = _refined_metadata()
    meta.optimizer = (
        "12 MACROCYCLES, EVERYTHING, FIELD_ANISO ADP, SCALE TARGET ML_NOALPHA, "
        "RIGID BODY 5 ITERATIONS"
    )
    lines = meta.render_pdb_header().splitlines()
    assert [line for line in lines if len(line) > 80] == []
    optimizer = [line for line in lines if "OPTIMIZER" in line]
    assert len(optimizer) == 1                       # one head line ...
    head = lines.index(optimizer[0])
    assert lines[head + 1].startswith("REMARK   3               : ")
    # ... and the value survives the wrap intact.
    joined = " ".join(
        line.split(":", 1)[1].strip()
        for line in lines[head:head + 3]
        if " : " in line
    )
    assert "RIGID BODY 5 ITERATIONS" in joined


# ====================================================================== #
#  Coordinates
# ====================================================================== #


@pytest.mark.unit
def test_numeric_columns_keep_their_decimals(tmp_path):
    """``18.3`` must be written ``  18.300`` and ``95.4`` as `` 95.40``.

    Both fields were formatted as ``str()`` of a rounded float rather than with
    an explicit precision, so trailing zeros vanished.
    """
    out = tmp_path / "out.pdb"
    pdb.write(_atom_df(), str(out))
    atoms = [
        line for line in out.read_text().splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]
    assert atoms
    for line in atoms:
        assert len(line) == 80, repr(line)
        # x, y, z: %8.3f
        for start in (30, 38, 46):
            field = line[start:start + 8]
            assert len(field.split(".")[1]) == 3, repr(field)
        # occupancy and B: %6.2f
        for start in (54, 60):
            field = line[start:start + 6]
            assert len(field.split(".")[1]) == 2, repr(field)
    assert " 95.40" in atoms[0]


# ====================================================================== #
#  The software chain -- mmCIF's only room for prior work
# ====================================================================== #


@pytest.mark.unit
def test_software_is_a_loop_with_an_ordinal():
    cats = _refined_metadata().render_cif_categories()
    software = cats["_software"]
    assert software["_software.pdbx_ordinal"] == ["1"]
    assert software["_software.name"] == ["TORCHREF"]


@pytest.mark.unit
def test_software_ordinal_increments_across_refinements(tmp_path):
    """Refining a refined file must append to the chain, not replace it."""
    first = tmp_path / "first.cif"
    cif.write_model(_atom_df(), str(first), metadata=_refined_metadata())

    # Read it back the way a second refinement would, then write again.
    carried = RefinementMetadata.from_cif_file(str(first))
    assert len(carried.software_chain) == 1

    second_meta = carried.merge(
        RefinementMetadata(program_version="0.7.0", optimizer="2 MACROCYCLES")
    )
    second = tmp_path / "second.cif"
    cif.write_model(_atom_df(), str(second), metadata=second_meta)

    chain = RefinementMetadata.from_cif_file(str(second)).software_chain
    assert [e.get("pdbx_ordinal") for e in chain] == ["1", "2"]


@pytest.mark.unit
def test_previous_refinement_description_survives_the_round_trip(tmp_path):
    """The chain is worthless if it cannot say what each program did."""
    out = tmp_path / "one.cif"
    meta = _refined_metadata()
    meta.optimizer = "7 MACROCYCLES, SEPARATE, ISOTROPIC ADP"
    cif.write_model(_atom_df(), str(out), metadata=meta)

    chain = RefinementMetadata.from_cif_file(str(out)).software_chain
    assert "7 MACROCYCLES" in chain[0]["description"]


@pytest.mark.unit
def test_loop_values_with_spaces_survive_a_round_trip(tmp_path):
    """An unquoted loop cell silently splits into extra columns when re-read."""
    out = tmp_path / "quoted.cif"
    meta = _refined_metadata()
    meta.optimizer = "MANY WORDS, WITH COMMAS, AND SPACES"
    cif.write_model(_atom_df(), str(out), metadata=meta)

    chain = RefinementMetadata.from_cif_file(str(out)).software_chain
    assert len(chain) == 1
    assert chain[0]["description"].endswith("AND SPACES")


@pytest.mark.unit
def test_input_refine_statistics_are_not_carried_into_cif(tmp_path):
    """The mmCIF form of the duplicated REMARK 3."""
    source = tmp_path / "prior.cif"
    prior = RefinementMetadata(program="REFMAC", r_work=0.213, r_free=0.251)
    cif.write_model(_atom_df(), str(source), metadata=prior)

    carried = RefinementMetadata.from_cif_file(str(source))
    assert "_refine" not in carried.passthrough_cif_categories
    assert carried.r_work is None
    # The prior program is remembered as a link in the chain, not as statistics.
    assert [e["name"] for e in carried.software_chain] == ["REFMAC"]


@pytest.mark.unit
def test_starting_model_is_recorded_as_an_accession(tmp_path):
    cats = _refined_metadata().render_cif_categories()
    initial = cats["_pdbx_initial_refinement_model"]
    assert initial["_pdbx_initial_refinement_model.accession_code"] == "3GR5"
    assert initial["_pdbx_initial_refinement_model.type"] == "experimental model"
    assert cats["_refine"]["_refine.pdbx_starting_model"] == INPUT_PDB


@pytest.mark.unit
def test_unaccessionable_starting_model_is_named_not_guessed():
    meta = _refined_metadata()
    meta.starting_model = "/tmp/my_working_model_v3.pdb"
    initial = meta.render_cif_categories()["_pdbx_initial_refinement_model"]
    assert "accession_code" not in " ".join(initial)
    assert initial["_pdbx_initial_refinement_model.details"] == (
        "my_working_model_v3.pdb"
    )


# ====================================================================== #
#  Annotating a file is not refining it
# ====================================================================== #


@pytest.mark.unit
def test_annotation_keeps_the_existing_refinement():
    """``add-metadata`` adds a title; it does not re-refine.

    Nothing supersedes the input's REMARK 3 or AUTHOR records in that case, so
    dropping them would discard statistics and credit that are still accurate.
    """
    meta = RefinementMetadata.from_pdb_file(
        INPUT_PDB, supersede_refinement=False
    )
    numbers = {_remark_number(r) for r in meta.passthrough_pdb_remarks}
    assert {2, 3, 500} <= numbers
    assert meta.authors  # REFMAC-era depositors keep their credit


@pytest.mark.unit
def test_refinement_output_supersedes_by_default():
    """The default is the refinement case: the old block goes."""
    meta = RefinementMetadata.from_pdb_file(INPUT_PDB)
    numbers = {_remark_number(r) for r in meta.passthrough_pdb_remarks}
    assert not ({2, 3, 500} & numbers)
    assert meta.authors == []

"""
Data Router - Automatic file type detection and reader selection.

This module provides a smart router that automatically detects the file type
and returns the appropriate reader (CIF or legacy format).

Supported file types:
- Structure factors: .mtz, .cif (with reflection data)
- Structure models: .pdb, .cif (with atom_site data)
- Restraints: .cif (with restraint dictionaries)

Usage:
    from torchref.io import DataRouter

    # Automatic detection
    router = DataRouter("data.mtz")
    reader = router.get_reader()
    data_type = router.data_type  # 'reflections', 'structure', or 'restraints'

    # Or use the factory method
    reader, data_type = DataRouter.route("data.cif")
"""

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import gemmi

from torchref.io import cif, mtz, pdb


class DataRouterError(Exception):
    """Exception raised when file type cannot be determined or is unsupported."""

    pass


class DataRouter:
    """
    Automatic file type detection and reader selection.

    This class examines a file and automatically selects the appropriate
    reader based on file extension and content.

    Parameters
    ----------
    filepath : str or Path
        Path to the file to read.
    verbose : int, default 1
        Verbosity level (0=quiet, 1=normal, 2+=debug).

    Attributes
    ----------
    filepath : Path
        Path to the file to read.
    verbose : int
        Verbosity level for logging.
    data_type : str or None
        Type of data detected ('reflections', 'structure', 'restraints', or None).
    file_format : str or None
        File format detected ('mtz', 'pdb', 'cif', or None).
    reader : object or None
        The appropriate reader instance (or None if not yet created).

    Examples
    --------
    >>> router = DataRouter("structure.cif")
    >>> reader = router.get_reader()
    >>> print(router.data_type)  # 'structure'
    """

    # Supported file extensions
    MTZ_EXTENSIONS = {".mtz"}
    PDB_EXTENSIONS = {".pdb", ".ent"}
    CIF_EXTENSIONS = {".cif", ".mmcif"}

    def __init__(self, filepath: Union[str, Path], verbose: int = 1):
        """
        Initialize the DataRouter.

        Parameters
        ----------
        filepath : str or Path
            Path to the data file.
        verbose : int, default 1
            Verbosity level (0=quiet, 1=normal, 2+=debug).
        """
        self.filepath = Path(filepath)
        self.verbose = verbose
        self.data_type: Optional[str] = None
        self.file_format: Optional[str] = None
        self.reader: Optional[Any] = None

        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        # Automatically detect file type
        self._detect_file_type()

    def _detect_file_type(self) -> None:
        """
        Detect the file type based on extension and content.

        Sets self.data_type and self.file_format.
        """
        extension = self.filepath.suffix.lower()

        if self.verbose > 1:
            print(f"DataRouter: Analyzing {self.filepath.name}")
            print(f"  Extension: {extension}")

        # MTZ files - always structure factors
        if extension in self.MTZ_EXTENSIONS:
            self.file_format = "mtz"
            self.data_type = "reflections"
            if self.verbose > 1:
                print("  Detected: MTZ structure factor file")
            return

        # PDB files - always structure models
        if extension in self.PDB_EXTENSIONS:
            self.file_format = "pdb"
            self.data_type = "structure"
            if self.verbose > 1:
                print("  Detected: PDB structure file")
            return

        # CIF files - need to examine content
        if extension in self.CIF_EXTENSIONS:
            self.file_format = "cif"
            self._detect_cif_type()
            return

        # Unknown extension
        raise DataRouterError(
            f"Unsupported file extension: {extension}\n"
            f"Supported extensions: .mtz, .pdb, .cif, .mmcif"
        )

    def _detect_cif_type(self) -> None:
        """
        Detect the type of CIF file by examining its content.

        Sets self.data_type to 'reflections', 'structure', or 'restraints'.
        """
        try:
            # Use gemmi to quickly parse CIF
            doc = gemmi.cif.read_file(str(self.filepath))

            # Check all blocks in the CIF file
            has_refln = False
            has_atom_site = False
            has_restraints = False

            for block in doc:
                # Check for reflection data
                if (
                    block.find_loop("_refln.index_h")
                    or block.find_loop("_refln_index_h")
                    or block.find_value("_refln.index_h")
                ):
                    has_refln = True

                # Check for structure data
                if (
                    block.find_loop("_atom_site.group_PDB")
                    or block.find_loop("_atom_site_group_PDB")
                    or block.find_value("_atom_site.group_PDB")
                ):
                    has_atom_site = True

                # Check for restraint dictionary data
                if (
                    block.find_loop("_chem_comp_atom.atom_id")
                    or block.find_loop("_chem_comp_bond.atom_id_1")
                    or block.find_value("_chem_comp.id")
                ):
                    has_restraints = True

            # Prioritize based on what we found
            if has_refln:
                self.data_type = "reflections"
                if self.verbose > 1:
                    print("  Detected: CIF reflection data (structure factors)")
            elif has_atom_site:
                self.data_type = "structure"
                if self.verbose > 1:
                    print("  Detected: CIF structure model (coordinates)")
            elif has_restraints:
                self.data_type = "restraints"
                if self.verbose > 1:
                    print("  Detected: CIF restraint dictionary")
            else:
                raise DataRouterError(
                    f"CIF file does not contain recognizable data: {self.filepath}\n"
                    f"Expected: reflection data (_refln), structure data (_atom_site), "
                    f"or restraint data (_chem_comp)"
                )

        except Exception as e:
            raise DataRouterError(
                f"Failed to parse CIF file: {self.filepath}\n" f"Error: {str(e)}"
            )

    def get_reader(self) -> Any:
        """
        Get the appropriate reader for this file.

        Returns
        -------
        object
            Reader instance (ReflectionCIFReader, ModelCIFReader, RestraintCIFReader,
            MTZ, or PDB depending on file type).

        Raises
        ------
        DataRouterError
            If file type is not supported or cannot be determined.
        """
        if self.reader is not None:
            return self.reader

        # Create the appropriate reader based on data type and format
        if self.data_type == "reflections":
            if self.file_format == "mtz":
                self.reader = mtz.MTZReader(verbose=self.verbose)
                self.reader.read(str(self.filepath))
            elif self.file_format == "cif":
                self.reader = cif.ReflectionCIFReader(
                    str(self.filepath), verbose=self.verbose
                )
            else:
                raise DataRouterError(
                    f"Unknown format for reflections: {self.file_format}"
                )

        elif self.data_type == "structure":
            if self.file_format == "pdb":
                self.reader = pdb.PDBReader(verbose=self.verbose)
                self.reader.read(str(self.filepath))
            elif self.file_format == "cif":
                self.reader = cif.ModelCIFReader(
                    str(self.filepath), verbose=self.verbose
                )
            else:
                raise DataRouterError(
                    f"Unknown format for structure: {self.file_format}"
                )

        elif self.data_type == "restraints":
            if self.file_format == "cif":
                self.reader = cif.RestraintCIFReader(
                    str(self.filepath), verbose=self.verbose
                )
            else:
                raise DataRouterError("Restraints only supported in CIF format")

        else:
            raise DataRouterError(f"Unknown data type: {self.data_type}")

        if self.verbose > 0:
            print(f"Created {self.reader.__class__.__name__} for {self.filepath.name}")

        return self.reader

    def get_data(self) -> Tuple[Any, ...]:
        """
        Get the data from the file using the appropriate reader.

        This is a convenience method that calls get_reader() and then
        invokes the reader to get the data.

        Returns
        -------
        tuple
            For reflections: (data_dict, cell, spacegroup)
            For structure: (dataframe, residues, spacegroup)
            For restraints: Restraint data (format depends on reader)
        """
        reader = self.get_reader()
        return reader()

    @classmethod
    def route(cls, filepath: Union[str, Path], verbose: int = 1) -> Tuple[Any, str]:
        """
        Factory method to quickly route a file to the appropriate reader.

        Parameters
        ----------
        filepath : str or Path
            Path to the data file.
        verbose : int, default 1
            Verbosity level.

        Returns
        -------
        tuple
            Tuple of (reader, data_type) where:
            - reader: The appropriate reader instance
            - data_type: String indicating the type ('reflections', 'structure', 'restraints')

        Examples
        --------
        >>> reader, data_type = DataRouter.route("7JI4-sf.cif")
        >>> if data_type == 'reflections':
        ...     data_dict, cell, spacegroup = reader()
        """
        router = cls(filepath, verbose=verbose)
        reader = router.get_reader()
        return reader, router.data_type

    def __repr__(self) -> str:
        """String representation of the DataRouter."""
        return (
            f"DataRouter(filepath={self.filepath.name}, "
            f"data_type={self.data_type}, "
            f"file_format={self.file_format})"
        )

    def __str__(self) -> str:
        """User-friendly string representation."""
        if self.data_type and self.file_format:
            return (
                f"DataRouter: {self.filepath.name}\n"
                f"  Type: {self.data_type}\n"
                f"  Format: {self.file_format}\n"
                f"  Reader: {self.reader.__class__.__name__ if self.reader else 'Not created'}"
            )
        return f"DataRouter: {self.filepath.name} (not analyzed)"

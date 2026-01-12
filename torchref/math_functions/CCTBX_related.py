from iotbx import pdb


def calculate_scattering_factor_cctbx(pdb_file,hkls = None,d_min=2.0):
    """
    """
    pdb_input = pdb.input(file_name=pdb_file)
    xray_structure = pdb_input.xray_structure_simple()
    f_calc = xray_structure.structure_factors(d_min=d_min).f_calc()
    idx = np.array(f_calc.indices())
    f_calc = np.array(f_calc.data())
    if hkls is not None:
        hkls = np.array(hkls,dtype=int)
        hkls = set([tuple(hkl) for hkl in hkls])
        f_calc_new = []
        idx_new = []
        for hkl,val in zip(idx,f_calc):
            if tuple(hkl) in hkls:
                f_calc_new.append(val)
                idx_new.append(hkl)
        f_calc = np.array(f_calc_new)
        idx = np.array(idx_new)
    return f_calc, idx
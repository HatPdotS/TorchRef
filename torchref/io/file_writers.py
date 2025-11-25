'''
A small collection of writer functions
'''
import numpy as np
import torch

def write_pdb_line(f,row):
    f.write(f'{row[0]:<{7}}{int(row[1]):<{6}}{str(row[2]):<{3}}{str(row[3]):>{1}}{str(row[4]):>{3}}{str(row[5]):>{2}}{int(row[6]):>{4}}{str(row[7]):>{4}}{round(row[8],3):>{8}}{round(row[9],3):>{8}}{round(row[10],3):>{8}}{row[11]:>{6}}{round(row[12],2):>{6}}{str(row[13]):>{12}}{str(row[14]):>{2}}\n')


def write_file(df,fname,template= None):
    '''
    
    Write a DataFrame to a PDB file
    Args:
        df (pd.DataFrame): DataFrame containing atom data
        fname (str): Output PDB filename
        template (str): Optional PDB template file to copy header from

    '''
    with open(fname,'w') as n:
        try: 
            cell = df.attrs['cell']
            spacegroup = df.attrs['spacegroup']
            cell_abc = cell[:3]
            cell_angles = cell[3:]
            z = df.attrs['z']
            try:
                strz = str(int(z))
            except:
                strz = ''
            line = 'CRYST1' + ''.join([f'{i:>9.3f}' for i in cell_abc]) + ''.join([f'{i:>7.2f}' for i in cell_angles]) + ' ' + f'{spacegroup:<14}' + strz + '\n'
            n.write(line)
        except:
            print('No cell information found, writing without cell and spacegroup')
            pass
        if template is not None:
            with open(template) as t:
                for line in t:
                    if 'REMARK' not in line and 'ATOM' in line:
                        break
                    n.write(line) 
        for i,row in df.iterrows():
            ATOM,serial,name,altloc,resname,chainid,resseq,icode,x,y,z,occupancy,tempfactor,element,charge = row[['ATOM', 'serial', 'name', 'altloc', 'resname', 'chainid', 'resseq','icode', 'x', 'y', 'z', 'occupancy', 'tempfactor', 'element', 'charge']]
            if charge > 0:
                charge = '+' + str(charge)
            elif charge == 0:
                charge = ''
            else:
                charge = str(charge)
            if len(name) > 3:
                name = name[-3:]
            if len(name) < 3:
                name = name + ' '*(3-len(name))
            if chainid is None or str(chainid) == 'nan':
                chainid = ''
            try:
                s = f'{str(ATOM):<6}{int(serial):>5}{str(name):>5}{str(altloc):>1}{str(resname):>3}{str(chainid):>2}{int(resseq):>4}{str(icode):>4}{round(x,3):>8}{round(y,3):>8}{round(z,3):>8}{round(occupancy,3):>6.2f}{round(tempfactor,2):>6}{str(element):>12}{charge:>2}\n'
                n.write(s)
            except:
                print('row',i,'failed')
                print(row)
                pass
            if row['anisou_flag']:
                u11,u22,u33,u12,u13,u23 = row[['u11','u22','u33','u12','u13','u23']]
                s = f'ANISOU{int(serial):>5}{str(name):>5}{str(altloc):>1}{str(resname):>3}{str(chainid):>2}{int(resseq):>4}  {int(u11*1e4):>{7}}{int(u22*1e4):>{7}}{int(u33*1e4):>{7}}{int(u12*1e4):>{7}}{int(u13*1e4):>{7}}{int(u23*1e4):>{7}}      {str(element):>{2}}{str(charge):>2}\n'
                n.write(s)
        n.write('END')


def write_ccp4(data, cell, fname):
    '''
    Write a 3D numpy array or torch tensor to a CCP4 file using gemmi
    Args:
        data (np.ndarray or torch.Tensor): 3D array of map data
        cell (list or torch.Tensor): Unit cell parameters [a, b, c, alpha, beta, gamma]
        fname (str): Output CCP4 filename
    '''
    import gemmi
    if isinstance(data, torch.Tensor):
        np_map = data.detach().cpu().numpy().astype(np.float32)
    else:
        np_map = data.astype(np.float32)
    if isinstance(cell, torch.Tensor):
        cell = cell.detach().cpu().numpy().tolist()
    elif isinstance(cell, np.ndarray):
        cell = cell.tolist()
    elif isinstance(cell, list):
        cell = cell
    else:
        raise ValueError("cell must be a list, numpy array, or torch tensor")

    map_ccp = gemmi.Ccp4Map()
    map_ccp.grid = gemmi.FloatGrid(np_map, gemmi.UnitCell(*cell), gemmi.SpaceGroup('P1'))
    map_ccp.setup(0.0)
    map_ccp.update_ccp4_header()
    map_ccp.write_ccp4_map(fname)
    return 1


def write_mtz(df, cell, spacegroup, fname):
    '''
    Write a DataFrame to an MTZ file using reciprocalspaceship
    
    Args:
        df (pd.DataFrame): DataFrame containing reflection data
        cell (list): Unit cell parameters [a, b, c, alpha, beta, gamma]
        spacegroup (str or gemmi.SpaceGroup): Spacegroup symbol, number, or gemmi SpaceGroup object
        fname (str): Output MTZ filename
    '''

    import reciprocalspaceship as rs
    import gemmi
    mtz_rs = rs.DataSet(df)
    if 'H' in mtz_rs.columns and 'K' in mtz_rs.columns and 'L' in mtz_rs.columns:
        mtz_rs = mtz_rs.set_index('H', 'K', 'L')
    if torch.is_tensor(cell):
        cell = cell.detach().cpu().numpy().tolist()
    
    # Handle different spacegroup input types
    if isinstance(spacegroup, gemmi.SpaceGroup):
        # Already a SpaceGroup object
        pass
    elif isinstance(spacegroup, str):
        # Check if it's a string representation of a gemmi.SpaceGroup
        if spacegroup.startswith('<gemmi.SpaceGroup'):
            # Extract the spacegroup name from the string representation
            # Format: '<gemmi.SpaceGroup("P 1")>'
            import re
            match = re.search(r'SpaceGroup\("([^"]+)"\)', spacegroup)
            if match:
                spacegroup = gemmi.SpaceGroup(match.group(1))
            else:
                raise ValueError(f"Could not parse spacegroup string: {spacegroup}")
        else:
            # Normal string spacegroup name
            spacegroup = gemmi.SpaceGroup(spacegroup)
    else:
        raise ValueError(f"Spacegroup must be str or gemmi.SpaceGroup, got {type(spacegroup)}")
    
    structure_factor_cols = ['Fobs', '2FOFCWT', 'FOFCWT','F-model']
    intensity_cols = ['I-obs']
    sigma_cols = ['SIGF-obs','SIGI-obs']
    phase_cols = ['PH2FOFCWT','PHFOFCWT','PH-model']
    flags = ['R-free-flags']

    # Assign correct MTZ data types for each column type
    for col in structure_factor_cols:
        if col in mtz_rs.columns:
            mtz_rs[col] = mtz_rs[col].astype('F')
    
    for col in intensity_cols:
        if col in mtz_rs.columns:
            mtz_rs[col] = mtz_rs[col].astype('J')
    
    for col in sigma_cols:
        if col in mtz_rs.columns:
            mtz_rs[col] = mtz_rs[col].astype('Q')
    
    for col in phase_cols:
        if col in mtz_rs.columns:
            mtz_rs[col] = mtz_rs[col].astype('P')
    
    for col in flags:
        if col in mtz_rs.columns:
            mtz_rs[col] = mtz_rs[col].astype('I')
            
    mtz_rs = mtz_rs.infer_mtz_dtypes()
    mtz_rs.cell = gemmi.UnitCell(*cell)
    mtz_rs.spacegroup = spacegroup
    mtz_rs.write_mtz(fname)
    return 1

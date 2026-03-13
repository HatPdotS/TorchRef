from glob import glob
import os



file_path = glob('/das/work/p17/p17490/Peter/Library/torchref/scientific_testing/data/*')
outfiles = []

for file in file_path:
    # Don't use restraints from the restraints/ folder - they are PDB structure files
    # without proper restraint parameters. The code will use the monomer library instead.
    # restraints = glob(file + '/restraints/*.cif')
    try:
        pdb = glob(file + '/*_shaken.pdb')[0]
        cif = glob(file + '/*.mtz')[0]
    except:
        print("Missing files in ", file)
        continue
    restraints = glob(file + '/restraints/*.cif')

    cif_agg = 'None'
    outdir = file + '/refine_final_w_rama'
    error = outdir + '/error.log'
    log = outdir + '/out.log'
    # if os.path.exists(os.path.join(outdir, 'refined.pdb')):
    #     continue

    cmd = '''sbatch -p day -c 8 -t 1-00:00:00 -o {} -e {} torchref.refine -f {} -s {} -o {} -n 10'''.format(log, error, cif, pdb, outdir)
    print(cmd)
    os.system(cmd)
    outfiles.append(outdir)

print(' '.join(outfiles))
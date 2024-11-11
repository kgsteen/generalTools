import sys
import numpy as np

evIt = 50				# only write out every evIt iterations to the movie file
rpt = [2, 2, 1]				# how many supercell repeats

"""
Script to analyze an XDATCAR file for NPT or NVT ensembles and generate a movie file in XYZ format.

Usage:
    python mkMov_npt.py <path_to_XDATCAR/XDAT_filename> <ensemble_type>

Arguments:
    /path/to/XDATCAR_filename : str
        The file path (path/to/) and filename (XDATCAR_filename) of the XDATCAR file to be analyzed.
    ensemble_type : str
        The ensemble type for the analysis. Must be either 'npt' or 'nvt' (no quotes on command line).

Example:
    python mkMov_npt.py /path/to/XDATCAR npt

This script reads an XDATCAR file, builds a supercell according to the specified ensemble (NPT or NVT),
and generates a movie file named 'newMov.xyz' with atomic positions in XYZ format.

As is, newMov.xyz will be written to whichever directory the script is run from.

Notes:
    - Ensure that you provide both arguments; otherwise, the script will exit with an error message.
    - The second argument, ensemble_type, must be exactly 'npt' or 'nvt' (case-insensitive).

Dependencies:
    - Python 3.12 or compatible version
    - numpy library

Author: Krista G. Steenbergen
Date: 1 Nov 2024 
"""

###################################################################
def read_input_file(filename):

    # Reads the input file and counts frames.
    with open(filename, 'r') as fr:
        x = fr.readlines()
    nfrm = sum('Direct' in s for s in x)
    return x, nfrm


###################################################################
def get_atom_data(x):

    # Extracts atom types, numbers, and initializes arrays.
    natAr = np.genfromtxt(x[6:7], comments='\n', dtype=np.int32)
    N = np.sum(natAr)
    strAr = x[5].split()
    return natAr, N, strAr


###################################################################
def build_superNPT(natAr, N, rpt, x, cnt):

    # Builds the supercell with repeated atomic coordinates.
    a = np.genfromtxt(x[cnt:cnt + 3], comments='\n')
    atAr = np.genfromtxt(x[cnt + 6:cnt + 6 + N], comments='\n') % 1.0
    nA = np.zeros((N * rpt[0] * rpt[1] * rpt[2], 3))

    cntNa = 0
    for nSp, nTyp in enumerate(natAr):
        curSt = sum(natAr[:nSp])
        for xr in range(rpt[0]):
            for yr in range(rpt[1]):
                for zr in range(rpt[2]):
                    addM = [xr, yr, zr]
                    nA[cntNa:cntNa + nTyp] = atAr[curSt:curSt + nTyp] + addM
                    cntNa += nTyp

    return np.dot(nA, a)  # Return Cartesian coordinates


###################################################################
def build_superNVT(natAr, N, rpt, x, cnt, a):

    # Builds the supercell with repeated atomic coordinates.
    atAr = np.genfromtxt(x[cnt + 6:cnt + 6 + N], comments='\n') % 1.0
    nA = np.zeros((N * rpt[0] * rpt[1] * rpt[2], 3))

    cntNa = 0
    for nSp, nTyp in enumerate(natAr):
        curSt = sum(natAr[:nSp])
        for xr in range(rpt[0]):
            for yr in range(rpt[1]):
                for zr in range(rpt[2]):
                    addM = [xr, yr, zr]
                    nA[cntNa:cntNa + nTyp] = atAr[curSt:curSt + nTyp] + addM
                    cntNa += nTyp

    return np.dot(nA, a)  # Return Cartesian coordinates


###################################################################
def write_xyzFile(filename, traj, supNatAr, strAr):

    # Writes trajectory data to an XYZ file format.
    namArr = []							# atomType array for the supercell
    for nSp, atom_type in enumerate(strAr):
        for _ in range(supNatAr[nSp]):
            namArr.append(atom_type)

    with open(filename, 'w') as fw:
        for i, frame in enumerate(traj):			# writes out xyz for each atom in each frame
            fw.write(f'{len(namArr)}\nFrame {i + 1}\n')
            for j, coord in enumerate(frame):
                fw.write(f"{namArr[j]}   {'  '.join(map(str, coord))}\n")



###################################################################
###################################################################

if __name__ == "__main__":

    # Check that there are exactly two command-line arguments
    if len(sys.argv) != 3:
        print("Error: Please provide exactly two arguments: the path to the XDATCAR file and the ensemble type (npt or nvt).")
        sys.exit(1)

    # Unpack the command-line arguments
    input_file = sys.argv[1]
    ensemble_type = sys.argv[2].lower()

    # Validate the ensemble type argument
    if ensemble_type not in ['npt', 'nvt']:
        print("Error: Ensemble type must be 'npt' or 'nvt'.")
        sys.exit(1)

    # Validate ensemble type
    if ensemble_type.lower() not in ['npt', 'nvt']:
        print("Error: Ensemble type must be 'npt' or 'nvt'.")
        sys.exit(1)  # Exit with a non-zero code to indicate an error

    x, nfrm = read_input_file(input_file)

    # Get atom data
    natAr, N, strAr = get_atom_data(x)

    # Initialize trajectory array
    traj = []
    cnt = 2

    # Process frames
    if (ensemble_type == 'npt'):
        for i in range(int(nfrm / evIt)):
            traj.append(build_superNPT(natAr, N, rpt, x, cnt))
            cnt += (N + 8) * evIt
    else:
        a = np.genfromtxt(x[2:5], comments='\n')
        for i in range(int(nfrm / evIt)):
            traj.append(build_superNVT(natAr, N, rpt, x, cnt, a))
            cnt += (N + 1) * evIt

    # Write to XYZ file
    write_xyzFile('newMov.xyz', traj, natAr * rpt[0] * rpt[1] * rpt[2], strAr)





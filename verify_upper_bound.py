from utilities import *

constructions = {
    '4AP_q5_c2': hex_state_to_coloring(r'0x1e3f98ba3cdec23569d4ef0d30e8909', 5, 3, shift=(0,1,2)),
    '3AP_q3_c3': np.array([[[0, 0, 1], [0, 0, 1], [1, 1, 2]], [[1, 1, 2], [1, 1, 2], [2, 2, 0]], [[2, 2, 0], [2, 2, 0], [0, 0, 1]]]),
    '5AP_q5_c2': hex_state_to_coloring(r'0x1e3f98ba3cdec23569d4ef0d30e8909', 5, 3, shift=(0,1,2)),
    'Schur_q2_c3': np.array([[0, 1], [1, 2]]),
    'Schur_q3_c3': np.array([[0, 0, 2], [0, 1, 2], [1, 1, 2]]),
}

color_dict = {0: '🔴', 1: '🔵', 2: '🟡'}

def array_to_string(array):
    if array.ndim == 1:
        return ''.join([color_dict[color] for color in array])
    
    elif array.ndim == 2:
        output = ''
        for row in array:
            output += array_to_string(row) + "\n"
        return output
    
    elif array.ndim == 3:
        temp = []
        for matrix in array:
            temp.append( array_to_string(matrix) + "\n")
        output = ""
        for i, x in enumerate(temp[0].split('\n')):
            if len(x) > 0: output += '  '.join([t.split('\n')[i] for t in temp])+'\n'
        return output

    else:
        raise ValueError

if __name__=='__main__':
    for problem in ['Schur_q2_c3', 'Schur_q3_c3', '3AP_q3_c3', '4AP_q5_c2', '5AP_q5_c2']:

        q = int(problem.split('_')[1].replace('q', ''))
        c = int(problem.split('_')[2].replace('c', ''))
        structure = problem.split('_')[0]
        coloring = constructions[problem]

        print("\n----------------------------------------")
        
        print(f"\nThe coloring for {'Schur triple' if structure=='Schur' else structure[0]+'-AP'} in {c}-colorings of F{q} is")

        print('\n' + array_to_string(coloring))

        val, _ = count_solutions(coloring, structure, q)

        print(f"The blowup value through Lemma 3.4 is {val} ({float(val):.6f}).")

        for color in range(c):
            modified_coloring = deepcopy(coloring)
            modified_coloring[(0,)*coloring.ndim] = color
            modified_val, _ = count_solutions(modified_coloring, structure, q)
            if modified_val != val:
                print(f"Proposition 3.5 does not apply.")
                break
        else:
            num_morphisms = QQ( (q**coloring.ndim)**2 ) # only valid for APs / unfixed morphisms
            rep_val = QQ((num_morphisms*val - 1) / (num_morphisms - 1))
            print(f"Proposition 3.5 can be applied, giving an upper bound of {rep_val} ({float(rep_val):.6f}).")

    print("\n----------------------------------------\n")
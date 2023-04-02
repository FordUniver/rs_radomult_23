from utilities import *

types_and_flags = {
    'Schur_q2_c3_N4': None,
    '3AP_q3_c3_N2': {
        '1': ['[1 1 1]', '[1 2 2]', '[1 1 2]', '[1 2 3]', '[1 1 3]'],
        '2': ['[2 2 2]', '[2 3 3]', '[2 2 3]', '[2 1 3]', '[2 1 2]'],
        '3': ['[3 3 3]', '[3 1 1]', '[3 1 3]', '[3 1 2]', '[3 2 3]'],
    },
    '4AP_q5_c2_N2': {
        '1': ['[1 1 1 1 1]', '[1 1 1 1 2]', '[1 1 2 2 2]', '[1 2 2 2 2]'],
        '2': ['[2 2 2 2 2]', '[2 1 2 2 2]', '[2 1 1 1 2]', '[2 1 1 1 1]'],
    }
}

coefficients = {
    'Schur_q2_c3_N4': None,
    '3AP_q3_c3_N2': [
        [
            sqrt(QQ(26/27))            * vector([QQ(1), -QQ(99/182),  QQ(75/208),    -QQ(11/28),         -QQ(3/26)]),
            sqrt(QQ(1685/1911))        * vector([QQ(0),  QQ(1),      -QQ(231/26960),  QQ(1703/6740),     -QQ(1869/3370)]),
            sqrt(QQ(71779/431360))     * vector([QQ(0),  QQ(0),       QQ(1),         -QQ(358196/502453), -QQ(412904/502453)]),
            sqrt(QQ(5431408/10551513)) * vector([QQ(0),  QQ(0),       QQ(0),          QQ(1),             -QQ(1/4)]),
        ],
    ] * 3,
    '4AP_q5_c2_N2': [
        [
            sqrt(QQ(9/10))   * vector([QQ(1),  QQ(5/27), -QQ(5/27), -QQ(10/27)]),
            sqrt(QQ(61/162)) * vector([QQ(0), -QQ(1/2),   QQ(1/2),   QQ(1)])
        ],
    ] * 2,
}

lambdas = {
    'Schur_q2_c3_N4': 0.04125873873847112,
    '3AP_q3_c3_N2':   QQ(1/27),
    '4AP_q5_c2_N2':   QQ(1/10)
}

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--problem', '-p',
        choices = ['Schur_q2_c3_N4', '3AP_q3_c3_N2', '4AP_q5_c2_N2'],
        required = True,
        help    = 'Specify which lower bound you want to verify.'
    )
    args = parser.parse_args()

    # Parse problem

    q = int(args.problem.split('_')[1].replace('q', ''))
    c = int(args.problem.split('_')[2].replace('c', ''))
    n = int(args.problem.split('_')[3].replace('N', ''))
    structure = args.problem.split('_')[0]
    invariant = not (structure == 'Schur')
    
    # Data will be stored in files

    colorings_file = f'{args.problem}_colorings.npy'
    values_file    = f'{args.problem}_values.npy'
    densities_file = f'{args.problem}_densities.npy'

    with mp.Pool() as pool:

        # Determine or load all relevant colorings

        tm = time.perf_counter()

        if os.path.isfile(colorings_file):
            print(f"\nLoading coloring isomorphism classes from {colorings_file} ...")
            with open(colorings_file, 'rb') as f: colorings = pickle.load(f)

        else:
            print(f"\nComputing coloring isomorphism classes ...")
            colorings = get_isomorphism_classes(q, n, c, invariant=invariant, pool=pool, verbose=True)
            with open(colorings_file, 'wb') as f: pickle.dump(colorings, f)
            
        print(f"Done in {time.perf_counter()-tm:.1f}s. Found {len(colorings)} colorings.")

        # Determine or load values of colorings

        tm = time.perf_counter()

        if os.path.isfile(values_file):
            print(f"\nLoading values of colorings from {values_file} ...")
            with open(values_file, 'rb') as f: values = pickle.load(f)

        else:
            print(f"\nComputing values of colorings ...")
            arguments = [(coloring, structure, q) for coloring in colorings]
            values = apply_pool(count_solutions, arguments, pool=pool, verbose=True)
            values = [v[1] for v in values]
            with open(values_file, 'wb') as f: pickle.dump(values, f)
            
        print(f"Done in {time.perf_counter()-tm:.1f}s.")


        # Determine or load flag pair densities

        tm = time.perf_counter()

        if os.path.isfile(densities_file):
            print(f"\nLoading pair densities from {densities_file} ...")
            with open(densities_file, 'rb') as f: type_to_flags, pair_densities = pickle.load(f)

        else:
            print(f"\nComputing pair densities ...")
            Fqn = list(VectorSpace(GF(q), n))
            PGqn = []
            for x in Fqn[1:]:
                for l in GF(q):
                    if arreq_in_list(l*x, PGqn): break
                else:
                    PGqn.append(x)

            bases = []
            for basis in itertools.combinations(PGqn, n):
                if Matrix(basis).rank() == n: bases.append(basis)
                
            sizes = [(s, floor((n+s)/2)) for s in range(n-1)]
            if invariant: sizes.append((None, n-1))
            
            pair_densities = [{} for _ in colorings]
            type_to_flags = {}

            for s, t in sizes:
                arguments = [(coloring, q, s, t, n, c, bases, invariant) for coloring in colorings]
                output = apply_pool(get_pair_densities, arguments, pool=pool, verbose=True, desc=f'type size {s} and flag size {t}')
                
                for idx, densities in enumerate(output):
                    check = 0
                    for key, val in densities.items():
                        for k, v in val.items():
                            check += v*(2 if k[0]!=k[1] else 1)
                    assert check==QQ(1), check

                    for key, val in densities.items():
                        for k in val.items(): assert k not in pair_densities[idx].get(k, {}).keys()
                        pair_densities[idx][key] = {**pair_densities[idx].get(key, {}), **val}
                        type_to_flags[key] = set(list(type_to_flags.get(key, [])) + [k[0] for k in val.keys()] + [k[1] for k in val.keys()])

            with open(densities_file, 'wb') as f: pickle.dump((type_to_flags, pair_densities), f)

        # print(pair_densities[0])
        # print(pair_densities[10])
        # print(pair_densities[100])

        # exit()

        used_types_and_flags = types_and_flags[args.problem] or {k: sorted(v) for k,v in type_to_flags.items()}

        print(f"Done in {time.perf_counter()-tm:.1f}s.")

        # Verify positive semidefiniteness
        # (actually follows by decomposition but still good to check)

        tm = time.perf_counter()
        print(f"\nVerifying positive semidefinitess ...")
        if coefficients[args.problem] is None:
            with open(f'{args.problem}_sol.npy', 'rb') as f: xmatrices = pickle.load(f)
            for ftype, X in tqdm(xmatrices.items()):
                assert X.is_positive_semidefinite()
        else:
            xmatrices = {}
            for (ftype, flags), coeffs in zip(used_types_and_flags.items(), tqdm(coefficients[args.problem])):
                Q = Matrix( [vec for vec in coeffs] + [vector([0,]*len(flags)) for _ in range(len(flags)-len(coeffs))] )
                xmatrices[ftype] = Q.T * Q
                assert xmatrices[ftype].is_positive_semidefinite()
        print(f"Done in {time.perf_counter()-tm:.1f}s.")

        # Verify values

        tm = time.perf_counter()
        print(f"\nVerifying values ...")
        collected_vals = []
        for coloring, pds, val in zip(tqdm(colorings), pair_densities, values):
            temp = val
            for ftype, flags in used_types_and_flags.items():
                if ftype not in pds: continue
                X = xmatrices[ftype]
                M = Matrix(QQ, np.zeros((len(flags), len(flags))))
                for (f1, f2), d in pds[ftype].items():
                    if f1 in flags and f2 in flags:
                        M[flags.index(f1), flags.index(f2)] = d
                        M[flags.index(f2), flags.index(f1)] = d
                temp -= sum(sum(M.elementwise_product(X)))
            collected_vals.append(temp)
        assert min(collected_vals) >= lambdas[args.problem]
        print(f"Done in {time.perf_counter()-tm:.1f}s.")


        # If no assert is triggered, verification passed

        print("\nVerification passed! 🎉\n")
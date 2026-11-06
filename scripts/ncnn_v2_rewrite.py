"""Fixed v2/8-token/4-head graph rewrite; no weights, dimensions or attention math changed."""


def split_attention_gemm(path):
    lines = path.read_text().splitlines()
    if lines[0].strip() != '7767517':
        raise ValueError('not an ncnn text graph')
    nodes = [line.split() for line in lines[2:] if line.strip()]
    positions = [i for i, row in enumerate(nodes) if row[0] == 'MatMul']
    if (len(positions) != 2 or positions[1] <= positions[0]
            or any(nodes[i][2:4] != ['2', '1'] or len(nodes[i]) != 7 for i in positions)
            or not any(row[0] == 'Softmax' and row[4] == nodes[positions[0]][6] for row in nodes[positions[0]+1:positions[1]])):
        raise ValueError('expected the fixed v2 pair of untransposed attention MatMul nodes')
    result, changes = [], []
    extra_nodes, extra_blobs = 0, 0
    for i, row in enumerate(nodes):
        if i not in positions:
            result.append(' '.join(row)); continue
        number = positions.index(i); prefix = f'nova_headgemm_{number}'
        if prefix in '\n'.join(lines):
            raise ValueError('head rewrite identifier collision')
        # QK: [4,8,32]@[4,32,8]. AV: [4,8,8]@[4,8,32].
        m, k, n = (8, 32, 8) if number == 0 else (8, 8, 32)
        a_names = [prefix+f'_a{head}' for head in range(4)]
        b_names = [prefix+f'_b{head}' for head in range(4)]
        result.append(f'Slice {prefix}_slice_a 1 4 {row[4]} '+ ' '.join(a_names)+' -23300=4,1,1,1,1 1=0')
        result.append(f'Slice {prefix}_slice_b 1 4 {row[5]} '+ ' '.join(b_names)+' -23300=4,1,1,1,1 1=0')
        outputs = []
        for head in range(4):
            base = prefix+f'_{head}'; a, b, c, out = [base+'_'+suffix for suffix in ('a2','b2','c2','c3')]
            result.append(f'Reshape {base}_reshape_a 1 1 {a_names[head]} {a} 0={k} 1={m}')
            result.append(f'Reshape {base}_reshape_b 1 1 {b_names[head]} {b} 0={n} 1={k}')
            # Both matrices dynamic, no C/bias, no external weights.
            result.append(f'Gemm {base}_gemm 2 1 {a} {b} {c} 2=0 3=0 4=0 5=0 6=1 10=-1')
            result.append(f'Reshape {base}_reshape_c 1 1 {c} {out} 0={n} 1={m} 2=1')
            outputs.append(out)
        result.append(f'Concat {prefix}_concat 4 1 '+' '.join(outputs)+f' {row[6]} 0=0')
        extra_nodes += 18; extra_blobs += 24
        changes.append({'original_layer':row[1], 'heads':4, 'matrix_dimensions':[m,k,n],
                        'replacement':'channel slices + 2D Gemm + channel concat', 'weights_modified':False})
    counts = [int(value) for value in lines[1].split()]
    path.write_text('7767517\n'+f'{counts[0]+extra_nodes} {counts[1]+extra_blobs}'+'\n'+'\n'.join(result)+'\n')
    return changes

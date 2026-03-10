import csv, math, statistics, pathlib, sys

mesh_path = pathlib.Path('.tmp/mesh.csv')
out_path = pathlib.Path('.tmp/out.csv')

if not mesh_path.exists() or not out_path.exists():
    print('Missing .tmp/mesh.csv or .tmp/out.csv in workspace root')
    sys.exit(1)

def read_rows(path):
    with open(path, newline='') as f:
        rdr = list(csv.reader(f))
    return rdr

mesh = read_rows(mesh_path)
out = read_rows(out_path)

# parse positions: mesh cols: input.position.x,y,z   out cols: _Position.x,y,z
def parse_rows(rows, xi, yi, zi):
    vals=[]
    for r in rows[1:]:
        if len(r) <= max(xi,yi,zi):
            continue
        try:
            x=float(r[xi]); y=float(r[yi]); z=float(r[zi])
            vals.append((x,y,z))
        except:
            continue
    return vals

mesh_pos = parse_rows(mesh, 2, 3, 4)
out_pos  = parse_rows(out, 2, 3, 4)

n = min(len(mesh_pos), len(out_pos))
if n == 0:
    print('No valid rows parsed')
    sys.exit(1)
mesh_pos = mesh_pos[:n]
out_pos = out_pos[:n]

def pearson(u,v):
    mu_u = statistics.mean(u); mu_v = statistics.mean(v)
    num = sum((ui-mu_u)*(vi-mu_v) for ui,vi in zip(u,v))
    den = math.sqrt(sum((ui-mu_u)**2 for ui in u)*sum((vi-mu_v)**2 for vi in v))
    return num/den if den>0 else 0.0

def corr(a,b):
    ax=[x for (x,_,_) in a]; ay=[y for (_,y,_) in a]; az=[z for (_,_,z) in a]
    bx=[x for (x,_,_) in b]; by=[y for (_,y,_) in b]; bz=[z for (_,_,z) in b]
    return {
        'X->X': pearson(ax,bx),'X->Y': pearson(ax,by),'X->Z': pearson(ax,bz),
        'Y->X': pearson(ay,bx),'Y->Y': pearson(ay,by),'Y->Z': pearson(ay,bz),
        'Z->X': pearson(az,bx),'Z->Y': pearson(az,by),'Z->Z': pearson(az,bz),
    }

# Candidate mappings
mapped_A = [(p[0], p[2], -p[1]) for p in mesh_pos]  # (x, z, -y)
mapped_B = [(p[0], -p[2], p[1]) for p in mesh_pos] # (x, -z, y)

print(f'Parsed {n} vertices')
print('\nMapping A (x, z, -y) correlations:')
ca = corr(mapped_A, out_pos)
for k in sorted(ca.keys()):
    print(f'{k}: {ca[k]:.4f}')

print('\nMapping B (x, -z, y) correlations:')
cb = corr(mapped_B, out_pos)
for k in sorted(cb.keys()):
    print(f'{k}: {cb[k]:.4f}')

# Determine best matching mapping by summing absolute of expected diagonal (X->X + Y->Y + Z->Z)
score_A = abs(ca['X->X']) + abs(ca['Y->Y']) + abs(ca['Z->Z'])
score_B = abs(cb['X->X']) + abs(cb['Y->Y']) + abs(cb['Z->Z'])
print('\nScores (sum abs diagonal correlations):')
print(f'A: {score_A:.4f}  B: {score_B:.4f}')

best = 'A' if score_A > score_B else 'B'
print(f'Best mapping: {best}')

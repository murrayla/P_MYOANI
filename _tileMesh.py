"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _tileMesh.py
        Simplified mesh generation using Gmsh's Box
"""

# ∆ Raw
import gmsh

# ∆ Constants
DIM = 3
ORDER = 2
BASE_TAGS = {
    'point': 1000,
    'line': 2000,
    'surf_x': 3000,
    'surf_y': 3010,
    'surf_z': 3020,
    'vol': 4000
}
CUBE = {"x": 1000, "y": 1000, "z": 100}
PXLS = {"x": 11, "y": 11, "z": 50}
G = {d: CUBE[d] * PXLS[d] for d in "xyz"}
ELEMS = {
        2: "3-node-triangle", 3: "4-node-quadrangle", 4: "4-node-tetrahedron",  
        5: "8-node-hexahedron",  9: "6-node-second-order-triangle", 
        10: "9-node-second-order-quadrangle",  11: "10-node-second-order-tetrahedron \n \n"
    }  
ENUM = 11

# ∆ Node data
def node_data(msh_size, file):
    print("\t" + "~> Load .msh file and output nodes")

    # ∆ Setup
    n_path = f"_msh/em_{msh_size}.nodes"
    e_path = f"_msh/em_{msh_size}.ele"

    # ∆ Set files
    with open(file, 'r') as msh_file, open(n_path, 'w') as n_file, open(e_path, 'w') as e_file:

        # ∆ Add checks and store lists
        in_n, in_e = False, False
        n_lines, e_lines = [], []

        # ∆ Iterate msh file
        for line in msh_file:

            # ∆ Analyse line content
            line = line.strip()
            if line.startswith('$'):
                sec = line.strip('$')
                if sec == 'Nodes':
                    in_n = True
                elif sec == 'EndNodes':
                    in_n = False
                elif sec == 'Elements':
                    in_e = True
                elif sec == 'EndElements':
                    in_e = False
                continue
            if in_n:
                n_lines.append(line)
            elif in_e:
                e_lines.append(line)

        # ∆ Node blocks
        i = 1
        n_file.write(n_lines[0] + "\n") 
        while i < len(n_lines):

            # ∆ Write data
            h = n_lines[i].split()
            s = int(h[3])

            # ∆ Iterate length to write
            for j in range(s):
                idx = i + 1 + j
                coords = n_lines[idx + s].split()
                node_id = n_lines[idx]
                n_file.write(f"{node_id}\t{coords[0]}\t{coords[1]}\t{coords[2]}\n")
            i += 1 + 2 * s

        # ∆ Element blocks
        i = 1
        e_file.write(e_lines[0] + "\n")  
        while i < len(e_lines):

            # ∆ Write data
            h = e_lines[i].split()
            ele_type = int(h[2])
            num_elem = int(h[3])

            # ∆ Iterate length to write
            for j in range(num_elem):
                elem_data = e_lines[i + 1 + j].split()
                e_file.write('\t'.join(elem_data) + "\n")
            i += 1 + num_elem

# ∆ .msh generation
def msh_(mesh_size):

    # ∆ Initialise
    print("\t" + f"+= Generate mesh with size: {mesh_size}")
    gmsh.initialize()
    gmsh.model.add(f"em_{mesh_size}")

    # ∆ Create a box
    box = gmsh.model.occ.addBox(0, 0, 0, G["x"], G["y"], G["z"])
    gmsh.model.occ.synchronize()

    # ∆ Label physical groups
    tol = 1e-6
    seen_tgs = set()
    for dim in range(DIM + 1):

        # ∆ Iterate dimensions
        for ent_dim, ent_tag in gmsh.model.occ.getEntities(dim):

            # ∆ Determine centre of mass 
            com = gmsh.model.occ.get_center_of_mass(ent_dim, ent_tag)

            # ∆ Assign physical groups
            if dim == 3:
                tag = BASE_TAGS["vol"]
                name = "vol"
            elif dim == 2:
                if abs(com[0]) < tol:
                    axis_str, pos = "x", 0
                elif abs(com[0] - G["x"]) < tol:
                    axis_str, pos = "x", 1
                elif abs(com[1]) < tol:
                    axis_str, pos = "y", 0
                elif abs(com[1] - G["y"]) < tol:
                    axis_str, pos = "y", 1
                elif abs(com[2]) < tol:
                    axis_str, pos = "z", 0
                elif abs(com[2] - G["z"]) < tol:
                    axis_str, pos = "z", 1
                tag = BASE_TAGS[f'surf_{axis_str}'] + pos
                name = f"sf_{axis_str}_{pos}"
            elif dim == 1:
                tag = BASE_TAGS['line'] + ent_tag
                name = f"ln_{ent_tag}"
            else:
                tag = BASE_TAGS['point'] + ent_tag
                name = f"pt_{ent_tag}"

            # ∆ Check tag logic
            if tag not in seen_tgs:
                gmsh.model.add_physical_group(dim, [ent_tag], tag)
                gmsh.model.set_physical_name(dim, tag, name)
                seen_tgs.add(tag)

    # ∆ Set mesh size
    for p in gmsh.model.getEntities(dim=0):
        gmsh.model.mesh.setSize([p], mesh_size)

    # ∆ Mesh generation
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.model.mesh.generate(DIM)
    # gmsh.model.mesh.refine()
    # gmsh.model.mesh.refine()
    # gmsh.model.mesh.refine()
    gmsh.model.mesh.setOrder(order=ORDER)

    # ∆ Save to file
    file = f"_msh/em_{mesh_size}.msh"
    gmsh.write(file)
    gmsh.finalize()

    return file

# ∆ Main
def main(msh_size):

    # ∆ Generate mesh
    file = msh_(msh_size)

    # ∆ Generate .mesh and .node
    node_data(msh_size, file)

# ∆ Initialise
if __name__ == '__main__':
    main(200)

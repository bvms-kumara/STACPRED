import graphviz

def create_thesis_flowchart():
    # 'strict' prevents duplicate edges
    dot = graphviz.Digraph('Thesis_Filtering_Pipeline', strict=False)

    # --- 1. Global Graph Attributes (High Visibility) ---
    dot.attr(rankdir='TB')       # Top-to-Bottom flow
    dot.attr(dpi='300')          # High Resolution for Thesis Printing
    dot.attr(splines='ortho')    # Orthogonal (straight) lines
    dot.attr(nodesep='0.8')      # More horizontal space between boxes
    dot.attr(ranksep='0.6')      # More vertical space between levels
    
    # --- 2. Node Attributes (Big & Clear) ---
    dot.attr('node', 
             shape='rect', 
             fontname='Times-Roman', 
             fontsize='14',           # INCREASED FONT SIZE
             margin='0.2,0.1',        # internal padding so text doesn't touch borders
             style='rounded,filled', 
             fillcolor='white',       
             color='black',           
             penwidth='2.0')          # Thicker borders

    # --- 3. Edge Attributes (Readable Labels) ---
    dot.attr('edge', 
             fontname='Times-Roman', 
             fontsize='12',           # INCREASED EDGE LABEL SIZE
             color='black', 
             penwidth='1.5',
             arrowsize='1.0')

    # ============================================================
    # NODES
    # ============================================================
    
    # Start
    dot.node('Start', 'Start', shape='oval', style='filled,bold', penwidth='3.0')

    # Input
    dot.node('Input', 'Input: SDF File\n(merged_unique.sdf)', shape='parallelogram', slope='0.6')

    # Load Function
    dot.node('Load', 'Function: load_molecules_from_sdf\n(Parse blocks, Preserve ZINC IDs)', shape='box')

    # Init Stats
    dot.node('InitStats', 'Initialize Statistics\n(Counters for PAINS, Lipinski, Veber, SA)', shape='box')

    # Loop Start
    dot.node('LoopStart', 'Loop: For each molecule', shape='hexagon', style='bold')
    
    # Validity Check
    dot.node('Check_Valid', 'Is Molecule Valid?', shape='diamond')

    # --- Filter 1: PAINS ---
    dot.node('Dec_PAINS', 'Filter 1: PAINS\n(RDKit Default Catalog)\n\nHas Match?', shape='diamond')
    dot.node('Fail_PAINS', 'Action: Reject\n(Reason: PAINS)', shape='box', style='dashed')

    # --- Filter 2: Lipinski ---
    lipinski_label = ("Filter 2: Lipinski Rule of 5\n\n"
                      "Violates Criteria?\n"
                      "(MW > 500  OR  LogP > 5  OR\n"
                      "HBD > 5  OR  HBA > 10)")
    dot.node('Dec_Lipinski', lipinski_label, shape='diamond')
    dot.node('Fail_Lipinski', 'Action: Reject\n(Reason: Lipinski)', shape='box', style='dashed')

    # --- Filter 3: Veber ---
    veber_label = ("Filter 3: Veber Rules\n\n"
                   "Violates Criteria?\n"
                   "(RotBonds > 10  OR\n"
                   "TPSA > 140)")
    dot.node('Dec_Veber', veber_label, shape='diamond')
    dot.node('Fail_Veber', 'Action: Reject\n(Reason: Veber)', shape='box', style='dashed')

    # --- Filter 4: SA Score ---
    sa_label = ("Filter 4: Synthetic Accessibility\n\n"
                "Violates Criteria?\n"
                "(SA Score > 6.0)")
    dot.node('Dec_SA', sa_label, shape='diamond')
    dot.node('Fail_SA', 'Action: Reject\n(Reason: SA Score)', shape='box', style='dashed')

    # --- Pass ---
    dot.node('Pass', 'Action: PASS\n(Add to Valid List)', shape='box', style='bold', penwidth='3.0')

    # Invisible join node for routing lines back nicely
    dot.node('LoopEnd', '', shape='point', width='0')

    # --- End Loop Logic ---
    dot.node('Check_Loop', 'More Molecules?', shape='diamond')

    # Reporting
    dot.node('Report', 'Generate Reports\n(Markdown & Data Table)', shape='box')

    # Save
    dot.node('Save', 'Save Passed Molecules\n(final_filtered.sdf)', shape='parallelogram', slope='0.6')

    # End
    dot.node('End', 'End', shape='oval', style='filled,bold', penwidth='3.0')

    # ============================================================
    # EDGES
    # ============================================================

    dot.edge('Start', 'Input')
    dot.edge('Input', 'Load')
    dot.edge('Load', 'InitStats')
    dot.edge('InitStats', 'LoopStart')
    dot.edge('LoopStart', 'Check_Valid')

    # Validity
    dot.edge('Check_Valid', 'LoopEnd', label=' No')
    dot.edge('Check_Valid', 'Dec_PAINS', label=' Yes')

    # PAINS
    dot.edge('Dec_PAINS', 'Fail_PAINS', label=' Yes')
    dot.edge('Dec_PAINS', 'Dec_Lipinski', label=' No')

    # Lipinski
    dot.edge('Dec_Lipinski', 'Fail_Lipinski', label=' Yes')
    dot.edge('Dec_Lipinski', 'Dec_Veber', label=' No')

    # Veber
    dot.edge('Dec_Veber', 'Fail_Veber', label=' Yes')
    dot.edge('Dec_Veber', 'Dec_SA', label=' No')

    # SA Score
    dot.edge('Dec_SA', 'Fail_SA', label=' Yes')
    dot.edge('Dec_SA', 'Pass', label=' No')

    # Routing Failures back
    dot.edge('Fail_PAINS', 'LoopEnd')
    dot.edge('Fail_Lipinski', 'LoopEnd')
    dot.edge('Fail_Veber', 'LoopEnd')
    dot.edge('Fail_SA', 'LoopEnd')
    dot.edge('Pass', 'LoopEnd')

    # Loop Logic
    dot.edge('LoopEnd', 'Check_Loop')
    dot.edge('Check_Loop', 'LoopStart', label=' Yes')
    dot.edge('Check_Loop', 'Report', label=' No')

    # Finalize
    dot.edge('Report', 'Save')
    dot.edge('Save', 'End')

    # Output
    output_filename = 'thesis_filtering_pipeline_v2'
    dot.render(output_filename, format='png', cleanup=True)
    dot.render(output_filename, format='pdf', cleanup=True)
    
    print(f"✅ High-Visibility Flowchart Generated: {output_filename}.png")

if __name__ == '__main__':
    create_thesis_flowchart()

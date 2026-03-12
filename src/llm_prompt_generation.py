import numpy as np
import pandas as pd
import os

def generate_prompt_part1(data_dir='data'):
    """Describe the reaction and list all tested conditions with yields."""
    prompt = (
        "We are studying a metal-mediated reductive hydroalkylation reaction. "
        "In this reaction, a substituted styrene (0.2 mmol) and an alkyl iodide "
        "(0.3 mmol) are combined with a metallic reductant and a proton source "
        "in a solvent at room temperature for 24 hours to produce hydroalkylated "
        "products (anti-Markovnikov addition across the double bond).\n\n"
        "The baseline conditions use Mg (5 equiv.) as reductant, AcOH (8 equiv.) "
        "as proton source, and DMAc (0.2 M) as solvent.\n\n"
        "Here are all the experimental conditions and their corresponding yields:\n\n"
    )

    experiments = pd.read_excel(os.path.join(data_dir, 'init_experiments.xlsx'))
    for i in range(len(experiments)):
        row = experiments.iloc[i]
        red = row.get('Reductant', 'Mg')
        ps = row.get('ProtonSource', 'AcOH')
        sol = row.get('Solvent', 'DMAc')
        yld = row.get('Yield', 'N/A')
        prompt += f"Reductant: {red}, Proton source: {ps}, Solvent: {sol}, Yield: {yld}%\n"

    return prompt

def generate_prompt_part2():
    """Provide chemical background knowledge about the reaction mechanism."""
    prompt = "\nHere is some relevant chemical background:\n\n"

    prompt += "1. Role of the Metallic Reductant\n"
    prompt += (
        "The metal (e.g. Mg, Zn, Mn) serves as a single-electron-transfer (SET) agent. "
        "It reacts with the alkyl iodide to generate a carbon-centered radical via "
        "oxidative addition or SET. The reduction potential of the metal is critical: "
        "Mg (E0 = -2.37 V vs SHE) is a strong reductant capable of activating "
        "unactivated alkyl iodides. Zn (E0 = -0.76 V) is milder and may show different "
        "selectivity. The physical form (powder mesh size, turnings) affects surface area "
        "and therefore reaction rate.\n\n"
    )

    prompt += "2. Role of the Proton Source\n"
    prompt += (
        "The proton source (e.g. AcOH, pKa ~4.76) protonates the organometallic "
        "intermediate after radical addition to the styrene. The pKa controls the rate "
        "of protonation: too strong an acid (e.g. TFA, pKa ~0.23) may cause premature "
        "protonation or side reactions; too weak may leave unreacted intermediates. "
        "The stoichiometry matters — 8 equiv. of AcOH is used in the baseline, suggesting "
        "the proton source also plays a role in activating the metal surface.\n\n"
    )

    prompt += "3. Role of the Solvent\n"
    prompt += (
        "DMAc (N,N-dimethylacetamide) is a polar aprotic solvent with high donor number "
        "(DN = 27.8) that can coordinate to metal surfaces and stabilize radical "
        "intermediates. It dissolves both organic substrates and helps solvate the metal "
        "surface. However, DMAc is a reproductive toxicant (SVHC). Greener alternatives "
        "such as 2-MeTHF, ethanol, or water/surfactant systems may be desirable but "
        "could affect radical stability, metal activation, and substrate solubility.\n\n"
    )

    prompt += "4. Substrate Compatibility Considerations\n"
    prompt += (
        "The reaction should tolerate: (a) electron-withdrawing groups (-CF3, -F, -Cl, -Br) "
        "without cleaving aryl-halogen bonds, (b) electron-donating groups (-OMe, -tBu), "
        "(c) heterocycles (furan, thiophene), (d) ester and ether linkages. "
        "Conditions that are too reductive may cause aryl C-Cl or C-Br bond cleavage.\n"
    )

    return prompt

def generate_prompt_part3():
    """Ask the LLM to generate hypotheses and suggestions."""
    prompt = (
        "\n\nBased on the above data and chemical knowledge, please:\n"
        "1. Generate 5 hypotheses about which combinations of reductant, proton source, "
        "and solvent might lead to higher yields or broader substrate scope.\n"
        "2. For each hypothesis, explain your reasoning and cite supporting data points.\n"
        "3. Identify potential trade-offs between yield and green chemistry metrics "
        "(lower toxicity solvent, less excess reagent).\n\n"
    )
    return prompt

def generate_prompt(data_dir='data'):
    return (generate_prompt_part1(data_dir)
            + generate_prompt_part2()
            + generate_prompt_part3())

def generate_suggestion_prompt(data_dir='data'):
    avail_reductant = pd.read_excel(
        os.path.join(data_dir, 'reductant.xlsx'))['name'].tolist()
    avail_proton_source = pd.read_excel(
        os.path.join(data_dir, 'proton_source.xlsx'))['name'].tolist()
    avail_solvent = pd.read_excel(
        os.path.join(data_dir, 'solvent.xlsx'))['name'].tolist()

    prompt = (
        "For each hypothesis above, please recommend 3 new combinations of "
        "(reductant, proton source, solvent) that have NOT been tested yet "
        "and that you expect to give higher catalytic yields.\n\n"
        "You MUST only choose from the following available chemicals:\n\n"
    )

    prompt += "Available reductants:\n"
    for i, name in enumerate(avail_reductant):
        prompt += f"  {i+1}. {name}\n"

    prompt += "\nAvailable proton sources:\n"
    for i, name in enumerate(avail_proton_source):
        prompt += f"  {i+1}. {name}\n"

    prompt += "\nAvailable solvents:\n"
    for i, name in enumerate(avail_solvent):
        prompt += f"  {i+1}. {name}\n"

    prompt += (
        "\nFor each hypothesis, give the 3 suggestions in this format:\n"
        "Hypothesis N: <brief statement>\n"
        "  1. Reductant: ..., Proton source: ..., Solvent: ...\n"
        "  2. ...\n"
        "  3. ...\n"
    )
    return prompt


if __name__ == '__main__':
    prompt = generate_prompt()
    print(prompt)
    print("\n" + "=" * 60 + "\n")
    print(generate_suggestion_prompt())

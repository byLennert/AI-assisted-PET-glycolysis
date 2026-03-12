from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from bayes_opt.util import ensure_rng
import numpy as np
from data import get_one_embedding
import os
import pandas as pd
import openai
from utils import get_openai_key, AcquisitionFunction
import warnings
import json


class LLMEmbeddingBO:
    """Bayesian Optimiser using LLM embeddings for Mg-mediated hydroalkylation.

    Search space has THREE discrete dimensions:
      - Reductant   (e.g. Mg, Zn, Mn, Al ...)
      - Proton source (e.g. AcOH, TFA, propionic acid ...)
      - Solvent     (e.g. DMAc, DMF, 2-MeTHF, EtOH ...)

    Each entity is represented by a 1536-dim LLM embedding.  The three
    embeddings are concatenated (4608-dim) then projected to a lower
    dimension via a random matrix before being fed to the GP.
    """

    def __init__(self,
                 embedding_length=1536,
                 random_embedding=20,
                 random_state=None,
                 acquisition='ucb',
                 lazy=False):

        self.embedding_length = embedding_length
        self._random_state = ensure_rng(random_state)
        self.acquisition = acquisition
        self.n_variables = 3  # reductant, proton_source, solvent

        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state
        )

        if lazy:
            self.client = None
        else:
            self.client = openai.Client(api_key=get_openai_key())

        # ---- lookup tables ------------------------------------------------
        # "avail" tables: candidates the BO is allowed to suggest
        self._avail_reductant_table = {}
        self._avail_proton_source_table = {}
        self._avail_solvent_table = {}

        # "full" tables: every entity we have ever seen (incl. init data)
        self._reductant_table = {}
        self._proton_source_table = {}
        self._solvent_table = {}

        # Load any previously-cached embeddings from disk
        self.load_embedding()

        # Read candidate lists from Excel
        avail_reductant_info = pd.read_excel('data/reductant.xlsx')['name'].tolist()
        avail_proton_source_info = pd.read_excel('data/proton_source.xlsx')['name'].tolist()
        avail_solvent_info = pd.read_excel('data/solvent.xlsx')['name'].tolist()

        # ---- build / load embedding arrays for candidates ------------------
        self._all_possible_reductant = self._load_or_build_embeddings(
            avail_reductant_info, self._reductant_table,
            'data/all_possible_reductant.npy', role='reductant')
        self._all_possible_proton_source = self._load_or_build_embeddings(
            avail_proton_source_info, self._proton_source_table,
            'data/all_possible_proton_source.npy', role='proton_source')
        self._all_possible_solvent = self._load_or_build_embeddings(
            avail_solvent_info, self._solvent_table,
            'data/all_possible_solvent.npy', role='solvent')

        # Fill the "avail" lookup dicts
        for i, name in enumerate(avail_reductant_info):
            self._avail_reductant_table[name] = self._all_possible_reductant[i]
        for i, name in enumerate(avail_proton_source_info):
            self._avail_proton_source_table[name] = self._all_possible_proton_source[i]
        for i, name in enumerate(avail_solvent_info):
            self._avail_solvent_table[name] = self._all_possible_solvent[i]

        # ---- observed data -------------------------------------------------
        self._params = []
        self._targets = []

        # ---- random projection matrix --------------------------------------
        full_dim = embedding_length * self.n_variables  # 4608
        if random_embedding is None:
            self.random_embedding_matrix = np.eye(full_dim)
        else:
            cache_path = 'data/random_embedding_{}.npy'.format(random_embedding)
            if os.path.exists(cache_path):
                mat = np.load(cache_path)
                # If shape doesn't match (e.g. old 2-var cache), regenerate
                if mat.shape != (random_embedding, full_dim):
                    warnings.warn(
                        f"Cached random matrix shape {mat.shape} doesn't match "
                        f"expected ({random_embedding}, {full_dim}). Regenerating.")
                    mat = np.random.randn(random_embedding, full_dim)
                    np.save(cache_path, mat)
                self.random_embedding_matrix = mat
            else:
                self.random_embedding_matrix = np.random.randn(
                    random_embedding, full_dim)
                np.save(cache_path, self.random_embedding_matrix)

        self._evaluated_triple = []
        self.init_register()
        self.save_embedding()

    # ================================================================
    # Embedding helpers
    # ================================================================

    def _load_or_build_embeddings(self, name_list, table, cache_path, role):
        """Load cached .npy embeddings or build them via the LLM API."""
        if os.path.exists(cache_path):
            arr = np.load(cache_path)
            if arr.shape[0] == len(name_list):
                return arr
            else:
                warnings.warn(
                    f"Cached {cache_path} has {arr.shape[0]} rows but "
                    f"candidate list has {len(name_list)}. Rebuilding.")

        embeddings = []
        for name in name_list:
            if name in table:
                embeddings.append(table[name])
            else:
                emb = get_one_embedding(name, self.client, role=role)
                table[name] = np.array(emb)
                embeddings.append(table[name])
        arr = np.array(embeddings)
        np.save(cache_path, arr)
        return arr

    def save_embedding(self):
        np.save('data/all_possible_reductant.npy', self._all_possible_reductant)
        np.save('data/all_possible_proton_source.npy', self._all_possible_proton_source)
        np.save('data/all_possible_solvent.npy', self._all_possible_solvent)

    def load_embedding(self):
        """Restore cached embeddings from data/embeddings/*.json."""
        data_dir = 'data/embeddings'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        for file_name in os.listdir(data_dir):
            if not file_name.endswith('.json'):
                continue
            with open(os.path.join(data_dir, file_name), 'r') as f:
                json_data = json.load(f)
                emb = np.array(json_data['embedding'])
                mol = json_data['molecule']
                role = json_data.get('role', json_data.get('type', 'reductant'))

                if role == 'reductant':
                    if mol not in self._reductant_table:
                        self._reductant_table[mol] = emb
                elif role == 'proton_source':
                    if mol not in self._proton_source_table:
                        self._proton_source_table[mol] = emb
                elif role == 'solvent':
                    if mol not in self._solvent_table:
                        self._solvent_table[mol] = emb
                else:
                    # Legacy format ('acid' -> reductant, 'base' -> proton_source)
                    if role == 'acid' and mol not in self._reductant_table:
                        self._reductant_table[mol] = emb
                    elif role == 'base' and mol not in self._proton_source_table:
                        self._proton_source_table[mol] = emb

    # ================================================================
    # Random projection
    # ================================================================

    def random_embedding(self, x):
        """Project from full embedding space to low-dim space."""
        if len(x.shape) == 1:
            x_vector = x.reshape(1, -1)
            return (x_vector @ self.random_embedding_matrix.T)[0]
        else:
            return x @ self.random_embedding_matrix.T

    def restore_random_embedding(self, x):
        """Approximate inverse of the random projection."""
        pinv = np.linalg.pinv(self.random_embedding_matrix.T)
        if len(x.shape) == 1:
            x_vector = x.reshape(1, -1)
            return (x_vector @ pinv)[0]
        else:
            return x @ pinv

    # ================================================================
    # Data registration
    # ================================================================

    def register(self, param, reductant, proton_source, solvent, target):
        """Register a new observed data point."""
        self._params.append(param)
        self._targets.append(target)
        self._evaluated_triple.append((reductant, proton_source, solvent))

    def register_by_name(self, reductant, proton_source, solvent, target):
        """Register by chemical names (will fetch embeddings if needed)."""
        if reductant not in self._reductant_table:
            warnings.warn(f"Reductant '{reductant}' not cached. Fetching embedding.")
            emb = get_one_embedding(reductant, self.client, role='reductant')
            self._reductant_table[reductant] = np.array(emb)
        if proton_source not in self._proton_source_table:
            warnings.warn(f"Proton source '{proton_source}' not cached. Fetching embedding.")
            emb = get_one_embedding(proton_source, self.client, role='proton_source')
            self._proton_source_table[proton_source] = np.array(emb)
        if solvent not in self._solvent_table:
            warnings.warn(f"Solvent '{solvent}' not cached. Fetching embedding.")
            emb = get_one_embedding(solvent, self.client, role='solvent')
            self._solvent_table[solvent] = np.array(emb)

        param = np.hstack((
            self._reductant_table[reductant],
            self._proton_source_table[proton_source],
            self._solvent_table[solvent]
        ))
        self.register(param, reductant, proton_source, solvent, target)

    # ================================================================
    # Initial experiment loading
    # ================================================================

    def _get_or_fetch_embedding(self, name, primary_table, fallback_tables,
                                all_possible_arr, role):
        """Look up an embedding in tables, or fetch via API if missing."""
        if name in primary_table:
            return primary_table[name], all_possible_arr
        for tbl in fallback_tables:
            if name in tbl:
                return tbl[name], all_possible_arr
        emb = get_one_embedding(name, self.client, role=role)
        emb = np.array(emb)
        primary_table[name] = emb
        if len(all_possible_arr) > 0:
            all_possible_arr = np.vstack((all_possible_arr, emb))
        else:
            all_possible_arr = emb.reshape(1, -1)
        return emb, all_possible_arr

    def init_register(self, file=None):
        """Load initial experiments from Excel and register them."""
        if file is None:
            file = 'data/init_experiments.xlsx'

        if not os.path.exists(file):
            warnings.warn(f"Initial experiments file '{file}' not found. "
                          "Starting with empty dataset.")
            return

        rate = pd.read_excel(file)

        # The Excel is expected to have columns:
        #   Reductant, ProtonSource, Solvent, Yield
        # Multiple rows with the same (Reductant, ProtonSource, Solvent) and
        # different Yield values are averaged automatically.

        required_cols = {'Reductant', 'ProtonSource', 'Solvent', 'Yield'}
        if not required_cols.issubset(set(rate.columns)):
            # Try legacy format with Acid / Base columns
            if {'Acid', 'Base', 'Yield'}.issubset(set(rate.columns)):
                warnings.warn(
                    "Detected legacy Acid/Base format. Mapping Acid->Reductant, "
                    "Base->ProtonSource, defaulting Solvent to 'DMAc'.")
                rate = rate.rename(columns={'Acid': 'Reductant', 'Base': 'ProtonSource'})
                if 'Solvent' not in rate.columns:
                    rate['Solvent'] = 'DMAc'
            else:
                raise ValueError(
                    f"init_experiments.xlsx must contain columns {required_cols}. "
                    f"Found: {set(rate.columns)}")

        # Group by condition triple and average yields
        grouped = rate.groupby(
            ['Reductant', 'ProtonSource', 'Solvent'], dropna=False
        )['Yield'].mean().reset_index()

        for _, row in grouped.iterrows():
            reductant = str(row['Reductant']).strip()
            proton_source = str(row['ProtonSource']).strip()
            solvent = str(row['Solvent']).strip()
            target = float(row['Yield'])

            r_emb, self._all_possible_reductant = self._get_or_fetch_embedding(
                reductant, self._reductant_table,
                [self._proton_source_table, self._solvent_table],
                self._all_possible_reductant, role='reductant')

            p_emb, self._all_possible_proton_source = self._get_or_fetch_embedding(
                proton_source, self._proton_source_table,
                [self._reductant_table, self._solvent_table],
                self._all_possible_proton_source, role='proton_source')

            s_emb, self._all_possible_solvent = self._get_or_fetch_embedding(
                solvent, self._solvent_table,
                [self._reductant_table, self._proton_source_table],
                self._all_possible_solvent, role='solvent')

            param = np.hstack((r_emb, p_emb, s_emb))
            self.register(param, reductant, proton_source, solvent, target)

    # ================================================================
    # Gaussian Process
    # ================================================================

    def fit_gp(self):
        """Fit the GP on all registered observations."""
        target = np.array(self._targets)
        params = np.array([self.random_embedding(p) for p in self._params])
        self._gp.fit(params, target)

    # ================================================================
    # Suggest next experiments
    # ================================================================

    def suggest(self, batch_size=1):
        """Suggest the next batch of (reductant, proton_source, solvent) to try.

        Returns three lists: best_reductant, best_proton_source, best_solvent.
        """
        acquisition_func = AcquisitionFunction(self.acquisition)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fit_gp()

        y_max = np.max(self._targets)

        # Collect all (acq_value, reductant, proton_source, solvent) pairs
        candidates = []

        for red in self._avail_reductant_table:
            for ps in self._avail_proton_source_table:
                for sol in self._avail_solvent_table:
                    if (red, ps, sol) in self._evaluated_triple:
                        continue

                    r_emb = self._avail_reductant_table[red]
                    p_emb = self._avail_proton_source_table[ps]
                    s_emb = self._avail_solvent_table[sol]
                    embedding = np.hstack((r_emb, p_emb, s_emb))
                    low_emb = self.random_embedding(embedding)
                    acq_val = acquisition_func.utility(low_emb, self._gp, y_max)
                    candidates.append((acq_val, red, ps, sol))

        # Sort by acquisition value (descending) and take top batch_size
        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:batch_size]

        best_reductant = [c[1] for c in top]
        best_proton_source = [c[2] for c in top]
        best_solvent = [c[3] for c in top]

        # Pretty-print suggestions
        print(f"\n{'='*60}")
        print(f"  Top {batch_size} suggested experiments")
        print(f"{'='*60}")
        for i, (acq, r, p, s) in enumerate(top):
            print(f"  {i+1}. Reductant={r}, ProtonSource={p}, Solvent={s}  "
                  f"(acq={acq:.4f})")
        print(f"{'='*60}\n")

        return best_reductant, best_proton_source, best_solvent


if __name__ == '__main__':
    bo = LLMEmbeddingBO()
    print(bo.suggest(batch_size=6))
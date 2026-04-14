"""Synthetic causal scenario generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .types import CausalScenario


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


@dataclass(slots=True)
class CausalGraph:
    """Graph description for synthetic scenarios."""

    nodes: list[str]
    edges: list[tuple[str, str]]
    hidden_variables: list[str] = field(default_factory=list)
    confounders: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass(slots=True)
class SyntheticDataset:
    """Generic synthetic dataset used by helper generators."""

    data: pd.DataFrame
    causal_graph: CausalGraph
    ground_truth: dict[str, Any]
    difficulty: float
    metadata: dict[str, Any] = field(default_factory=dict)


class DataGenerator:
    """
    Generates benchmark causal scenarios with controllable difficulty.

    The returned object for round execution is ``CausalScenario`` so that
    downstream modules do not need to understand generator internals.
    """

    def __init__(self, config: dict[str, Any] | None = None, seed: int = 42):
        self.config = config or {}
        self.rng = np.random.default_rng(seed)

    def generate_scenario(
        self,
        difficulty: float,
        causal_level: int | None = None,
        scenario_id: str | None = None,
        n_samples: int | None = None,
    ) -> CausalScenario:
        """Generate one scenario tailored to the requested Pearl level."""

        level = causal_level or int(self.rng.choice([1, 2, 3]))
        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        scenario_name = scenario_id or {
            1: "smoking_cancer",
            2: "education_income",
            3: "drug_recovery",
        }[level]
        size = n_samples or self._resolve_sample_size(difficulty)

        if scenario_name == "smoking_cancer":
            scenario = self._build_smoking_scenario(difficulty, size)
        elif scenario_name == "education_income":
            scenario = self._build_education_scenario(difficulty, size)
        elif scenario_name == "drug_recovery":
            scenario = self._build_drug_scenario(difficulty, size)
        else:
            raise ValueError(f"Unknown scenario_id: {scenario_name}")

        scenario.causal_level = level
        return scenario

    def generate_linear_scm(self, n_vars: int, n_samples: int) -> SyntheticDataset:
        """Generate a random linear SCM for smoke-test and fallback usage."""

        if n_vars < 3:
            raise ValueError("n_vars must be >= 3")

        columns = [f"x{i}" for i in range(n_vars)]
        hidden = self.rng.normal(size=n_samples)
        data = {columns[0]: 0.8 * hidden + self.rng.normal(scale=0.8, size=n_samples)}
        edges: list[tuple[str, str]] = []

        for index in range(1, n_vars):
            parent = columns[index - 1]
            current = columns[index]
            weight = 0.5 + 0.2 * self.rng.random()
            data[current] = weight * data[parent] + 0.3 * hidden + self.rng.normal(
                scale=0.6, size=n_samples
            )
            edges.append((parent, current))

        frame = pd.DataFrame(data)
        graph = CausalGraph(nodes=columns + ["u0"], edges=edges, hidden_variables=["u0"])
        ground_truth = {
            "treatment": columns[0],
            "outcome": columns[-1],
            "approx_effect": float(frame[columns[-1]].corr(frame[columns[0]])),
        }
        return SyntheticDataset(
            data=frame,
            causal_graph=graph,
            ground_truth=ground_truth,
            difficulty=0.5,
            metadata={"family": "linear", "hidden_strength": 0.3},
        )

    def generate_nonlinear_scm(self, n_vars: int, n_samples: int) -> SyntheticDataset:
        """Generate a nonlinear SCM with saturating transforms."""

        if n_vars < 3:
            raise ValueError("n_vars must be >= 3")

        columns = [f"x{i}" for i in range(n_vars)]
        hidden = self.rng.normal(size=n_samples)
        base = self.rng.normal(size=n_samples)
        data = {
            columns[0]: np.tanh(base + 0.6 * hidden) + self.rng.normal(scale=0.15, size=n_samples)
        }
        edges: list[tuple[str, str]] = []

        for index in range(1, n_vars):
            parent = columns[index - 1]
            current = columns[index]
            data[current] = np.tanh(1.2 * data[parent] + 0.25 * hidden) + self.rng.normal(
                scale=0.18, size=n_samples
            )
            edges.append((parent, current))

        frame = pd.DataFrame(data)
        graph = CausalGraph(nodes=columns + ["u0"], edges=edges, hidden_variables=["u0"])
        ground_truth = {
            "treatment": columns[0],
            "outcome": columns[-1],
            "approx_effect": float(np.mean(data[columns[-1]] - data[columns[0]])),
        }
        return SyntheticDataset(
            data=frame,
            causal_graph=graph,
            ground_truth=ground_truth,
            difficulty=0.7,
            metadata={"family": "nonlinear", "hidden_strength": 0.25},
        )

    def inject_confounder(
        self, dataset: SyntheticDataset, target_pair: tuple[str, str]
    ) -> SyntheticDataset:
        """Inject a latent confounder that affects a variable pair."""

        source, target = target_pair
        if source not in dataset.data or target not in dataset.data:
            raise KeyError(f"Target pair not found in dataset: {target_pair}")

        confounder_name = f"u_conf_{source}_{target}"
        confounder = self.rng.normal(size=len(dataset.data))
        modified = dataset.data.copy()
        modified[source] = modified[source] + 0.35 * confounder
        modified[target] = modified[target] + 0.45 * confounder

        graph = CausalGraph(
            nodes=dataset.causal_graph.nodes + [confounder_name],
            edges=list(dataset.causal_graph.edges),
            hidden_variables=dataset.causal_graph.hidden_variables + [confounder_name],
            confounders=dataset.causal_graph.confounders + [(confounder_name, source, target)],
        )
        metadata = dict(dataset.metadata)
        metadata.setdefault("injected_biases", []).append("confounder")
        return SyntheticDataset(
            data=modified,
            causal_graph=graph,
            ground_truth=dict(dataset.ground_truth),
            difficulty=min(1.0, dataset.difficulty + 0.1),
            metadata=metadata,
        )

    def inject_collider(
        self, dataset: SyntheticDataset, target_pair: tuple[str, str]
    ) -> SyntheticDataset:
        """Inject a collider derived from two observed variables."""

        source, target = target_pair
        if source not in dataset.data or target not in dataset.data:
            raise KeyError(f"Target pair not found in dataset: {target_pair}")

        collider_name = f"collider_{source}_{target}"
        modified = dataset.data.copy()
        modified[collider_name] = (
            0.5 * modified[source]
            + 0.5 * modified[target]
            + self.rng.normal(scale=0.25, size=len(modified))
        )
        graph = CausalGraph(
            nodes=dataset.causal_graph.nodes + [collider_name],
            edges=list(dataset.causal_graph.edges) + [(source, collider_name), (target, collider_name)],
            hidden_variables=list(dataset.causal_graph.hidden_variables),
            confounders=list(dataset.causal_graph.confounders),
        )
        metadata = dict(dataset.metadata)
        metadata.setdefault("injected_biases", []).append("collider")
        return SyntheticDataset(
            data=modified,
            causal_graph=graph,
            ground_truth=dict(dataset.ground_truth),
            difficulty=min(1.0, dataset.difficulty + 0.05),
            metadata=metadata,
        )

    def inject_selection_bias(
        self, dataset: SyntheticDataset, condition_var: str
    ) -> SyntheticDataset:
        """Subsample rows to simulate selection bias."""

        if condition_var not in dataset.data:
            raise KeyError(f"Selection variable not found in dataset: {condition_var}")

        threshold = float(dataset.data[condition_var].quantile(0.45))
        mask = dataset.data[condition_var] >= threshold
        selected = dataset.data.loc[mask].reset_index(drop=True)
        metadata = dict(dataset.metadata)
        metadata.setdefault("injected_biases", []).append("selection_bias")
        metadata["selection_ratio"] = float(mask.mean())
        return SyntheticDataset(
            data=selected,
            causal_graph=dataset.causal_graph,
            ground_truth=dict(dataset.ground_truth),
            difficulty=min(1.0, dataset.difficulty + 0.05),
            metadata=metadata,
        )

    def generate_intervention_data(
        self, dataset: CausalScenario, do_var: str, do_value: float
    ) -> pd.DataFrame:
        """Generate do(X=x) data from one scenario."""

        if do_var not in dataset.full_data:
            raise KeyError(f"Intervention variable not found: {do_var}")

        full = dataset.full_data.copy()
        scenario_id = dataset.scenario_id

        if scenario_id == "smoking_cancer":
            full["smoking"] = do_value
            latent = (
                1.4 * full["smoking"]
                + 0.9 * full["genetic_risk"]
                + 0.018 * (full["age"] - 45.0)
                + full["noise_cancer"]
            )
            full["cancer_risk"] = latent
            full["cancer"] = (latent > 0.6).astype(int)
        elif scenario_id == "education_income":
            full["education_years"] = do_value
            full["income"] = (
                22.0
                + 3.4 * full["education_years"]
                + 6.5 * full["family_background"]
                + 0.8 * full["ability"]
                + full["noise_income"]
            )
        elif scenario_id == "drug_recovery":
            full["drug_taken"] = do_value
            full["biomarker"] = np.tanh(
                0.9 * full["drug_taken"]
                - 0.7 * full["severity"]
                + 0.5 * full["genotype"]
                + full["noise_biomarker"]
            )
            full["recovery_score"] = (
                0.8 * full["drug_taken"]
                + 0.6 * full["biomarker"]
                - 0.9 * full["severity"]
                + 0.5 * full["genotype"]
                + full["noise_recovery"]
            )
            full["recovered"] = (full["recovery_score"] > 0.2).astype(int)
        else:
            raise ValueError(f"Unsupported scenario for intervention: {scenario_id}")

        return full.drop(columns=dataset.hidden_variables, errors="ignore")

    def generate_counterfactual(
        self, dataset: CausalScenario, factual: dict[str, Any], intervention: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute a simple counterfactual for the closest factual row."""

        full = dataset.full_data
        factual_keys = [key for key in factual if key in full.columns]
        if not factual_keys:
            index = 0
        else:
            distances = np.zeros(len(full))
            for key in factual_keys:
                distances = distances + (full[key].to_numpy() - factual[key]) ** 2
            index = int(np.argmin(distances))

        row = full.iloc[[index]].copy()
        do_var, do_value = next(iter(intervention.items()))
        counterfactual = self.generate_intervention_data(
            CausalScenario(
                scenario_id=dataset.scenario_id,
                description=dataset.description,
                true_dag=dataset.true_dag,
                variables=dataset.variables,
                hidden_variables=dataset.hidden_variables,
                observed_data=row.drop(columns=dataset.hidden_variables, errors="ignore"),
                full_data=row,
                ground_truth=dataset.ground_truth,
                causal_level=dataset.causal_level,
                difficulty=dataset.difficulty,
                difficulty_config=dataset.difficulty_config,
                metadata=dataset.metadata,
            ),
            do_var=do_var,
            do_value=do_value,
        ).iloc[0]

        observed = dataset.observed_data.iloc[index].to_dict()
        return {
            "row_index": index,
            "factual": observed,
            "intervention": intervention,
            "counterfactual": counterfactual.to_dict(),
        }

    def _resolve_sample_size(self, difficulty: float) -> int:
        base = int(self.config.get("base_samples", 400))
        spread = int(self.config.get("difficulty_sample_span", 500))
        return max(120, base + int(spread * difficulty))

    def _difficulty_profile(self, difficulty: float) -> dict[str, float]:
        return {
            "noise_scale": 0.15 + 0.25 * difficulty,
            "hidden_strength": 0.55 + 0.6 * difficulty,
            "selection_bias_strength": 0.1 + 0.25 * difficulty,
            "nonlinearity": 0.2 + 0.6 * difficulty,
        }

    def _build_smoking_scenario(self, difficulty: float, n_samples: int) -> CausalScenario:
        profile = self._difficulty_profile(difficulty)
        genetic_risk = self.rng.normal(size=n_samples)
        age = self.rng.normal(45.0, 12.0, size=n_samples).clip(18, 80)
        smoke_latent = (
            0.7 * genetic_risk
            + 0.025 * (age - 45.0)
            + self.rng.normal(scale=profile["noise_scale"], size=n_samples)
        )
        smoking = self.rng.binomial(1, _sigmoid(smoke_latent))
        noise_cancer = self.rng.normal(scale=profile["noise_scale"], size=n_samples)
        cancer_risk = 1.4 * smoking + 0.9 * genetic_risk + 0.018 * (age - 45.0) + noise_cancer
        cancer = (cancer_risk > 0.6).astype(int)

        full = pd.DataFrame(
            {
                "age": age,
                "smoking": smoking,
                "cancer_risk": cancer_risk,
                "cancer": cancer,
                "genetic_risk": genetic_risk,
                "noise_cancer": noise_cancer,
            }
        )
        observed = full[["age", "smoking", "cancer_risk", "cancer"]].copy()
        intervention_one = self._smoking_do(full, 1.0)["cancer"].mean()
        intervention_zero = self._smoking_do(full, 0.0)["cancer"].mean()

        return CausalScenario(
            scenario_id="smoking_cancer",
            description="Smoking and cancer with hidden genetic risk.",
            true_dag={
                "genetic_risk": ["smoking", "cancer_risk", "cancer"],
                "age": ["smoking", "cancer_risk", "cancer"],
                "smoking": ["cancer_risk", "cancer"],
                "cancer_risk": ["cancer"],
                "cancer": [],
            },
            variables=list(observed.columns),
            hidden_variables=["genetic_risk", "noise_cancer"],
            observed_data=observed,
            full_data=full,
            ground_truth={
                "treatment": "smoking",
                "outcome": "cancer",
                "observational_difference": float(
                    observed.loc[observed["smoking"] == 1, "cancer"].mean()
                    - observed.loc[observed["smoking"] == 0, "cancer"].mean()
                ),
                "ate": float(intervention_one - intervention_zero),
            },
            causal_level=1,
            difficulty=difficulty,
            difficulty_config=profile,
            metadata={"scenario_family": "l1_association"},
        )

    def _build_education_scenario(self, difficulty: float, n_samples: int) -> CausalScenario:
        profile = self._difficulty_profile(difficulty)
        family_background = self.rng.normal(size=n_samples)
        ability = self.rng.normal(size=n_samples)
        quarter_of_birth = self.rng.integers(1, 5, size=n_samples)
        quarter_effect = np.take([0.0, 0.35, -0.1, 0.2], quarter_of_birth - 1)
        education_years = (
            12.0
            + 1.4 * family_background
            + 0.9 * ability
            + 0.7 * quarter_effect
            + self.rng.normal(scale=1.1 + profile["noise_scale"], size=n_samples)
        )
        noise_income = self.rng.normal(scale=2.0 + 2.0 * profile["noise_scale"], size=n_samples)
        income = 22.0 + 3.4 * education_years + 6.5 * family_background + 0.8 * ability + noise_income

        full = pd.DataFrame(
            {
                "quarter_of_birth": quarter_of_birth,
                "ability": ability,
                "education_years": education_years,
                "income": income,
                "family_background": family_background,
                "noise_income": noise_income,
            }
        )
        observed = full[["quarter_of_birth", "ability", "education_years", "income"]].copy()
        intervention_high = self._education_do(full, 16.0)["income"].mean()
        intervention_low = self._education_do(full, 12.0)["income"].mean()

        return CausalScenario(
            scenario_id="education_income",
            description="Education and income with hidden family background.",
            true_dag={
                "quarter_of_birth": ["education_years"],
                "family_background": ["education_years", "income"],
                "ability": ["education_years", "income"],
                "education_years": ["income"],
                "income": [],
            },
            variables=list(observed.columns),
            hidden_variables=["family_background", "noise_income"],
            observed_data=observed,
            full_data=full,
            ground_truth={
                "treatment": "education_years",
                "outcome": "income",
                "instrument": "quarter_of_birth",
                "observational_slope": float(
                    np.polyfit(observed["education_years"], observed["income"], deg=1)[0]
                ),
                "ate_16_vs_12": float(intervention_high - intervention_low),
            },
            causal_level=2,
            difficulty=difficulty,
            difficulty_config=profile,
            metadata={"scenario_family": "l2_intervention"},
        )

    def _build_drug_scenario(self, difficulty: float, n_samples: int) -> CausalScenario:
        profile = self._difficulty_profile(difficulty)
        genotype = self.rng.normal(size=n_samples)
        severity = self.rng.normal(loc=0.0, scale=1.0 + profile["noise_scale"], size=n_samples)
        treatment_logit = 0.6 * severity - 0.35 * genotype
        drug_taken = self.rng.binomial(1, _sigmoid(treatment_logit))
        noise_biomarker = self.rng.normal(scale=profile["noise_scale"], size=n_samples)
        biomarker = np.tanh(0.9 * drug_taken - 0.7 * severity + 0.5 * genotype + noise_biomarker)
        noise_recovery = self.rng.normal(scale=profile["noise_scale"], size=n_samples)
        recovery_score = 0.8 * drug_taken + 0.6 * biomarker - 0.9 * severity + 0.5 * genotype + noise_recovery
        recovered = (recovery_score > 0.2).astype(int)

        full = pd.DataFrame(
            {
                "severity": severity,
                "drug_taken": drug_taken,
                "biomarker": biomarker,
                "recovery_score": recovery_score,
                "recovered": recovered,
                "genotype": genotype,
                "noise_biomarker": noise_biomarker,
                "noise_recovery": noise_recovery,
            }
        )
        observed = full[["severity", "drug_taken", "biomarker", "recovery_score", "recovered"]].copy()
        intervention_treated = self._drug_do(full, 1.0)["recovered"].mean()
        intervention_control = self._drug_do(full, 0.0)["recovered"].mean()

        return CausalScenario(
            scenario_id="drug_recovery",
            description="Drug recovery with hidden genotype and mediated recovery.",
            true_dag={
                "genotype": ["severity", "biomarker", "recovery_score", "recovered"],
                "severity": ["drug_taken", "biomarker", "recovery_score", "recovered"],
                "drug_taken": ["biomarker", "recovery_score", "recovered"],
                "biomarker": ["recovery_score", "recovered"],
                "recovery_score": ["recovered"],
                "recovered": [],
            },
            variables=list(observed.columns),
            hidden_variables=["genotype", "noise_biomarker", "noise_recovery"],
            observed_data=observed,
            full_data=full,
            ground_truth={
                "treatment": "drug_taken",
                "outcome": "recovered",
                "mediator": "biomarker",
                "observational_difference": float(
                    observed.loc[observed["drug_taken"] == 1, "recovered"].mean()
                    - observed.loc[observed["drug_taken"] == 0, "recovered"].mean()
                ),
                "ate": float(intervention_treated - intervention_control),
            },
            causal_level=3,
            difficulty=difficulty,
            difficulty_config=profile,
            metadata={"scenario_family": "l3_counterfactual"},
        )

    def _smoking_do(self, full: pd.DataFrame, do_value: float) -> pd.DataFrame:
        intervened = full.copy()
        intervened["smoking"] = do_value
        cancer_risk = (
            1.4 * intervened["smoking"]
            + 0.9 * intervened["genetic_risk"]
            + 0.018 * (intervened["age"] - 45.0)
            + intervened["noise_cancer"]
        )
        intervened["cancer_risk"] = cancer_risk
        intervened["cancer"] = (cancer_risk > 0.6).astype(int)
        return intervened

    def _education_do(self, full: pd.DataFrame, do_value: float) -> pd.DataFrame:
        intervened = full.copy()
        intervened["education_years"] = do_value
        intervened["income"] = (
            22.0
            + 3.4 * intervened["education_years"]
            + 6.5 * intervened["family_background"]
            + 0.8 * intervened["ability"]
            + intervened["noise_income"]
        )
        return intervened

    def _drug_do(self, full: pd.DataFrame, do_value: float) -> pd.DataFrame:
        intervened = full.copy()
        intervened["drug_taken"] = do_value
        intervened["biomarker"] = np.tanh(
            0.9 * intervened["drug_taken"]
            - 0.7 * intervened["severity"]
            + 0.5 * intervened["genotype"]
            + intervened["noise_biomarker"]
        )
        intervened["recovery_score"] = (
            0.8 * intervened["drug_taken"]
            + 0.6 * intervened["biomarker"]
            - 0.9 * intervened["severity"]
            + 0.5 * intervened["genotype"]
            + intervened["noise_recovery"]
        )
        intervened["recovered"] = (intervened["recovery_score"] > 0.2).astype(int)
        return intervened

# Employment-Status Prediction from Developer Skills

A team machine-learning project comparing how education, coding experience and
technical-skill representations predict respondents' employment status in the
2024 Stack Overflow Developer Survey.

> **Project attribution:** This repository is a portfolio fork of the
> [original Group 5 university project](https://github.com/mchikovaniieu2024-create/ML-fundamentals-2026-Final-Project-Group5).
> It is presented as collaborative work, with individual ownership documented
> below.

## Deliverables

- [Final report](reports/final_report.pdf)
- [Presentation](slides/final_presentation.pdf)
- [Reported model results](results/reported_model_results.csv)

## Research question

Which feature group is most predictive of employment status within this
developer-survey dataset: education, coding experience or technical skills?

The target is `Employed`, where `1` represents an employed respondent and `0`
represents a non-employed respondent. The project predicts **employment
status**, not an employer's hiring decision, and its findings should not be
interpreted causally.

## Dataset and leakage controls

- 73,462 observations and 15 original columns
- 39,392 employed respondents (53.6%) and 34,070 non-employed respondents
  (46.4%)
- Stratified 70%/15%/15% train, validation and test split
- Preprocessing fitted on training data only
- `Employment` removed because it directly encodes the target
- `PreviousSalary` removed because it is a strong employment proxy
- 63 missing `HaveWorkedWith` entries replaced with empty skill lists

## Feature engineering and modelling

The project compares five experimental configurations:

1. **Baseline:** education, demographics and coding experience
2. **Education:** ordinal and one-hot encoded background variables
3. **Experience:** standardised years coding and years coding professionally
4. **Skills:** four representations of semicolon-separated technology lists
   - total skill count
   - grouped technology domains
   - sparse binary skill indicators
   - semantic embeddings from `all-MiniLM-L6-v2`
5. **Combined:** education, experience and semantic skill embeddings

Models include a zero-rule benchmark, logistic regression, random forest and
gradient boosting. Evaluation uses accuracy, precision, recall, F1 and ROC-AUC.

## Selected results

| Experiment | Model | Test F1 | Test ROC-AUC |
|---|---|---:|---:|
| Combined | Logistic regression | 0.955 | **0.992** |
| Skills embeddings | Logistic regression | 0.951 | **0.990** |
| Skills grouped | Logistic regression | 0.846 | 0.921 |
| Skills count | Logistic regression | 0.788 | 0.872 |
| Education | Logistic regression | 0.689 | 0.569 |
| Experience | Logistic regression | 0.698 | 0.491 |

The central result was that feature representation mattered more than model
complexity: semantic representations of technical skills carried substantially
more predictive signal than education or years of coding alone.

### Why the perfect binary-skills score is not the headline result

The sparse binary-skills logistic-regression model returned a test ROC-AUC of
1.000. The team treated this as suspicious rather than as proof of a perfect
real-world model. A shuffled-label diagnostic returned chance-level
performance, which reduced the likelihood of direct pipeline leakage, but the
result may still reflect dataset-specific separability or memorisation.

For that reason, the project uses the semantic-embedding model as the more
defensible portfolio result and explicitly notes that external validation is
required.

## My contribution — Jade Beeks

My work focused on the technical-skills modelling stream:

- Implemented the skill parsing, normalisation and train-only vocabulary logic
- Engineered count, grouped, sparse binary and sentence-embedding skill
  representations
- Implemented the logistic-regression and random-forest skills experiments

The original Git history and final report preserve the team's individual
contributions.

## Team and module ownership

| Team member | Primary contribution |
|---|---|
| Massimo Vitale | Data preprocessing, cleaning pipelines and configuration |
| Avisa Ansari | Baseline and education-only models |
| Jade Beeks | Skills feature engineering and skills models |
| Marie Chikovani | Experience-only and combined models |
| Maya Tamimi | Evaluation metrics and plots |
| Faris Selimovic | Main integration script |

## Repository structure

```text
.
├── data/raw/                  # source dataset
├── reports/final_report.pdf
├── results/                   # preserved final reported metrics
├── slides/final_presentation.pdf
├── src/
│   ├── baseline.py
│   ├── combined.py
│   ├── config.py
│   ├── data_utils.py
│   ├── education.py
│   ├── evaluation.py
│   ├── experience.py
│   ├── feature_engineering.py
│   ├── process_skills.py
│   ├── skills.py
│   └── results/               # regenerated plots and tables
├── tests/
├── main.py
└── requirements.txt
```

## Reproduce the project

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.
The first embedding run downloads the `all-MiniLM-L6-v2` model and therefore
requires an internet connection. Generated tables and plots are written to
`src/results/`.

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

The test suite checks data cleaning, leakage-column removal, split isolation,
skill parsing, train-only vocabulary behaviour and feature-transformer output.

## Limitations and responsible use

- Metrics come from one fixed split rather than repeated cross-validation.
- Hyperparameters were selected manually rather than through systematic search.
- There is no external or temporal validation dataset.
- Embeddings improve prediction but reduce interpretability.
- Demographic variables can encode structural bias; the model should not be
  used to make real hiring decisions without subgroup fairness analysis.
- Results describe associations within this survey sample and do not establish
  that a particular skill, degree or amount of experience causes employment.

## License

The project code is available under the MIT License included in this
repository. The Stack Overflow survey data remains subject to its original
terms and attribution requirements.

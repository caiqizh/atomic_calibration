## Notes for FactChecking

- We support `bios`, `longfact` and `wildhallu` datasets. 
    - `longfact` only supports "zero".

- The tool supports different evidence types:
  - "zero": No evidence is provided.
  - "topk": Top-k evidence passages are provided. **Defatul value: 1**
  - "all": All evidence passages are provided.


## Usage

1. Import the `FactChecker` class:

```python
from fact_checker import FactChecker
```

2. Create a `FactChecker` instance with a maximum evidence length:

```python
fc = FactChecker(max_evidence_length=100000)
```

3. Build evidence from different knowledge sources (e.g., English Wikipedia, Google Search):

```python
fc.build_enwiki_evidence("../factcheck_cache/enwiki-20230401.db")
fc.build_google_search(db_path="../factcheck_cache/wildhallu.db")
```

4. Get veracity labels for a list of facts using the `get_veracity_labels` method:

```python
topic = "University of Cambridge"
fact_list = [
    "Cambridge University is a public research university in Cambridge, England.",
    "Cambridge University was founded in 1209.",
    "Cambridge University has 29 colleges.",
    "Cambridge University was founded in 1309.",
    "Cambridge University has 31 colleges."
]

veracity_labels = fc.get_veracity_labels(topic, fact_list, "wildhallu", evidence_type="topk")
for fact, label in zip(fact_list, veracity_labels):
    print(fact, label)
```

The `get_veracity_labels` method takes the following arguments:

- `topic` (str): The subject of the facts.
- `atomic_facts` (list): A list of facts to verify.
- `knowledge_source` (str): The knowledge source to use (e.g., "bios", "wildhallu", "longfact").
- `evidence_type` (str): The type of evidence to use for fact-checking (e.g., "zero", "topk", "all").



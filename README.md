# Atomic Calibration
We support dataset options `bios`, `longfact`, and `wildhallu`.

- For a given prompt (e.g. "tell me a bio of jacky chan"), we first generate a answer *A* for it (with temperature 0) in `src/generate_responses_vllm.py`. 

- Then, we generate another 20 samples (with temperature 1) for generative sampling in `src/generate_responses_vllm.py`. 

> For the above two generations, we can do it automatically in `scripts/generate_responses.sh`

- Then, we break the answer *A* into atomic facts with `src/generate_atomic_facts.py`.

> To generate more than one datasets, we can use `scripts/generate_atomic_facts.sh`

- Then we calculate the confidence using `src/calculate_uncertainty.py`
    - the codes for generative confidence is in `src/luq_vllm.py`
    - the codes for discriminate confidence is in  `src/dis_vllm.py`

- Visualization and analysis are in `visualization/visualization.ipynb`
use once_cell::sync::Lazy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

static RUNTIME: Lazy<tokio::runtime::Runtime> =
    Lazy::new(|| tokio::runtime::Runtime::new().expect("tokio runtime"));

struct PyTaskLlm {
    callback: PyObject,
}

#[async_trait::async_trait]
impl prompt_opt::LlmClient for PyTaskLlm {
    async fn complete(&self, prompt: &str) -> prompt_opt::Result<String> {
        Python::with_gil(|py| {
            let out = self
                .callback
                .call1(py, (prompt,))
                .map_err(|e| prompt_opt::MiproError::Llm(e.to_string()))?;
            out.extract::<String>(py)
                .map_err(|e| prompt_opt::MiproError::Llm(e.to_string()))
        })
    }

    fn name(&self) -> &'static str {
        "python-callback-llm"
    }
}

fn map_err(err: prompt_opt::MiproError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyfunction]
fn proposer_backends() -> Vec<String> {
    vec!["single_prompt".to_string(), "rlm".to_string()]
}

#[pyfunction]
fn run_mipro_json(
    config_json: &str,
    initial_policy_json: &str,
    dataset_json: &str,
    task_llm: PyObject,
) -> PyResult<String> {
    let cfg: prompt_opt::MiproConfig =
        serde_json::from_str(config_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let initial_policy: prompt_opt::Policy = serde_json::from_str(initial_policy_json)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dataset: prompt_opt::Dataset =
        serde_json::from_str(dataset_json).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let llm = Arc::new(PyTaskLlm { callback: task_llm }) as Arc<dyn prompt_opt::LlmClient>;
    let metric = Arc::new(prompt_opt::ExactMatchMetric) as Arc<dyn prompt_opt::EvalMetric>;
    let sampler = Arc::new(prompt_opt::BasicSampler::new()) as Arc<dyn prompt_opt::VariantSampler>;
    let evaluator = prompt_opt::Evaluator::new(llm, metric);
    let optimizer = prompt_opt::Optimizer::new(sampler, evaluator);

    let result = RUNTIME
        .block_on(async { optimizer.run_batch(cfg, initial_policy, dataset).await })
        .map_err(map_err)?;

    serde_json::to_string(&result).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pymodule]
fn prompt_opt_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(proposer_backends, m)?)?;
    m.add_function(wrap_pyfunction!(run_mipro_json, m)?)?;
    Ok(())
}

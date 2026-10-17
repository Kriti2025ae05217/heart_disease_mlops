"""Thin wrapper around MLflow so training also works with --no-mlflow."""
from contextlib import contextmanager


class Tracker:
    def __init__(self, enabled: bool = True, experiment: str = "heart-disease",
                 tracking_uri: str = None):
        self.enabled = enabled
        self.mlflow = None
        if enabled:
            import mlflow  # imported lazily
            self.mlflow = mlflow
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment)

    @contextmanager
    def run(self, name: str, nested: bool = False):
        if not self.enabled:
            yield None
            return
        with self.mlflow.start_run(run_name=name, nested=nested) as run:
            yield run

    def params(self, params: dict):
        if self.enabled:
            self.mlflow.log_params({k: str(v) for k, v in params.items()})

    def metrics(self, metrics: dict):
        if self.enabled:
            self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    def artifact(self, path):
        if self.enabled:
            self.mlflow.log_artifact(str(path))

    def tag(self, key, value):
        if self.enabled:
            self.mlflow.set_tag(key, value)

    def log_model(self, pipeline, X_example, name="model"):
        if not self.enabled:
            return
        import mlflow.sklearn
        from mlflow.models import infer_signature
        sig = infer_signature(X_example, pipeline.predict_proba(X_example)[:, 1])
        kwargs = dict(signature=sig, input_example=X_example.head(3),
                      # MLflow 3.x defaults to skops, whose audit rejects numpy.dtype
                      serialization_format="cloudpickle")
        try:
            mlflow.sklearn.log_model(pipeline, name=name, **kwargs)   # MLflow >= 3
        except TypeError:
            mlflow.sklearn.log_model(pipeline, artifact_path=name, **kwargs)

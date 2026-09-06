from dataclasses import asdict, dataclass
import os


@dataclass
class VLLMConfig:
    """Launch and runtime configuration for vLLM model engine."""

    model: str = "Qwen/Qwen2.5-14B-Instruct"
    host: str = "0.0.0.0"
    port: int = 8001
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    quantization: str | None = None
    swap_space: int = 4
    enable_prefix_caching: bool = True
    trust_remote_code: bool = True
    api_key: str | None = None

    def to_cli_args(self) -> list[str]:
        """Convert config to command line arguments for vllm.entrypoints.openai.api_server."""
        args = [
            "--model",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--dtype",
            self.dtype,
            "--swap-space",
            str(self.swap_space),
        ]
        if self.enable_prefix_caching:
            args.append("--enable-prefix-caching")
        if self.trust_remote_code:
            args.append("--trust-remote-code")
        if self.quantization:
            args.extend(["--quantization", self.quantization])
        if self.api_key:
            args.extend(["--api-key", self.api_key])
        return args

    def to_dict(self) -> dict:
        return asdict(self)


def get_vllm_launch_command(config: VLLMConfig | None = None) -> str:
    """Produce executable shell string for starting vLLM."""
    cfg = config or VLLMConfig(
        model=os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-14B-Instruct"),
        port=int(os.getenv("VLLM_PORT", "8001")),
    )
    args = cfg.to_cli_args()
    return f"python -m vllm.entrypoints.openai.api_server {' '.join(args)}"

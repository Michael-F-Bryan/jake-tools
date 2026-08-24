from .dyld_fallback import ensure_homebrew_dyld_fallback_path

# Must run before anything that might import torch/torchcodec (dotenv, the
# CLI package, ...) — see dyld_fallback's module docstring for why.
ensure_homebrew_dyld_fallback_path()

import dotenv  # noqa: E402

# Must be executed before importing the CLI
dotenv.load_dotenv()

from .cli import main  # noqa: E402

if __name__ == "__main__":
    main()

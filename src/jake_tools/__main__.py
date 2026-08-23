import dotenv

# Must be executed before importing the CLI
dotenv.load_dotenv()

from .cli import main  # noqa: E402

if __name__ == "__main__":
    main()

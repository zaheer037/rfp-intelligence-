"""Allow python -m rfp to use the same CLI as the installed rfp command."""

from rfp.cli import main

if __name__ == "__main__":
    main()

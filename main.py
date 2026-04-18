from __future__ import annotations


def main(argv: list[str] | None = None) -> None:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "standardize":
        from src.datasets.standardize import cli as standardize_cli

        standardize_cli.main(args[1:])
        return

    from src.models import dynnet

    dynnet.main(args)


if __name__ == "__main__":
    main()

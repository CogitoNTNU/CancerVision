from __future__ import annotations


def main(argv: list[str] | None = None) -> None:
    import sys

    from src.models import dynnet

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "standardize":
        from src.datasets.standardize import cli as standardize_cli

        standardize_cli.main(args[1:])
        return

    dynnet.main(args)


if __name__ == "__main__":
    main()

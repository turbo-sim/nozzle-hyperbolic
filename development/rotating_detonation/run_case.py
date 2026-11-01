"""Execute the angle-only diffuser study as a file or repository module."""
if __package__:
    from .scripts.diffuser_run import main
else:
    from scripts.diffuser_run import main

if __name__ == '__main__':
    main()

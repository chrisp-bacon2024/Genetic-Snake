"""Entry point for the Genetic Snake pygame application."""

from ui.app import SnakeApp


def main() -> None:
    """Create the app window and run the main loop until the user quits."""
    SnakeApp().run()


if __name__ == "__main__":
    main()

"""Modal dialog to jump the live game to a target score."""

from __future__ import annotations

import pygame

import config


class _DoubleClickDetector:
    """Detect two clicks on the same target within a short interval."""

    def __init__(self, interval_s: float = 0.4) -> None:
        self._interval = interval_s
        self._last_time = -1.0
        self._last_target: str | None = None

    def register(self, target: str, now: float) -> bool:
        if self._last_target == target and 0 <= now - self._last_time <= self._interval:
            self._last_time = -1.0
            self._last_target = None
            return True
        self._last_time = now
        self._last_target = target
        return False


class ScoreJumpDialog:
    """Overlay on the playfield: type a score and confirm with OK or Enter."""

    _BOX_WIDTH = 240
    _BOX_HEIGHT = 132
    _INPUT_HEIGHT = 26
    _BUTTON_WIDTH = 72
    _BUTTON_HEIGHT = 26
    _MAX_DIGITS = 4

    def __init__(self, surface_width: int, surface_height: int) -> None:
        self._surface_width = surface_width
        self._surface_height = surface_height
        self._font = pygame.font.SysFont("consolas", 16)
        self._hint_font = pygame.font.SysFont("consolas", 12)
        self._button_font = pygame.font.SysFont("consolas", 14)
        self.active = False
        self.text = ""
        self.message = ""
        self._panel_origin_x = 0
        self._panel_origin_y = 0
        self._input_rect = pygame.Rect(0, 0, 0, self._INPUT_HEIGHT)
        self._ok_rect = pygame.Rect(0, 0, self._BUTTON_WIDTH, self._BUTTON_HEIGHT)
        self._cancel_rect = pygame.Rect(0, 0, self._BUTTON_WIDTH, self._BUTTON_HEIGHT)
        self._box_rect = pygame.Rect(0, 0, self._BOX_WIDTH, self._BOX_HEIGHT)
        self._layout()

    @property
    def panel_origin(self) -> tuple[int, int]:
        return self._panel_origin_x, self._panel_origin_y

    def set_panel_origin(self, x: int, y: int = 0) -> None:
        self._panel_origin_x = x
        self._panel_origin_y = y
        self._layout()

    def open(self) -> None:
        self.active = True
        self.text = ""
        self.message = ""

    def close(self) -> None:
        self.active = False
        self.text = ""
        self.message = ""

    def handle_event(
        self,
        event: pygame.event.Event,
        *,
        max_score: int,
    ) -> bool | int:
        """
        Handle pygame events while the dialog is open.

        Returns False if inactive or unhandled, True if consumed, or the parsed
        target score when the user confirms with OK / Enter.
        """
        if not self.active:
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            screen_x, screen_y = event.pos
            local = (screen_x - self._panel_origin_x, screen_y - self._panel_origin_y)
            if self._ok_rect.collidepoint(local):
                return self._confirm(max_score)
            if self._cancel_rect.collidepoint(local):
                self.close()
                return True
            if not self._box_rect.collidepoint(local):
                self.close()
                return True
            return True

        if event.type != pygame.KEYDOWN:
            return True

        if event.key == pygame.K_ESCAPE:
            self.close()
            return True
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            return self._confirm(max_score)
        if event.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
            self.message = ""
            return True
        if event.unicode.isdigit() and len(self.text) < self._MAX_DIGITS:
            self.text += event.unicode
            self.message = ""
            return True
        return True

    def draw(self, surface: pygame.Surface, *, max_score: int) -> None:
        if not self.active:
            return

        overlay = pygame.Surface((self._surface_width, self._surface_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        surface.blit(overlay, (0, 0))

        pygame.draw.rect(surface, config.COLOR_PANEL, self._box_rect, border_radius=6)
        pygame.draw.rect(surface, config.COLOR_CONTROL_BORDER, self._box_rect, width=1, border_radius=6)

        title = self._font.render("Go to score", True, config.COLOR_TEXT)
        title_rect = title.get_rect(centerx=self._box_rect.centerx, top=self._box_rect.top + 10)
        surface.blit(title, title_rect)

        hint = self._hint_font.render(f"0–{max_score} (same apple seed)", True, config.COLOR_TEXT_DIM)
        hint_rect = hint.get_rect(centerx=self._box_rect.centerx, top=title_rect.bottom + 4)
        surface.blit(hint, hint_rect)

        pygame.draw.rect(surface, config.COLOR_CONTROL_INACTIVE, self._input_rect, border_radius=4)
        pygame.draw.rect(surface, config.COLOR_CONTROL_ACTIVE, self._input_rect, width=1, border_radius=4)
        display = self.text if self.text else "…"
        input_text = self._font.render(display, True, config.COLOR_TEXT)
        surface.blit(input_text, input_text.get_rect(center=self._input_rect.center))

        self._draw_button(surface, self._ok_rect, "OK", primary=True)
        self._draw_button(surface, self._cancel_rect, "Cancel", primary=False)

        if self.message:
            msg = self._hint_font.render(self.message, True, config.COLOR_GAME_OVER)
            msg_rect = msg.get_rect(centerx=self._box_rect.centerx, top=self._box_rect.bottom + 6)
            surface.blit(msg, msg_rect)

    def _draw_button(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        label: str,
        *,
        primary: bool,
    ) -> None:
        border = config.COLOR_CONTROL_ACTIVE if primary else config.COLOR_CONTROL_BORDER
        pygame.draw.rect(surface, config.COLOR_CONTROL_INACTIVE, rect, border_radius=4)
        pygame.draw.rect(surface, border, rect, width=1, border_radius=4)
        text = self._button_font.render(label, True, config.COLOR_TEXT)
        surface.blit(text, text.get_rect(center=rect.center))

    def _confirm(self, max_score: int) -> bool | int:
        target = self._parse(max_score)
        if target is None:
            self.message = f"Enter 0–{max_score}"
            return True
        self.close()
        return target

    def _parse(self, max_score: int) -> int | None:
        if not self.text:
            return None
        try:
            value = int(self.text)
        except ValueError:
            return None
        if value < 0 or value > max_score:
            return None
        return value

    def _layout(self) -> None:
        box_x = (self._surface_width - self._BOX_WIDTH) // 2
        box_y = (self._surface_height - self._BOX_HEIGHT) // 2
        self._box_rect = pygame.Rect(box_x, box_y, self._BOX_WIDTH, self._BOX_HEIGHT)

        input_width = self._BOX_WIDTH - 32
        input_x = box_x + (self._BOX_WIDTH - input_width) // 2
        self._input_rect = pygame.Rect(input_x, box_y + 52, input_width, self._INPUT_HEIGHT)

        button_y = box_y + self._BOX_HEIGHT - self._BUTTON_HEIGHT - 12
        gap = 12
        buttons_width = self._BUTTON_WIDTH * 2 + gap
        start_x = box_x + (self._BOX_WIDTH - buttons_width) // 2
        self._ok_rect = pygame.Rect(start_x, button_y, self._BUTTON_WIDTH, self._BUTTON_HEIGHT)
        self._cancel_rect = pygame.Rect(
            start_x + self._BUTTON_WIDTH + gap,
            button_y,
            self._BUTTON_WIDTH,
            self._BUTTON_HEIGHT,
        )


def score_rect_screen(
    renderer_score_rect: pygame.Rect,
    *,
    panel_width: int,
    panel_origin_y: int = 0,
) -> pygame.Rect:
    """Convert a score hit rect from playfield coords to window/screen coords."""
    return pygame.Rect(
        panel_width + renderer_score_rect.x,
        panel_origin_y + renderer_score_rect.y,
        renderer_score_rect.width,
        renderer_score_rect.height,
    )


def try_open_score_dialog_on_double_click(
    event: pygame.event.Event,
    *,
    score_rect: pygame.Rect,
    panel_width: int,
    dialog: ScoreJumpDialog,
    detector: _DoubleClickDetector,
    now: float,
) -> bool:
    """
    Open the score dialog on a double-click of the HUD score label.

    Returns True if the event was consumed.
    """
    if dialog.active:
        return False
    if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
        return False
    screen_rect = score_rect_screen(score_rect, panel_width=panel_width)
    if not screen_rect.collidepoint(event.pos):
        return False
    if detector.register("score", now):
        dialog.open()
        return True
    return False

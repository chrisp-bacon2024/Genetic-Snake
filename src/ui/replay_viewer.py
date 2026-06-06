"""
Watch the saved best snakes from a training run.

Each generation's best genome is stored in ``replays/gen_XXXX.npz`` (genes + the food
seed of its best run). The viewer re-simulates each one live, reusing the same game
renderer and neural-network panel as the interactive app, and cycles through them in
generation order. The current generation and score are shown in the window caption.

Controls: N / Right / Space = next, Left = previous, Esc = quit.
Hold a nav key for 3+ seconds to skip 10 generations at a time.
G focuses the generation box; type a number and press Enter to jump.

``LiveReplayViewer`` polls the replays folder during training and plays each generation
as soon as its file appears, staying on the death screen until the next one is ready.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

import config
from controllers.ai_controller import AIController
from evolution.genome import Genome
from game.game import Game
from models.grid import Grid
from neural.network import NeuralNetwork

from .control_panel import ControlPanel
from .game_renderer import GameRenderer, board_layout


@dataclass
class _ReplaySession:
    generation: int
    score: int
    grid_cols: int
    grid_rows: int
    game: Game
    controller: AIController
    control_panel: ControlPanel
    renderer: GameRenderer
    steps: int = 0
    tick_accumulator: float = 0.0


def _load_genome_file(path: Path) -> tuple[int, int, Genome, int, int, int]:
    data = np.load(path)
    genome = Genome(np.asarray(data["genes"], dtype=np.float64))
    generation = int(data["generation"])
    score = int(data["score"])
    food_seed = int(data["food_seed"])
    grid_cols = int(data["grid_cols"]) if "grid_cols" in data else config.GRID_COLS
    grid_rows = int(data["grid_rows"]) if "grid_rows" in data else config.GRID_ROWS
    return generation, score, genome, food_seed, grid_cols, grid_rows


def _gen_path(replays_dir: Path, generation: int) -> Path | None:
    path = replays_dir / f"gen_{generation:04d}.npz"
    return path if path.exists() else None


_NAV_NEXT_KEYS = (pygame.K_n, pygame.K_RIGHT, pygame.K_SPACE)
_HOLD_FAST_THRESHOLD_S = 3.0
_HOLD_REPEAT_NORMAL_S = 0.35
_HOLD_REPEAT_FAST_S = 0.18
_HOLD_FAST_SKIP = 10
_GEN_INPUT_GAP = 8
_NAV_BAR_HEIGHT = 16


def _draw_replay_nav_bar(surface: pygame.Surface) -> None:
    """Single horizontal row of replay shortcuts along the panel bottom."""
    font = pygame.font.SysFont("consolas", 10)
    segments = ("N/Space next", "<- prev", "G gen", "Esc quit")
    rendered = [font.render(text, True, config.COLOR_TEXT_DIM) for text in segments]
    gap = 10
    total_width = sum(text.get_width() for text in rendered) + gap * (len(rendered) - 1)
    x = (config.PANEL_WIDTH - total_width) // 2
    y = config.WINDOW_HEIGHT - _NAV_BAR_HEIGHT
    for text in rendered:
        surface.blit(text, (x, y))
        x += text.get_width() + gap


class _NavHoldTracker:
    """Tap = 1 generation; hold 3s+ then repeats skip HOLD_FAST_SKIP at a time."""

    def __init__(self) -> None:
        self._direction = 0
        self._hold_time = 0.0
        self._repeat_accumulator = 0.0

    def on_key_down(self, direction: int) -> int:
        self._direction = direction
        self._hold_time = 0.0
        self._repeat_accumulator = 0.0
        return direction

    def update(self, delta: float) -> int:
        direction = _held_nav_direction()
        if direction == 0:
            self._direction = 0
            self._hold_time = 0.0
            self._repeat_accumulator = 0.0
            return 0

        if direction != self._direction:
            self._direction = direction
            self._hold_time = 0.0
            self._repeat_accumulator = 0.0

        self._hold_time += delta
        self._repeat_accumulator += delta

        fast = self._hold_time >= _HOLD_FAST_THRESHOLD_S
        step = _HOLD_FAST_SKIP if fast else 1
        interval = _HOLD_REPEAT_FAST_S if fast else _HOLD_REPEAT_NORMAL_S
        if self._repeat_accumulator < interval:
            return 0

        self._repeat_accumulator = 0.0
        return direction * step


def _held_nav_direction() -> int:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        return -1
    if any(keys[key] for key in _NAV_NEXT_KEYS):
        return 1
    return 0


def _nav_from_key_event(key: int) -> int | None:
    if key == pygame.K_LEFT:
        return -1
    if key in _NAV_NEXT_KEYS:
        return 1
    return None


def _is_key_repeat(event: pygame.event.Event) -> bool:
    return bool(getattr(event, "repeat", False))


@dataclass(frozen=True)
class _Navigate:
    advance: int = 0
    jump_to: int | None = None


class _GenerationJumpInput:
    """Text field to jump directly to a generation number."""

    _BOX_HEIGHT = 22
    _MAX_DIGITS = 5

    def __init__(self) -> None:
        self.text = ""
        self.active = False
        self.message = ""
        self._font = pygame.font.SysFont("consolas", 14)
        self._hint_font = pygame.font.SysFont("consolas", 11)
        self.rect = pygame.Rect(0, 0, 0, self._BOX_HEIGHT)
        self._label_y = 0

    def layout_below(self, network_bottom: int) -> None:
        """Position the compact input directly under the network output layer."""
        label_gap = 6
        self._label_y = network_bottom + _GEN_INPUT_GAP
        sample = self._font.render("9" * self._MAX_DIGITS, True, config.COLOR_TEXT)
        box_width = sample.get_width() + 12
        box_x = (config.PANEL_WIDTH - box_width) // 2
        self.rect = pygame.Rect(
            box_x,
            self._label_y + label_gap + 12,
            box_width,
            self._BOX_HEIGHT,
        )

    @property
    def bottom_y(self) -> int:
        """Lowest drawn pixel (including hint line)."""
        return self.rect.bottom + 16

    def handle_event(self, event: pygame.event.Event) -> tuple[bool, int | None]:
        """
        Handle pygame input for the jump box.

        Returns (consumed, jump_generation). consumed=True means the caller should
        not treat Esc / nav keys as global shortcuts for this event.
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if event.pos[0] >= config.PANEL_WIDTH:
                if self.active:
                    self.active = False
                return False, None
            clicked = self.rect.collidepoint(event.pos)
            if clicked:
                self.active = True
                self.message = ""
                return True, None
            if self.active:
                self.active = False
            return False, None

        if event.type != pygame.KEYDOWN:
            return False, None

        if event.key == pygame.K_g and not self.active and not _is_key_repeat(event):
            self.active = True
            self.message = ""
            return True, None

        if not self.active:
            return False, None

        if event.key == pygame.K_ESCAPE:
            self.active = False
            return True, None
        if event.key == pygame.K_RETURN:
            generation = self._parse()
            if generation is None:
                self.message = "Enter a generation number"
                return True, None
            self.active = False
            self.text = ""
            self.message = ""
            return True, generation
        if event.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
            self.message = ""
            return True, None
        if event.unicode.isdigit() and len(self.text) < self._MAX_DIGITS:
            self.text += event.unicode
            self.message = ""
            return True, None
        return True, None

    def set_message(self, message: str) -> None:
        self.message = message

    def draw(self, surface: pygame.Surface) -> None:
        label = self._hint_font.render("Go to gen", True, config.COLOR_TEXT_DIM)
        label_rect = label.get_rect(centerx=config.PANEL_WIDTH // 2, top=self._label_y)
        surface.blit(label, label_rect)

        border_color = config.COLOR_CONTROL_ACTIVE if self.active else config.COLOR_CONTROL_BORDER
        fill_color = config.COLOR_CONTROL_INACTIVE
        pygame.draw.rect(surface, fill_color, self.rect, border_radius=4)
        pygame.draw.rect(surface, border_color, self.rect, width=1, border_radius=4)

        display = self.text if self.text else ("…" if self.active else "")
        text_color = config.COLOR_TEXT if self.text or self.active else config.COLOR_TEXT_DIM
        text_surf = self._font.render(display, True, text_color)
        surface.blit(text_surf, text_surf.get_rect(center=self.rect.center))

        if self.message:
            msg = self._hint_font.render(self.message, True, config.COLOR_GAME_OVER)
            msg_rect = msg.get_rect(centerx=config.PANEL_WIDTH // 2, top=self.rect.bottom + 2)
            surface.blit(msg, msg_rect)

    def _parse(self) -> int | None:
        if not self.text:
            return None
        try:
            value = int(self.text)
        except ValueError:
            return None
        if value < 0:
            return None
        return value


def _generation_from_path(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])


def _index_for_generation(files: list[Path], generation: int) -> int | None:
    for index, path in enumerate(files):
        if _generation_from_path(path) == generation:
            return index
    return None


class ReplayViewer:
    """Cycles through saved per-generation best genomes, re-simulated live."""

    def __init__(self, replays_dir: Path, ticks_per_second: int | None = None) -> None:
        self._replays_dir = replays_dir
        self._files = sorted(replays_dir.glob("gen_*.npz"))
        if not self._files:
            raise FileNotFoundError(
                f"No gen_*.npz replays found in {replays_dir.resolve()}. Train first."
            )
        self._tick_interval = 1.0 / (ticks_per_second or config.TICKS_PER_SECOND)
        self._index = 0

    def run(self) -> None:
        screen, panel_surface, game_surface, clock = _init_display()
        gen_input = _GenerationJumpInput()
        running = True
        while running and self._files:
            session = self._start_session(
                self._files[self._index],
                panel_surface,
                game_surface,
            )
            result = self._play_session(
                screen,
                panel_surface,
                game_surface,
                clock,
                session,
                gen_input,
                auto_advance_on_death=True,
            )
            if result is None:
                running = False
            elif result.jump_to is not None:
                index = _index_for_generation(self._files, result.jump_to)
                if index is None:
                    gen_input.set_message(f"Gen {result.jump_to} not found")
                else:
                    self._index = index
            elif result.advance != 0:
                self._index = (self._index + result.advance) % len(self._files)

        pygame.quit()

    def _start_session(
        self,
        path: Path,
        panel_surface: pygame.Surface,
        game_surface: pygame.Surface,
    ) -> _ReplaySession:
        generation, score, genome, food_seed, grid_cols, grid_rows = _load_genome_file(path)
        grid = Grid(grid_cols, grid_rows)
        game = Game(grid, food_seed=food_seed)
        controller = AIController(game, NeuralNetwork.from_genome(genome))
        cell_size, offset_x, offset_y = board_layout(
            grid_cols,
            grid_rows,
            game_surface.get_width(),
            game_surface.get_height(),
        )
        renderer = GameRenderer(
            game_surface,
            game,
            cell_size=cell_size,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        controller.get_direction()
        return _ReplaySession(
            generation=generation,
            score=score,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            game=game,
            controller=controller,
            control_panel=ControlPanel(panel_surface),
            renderer=renderer,
        )

    def _play_session(
        self,
        screen: pygame.Surface,
        panel_surface: pygame.Surface,
        game_surface: pygame.Surface,
        clock: pygame.time.Clock,
        session: _ReplaySession,
        gen_input: _GenerationJumpInput,
        *,
        auto_advance_on_death: bool,
        waiting_for: int | None = None,
        training_active: bool = False,
    ) -> _Navigate | None:
        """
        Run one replay until the user navigates away or (optionally) auto-advances on death.

        Returns navigation intent, or None if the viewer should quit.
        """
        _set_caption(session.generation, session.score, session.grid_cols, session.grid_rows)
        navigate = _Navigate()
        dead_linger = 0.0
        running = True
        nav_hold = _NavHoldTracker()
        session.control_panel.draw(session.controller.last_snapshot, replay_mode=True)
        gen_input.layout_below(session.control_panel.network_bottom_y)

        while running and navigate.advance == 0 and navigate.jump_to is None:
            delta = clock.tick(config.RENDER_FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None

                consumed, jump_to = gen_input.handle_event(event)
                if jump_to is not None:
                    return _Navigate(jump_to=jump_to)
                if consumed:
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        continue
                    continue

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    if gen_input.active:
                        continue
                    nav_dir = _nav_from_key_event(event.key)
                    if nav_dir is not None and not _is_key_repeat(event):
                        navigate = _Navigate(advance=nav_hold.on_key_down(nav_dir))

            if navigate.advance == 0 and navigate.jump_to is None and not gen_input.active:
                offset = nav_hold.update(delta)
                if offset != 0:
                    navigate = _Navigate(advance=offset)

            if session.game.alive:
                session.tick_accumulator += delta
                while session.tick_accumulator >= self._tick_interval:
                    session.tick_accumulator -= self._tick_interval
                    direction = session.controller.get_direction()
                    session.game.tick(direction)
                    session.steps += 1
                    if not session.game.alive:
                        break
            elif auto_advance_on_death:
                dead_linger += delta
                if dead_linger >= 1.5:
                    navigate = _Navigate(advance=1)

            session.control_panel.draw(
                session.controller.last_snapshot, replay_mode=True
            )
            gen_input.layout_below(session.control_panel.network_bottom_y)
            gen_input.draw(panel_surface)
            _draw_replay_nav_bar(panel_surface)
            session.renderer.draw()
            if waiting_for is not None:
                _draw_status_overlay(
                    game_surface,
                    session.renderer,
                    _waiting_message(waiting_for, training_active),
                )
            elif not session.game.alive and training_active:
                _draw_status_overlay(
                    game_surface,
                    session.renderer,
                    _waiting_message(session.generation + 1, training_active=True),
                )
            _blit_frames(screen, panel_surface, game_surface)
            pygame.display.flip()

        return navigate


class LiveReplayViewer:
    """
    Watch training progress in real time.

    Polls ``replays/gen_XXXX.npz`` as each generation completes. When the snake dies,
    the death screen stays up until the next generation's replay file is written.
    """

    def __init__(
        self,
        replays_dir: Path,
        *,
        training_done: threading.Event | None = None,
        ticks_per_second: int | None = None,
    ) -> None:
        self._replays_dir = replays_dir
        self._training_done = training_done
        self._tick_interval = 1.0 / (ticks_per_second or config.TICKS_PER_SECOND)
        self._next_gen = 0
        self._latest_available = -1

    def run(self) -> None:
        screen, panel_surface, game_surface, clock = _init_display()
        pygame.display.set_caption("Genetic Snake - Live training watch")
        session: _ReplaySession | None = None
        running = True
        nav_hold = _NavHoldTracker()
        gen_input = _GenerationJumpInput()

        while running:
            self._refresh_latest_available()

            if session is None:
                path = _gen_path(self._replays_dir, self._next_gen)
                if path is not None:
                    session = self._start_session(path, panel_surface, game_surface)
                else:
                    running = self._draw_waiting_frame(
                        screen,
                        panel_surface,
                        game_surface,
                        clock,
                        gen_input,
                        waiting_for=self._next_gen,
                    )
                    continue

            delta = clock.tick(config.RENDER_FPS) / 1000.0
            advance = 0
            jump_to: int | None = None

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

                consumed, jump_generation = gen_input.handle_event(event)
                if jump_generation is not None:
                    jump_to = jump_generation
                    break
                if consumed:
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        continue
                    continue

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    if gen_input.active:
                        continue
                    nav_dir = _nav_from_key_event(event.key)
                    if nav_dir is not None and not _is_key_repeat(event):
                        advance = nav_hold.on_key_down(nav_dir)

            if jump_to is None and advance == 0 and not gen_input.active:
                advance = nav_hold.update(delta)

            if not running:
                break

            if jump_to is not None:
                path = _gen_path(self._replays_dir, jump_to)
                if path is None:
                    if jump_to > self._latest_available and self._is_training_active():
                        gen_input.set_message(f"Gen {jump_to} not ready yet")
                    else:
                        gen_input.set_message(f"Gen {jump_to} not found")
                else:
                    self._next_gen = jump_to
                    session = self._start_session(path, panel_surface, game_surface)
                    continue

            if advance != 0 and session is not None:
                if advance > 0:
                    target = min(session.generation + advance, self._latest_available)
                else:
                    target = max(0, session.generation + advance)
                if 0 <= target <= self._latest_available and target != session.generation:
                    self._next_gen = target
                    session = self._start_session(
                        _gen_path(self._replays_dir, target),
                        panel_surface,
                        game_surface,
                    )
                    continue

            training_active = self._is_training_active()

            if session.game.alive:
                session.tick_accumulator += delta
                while session.tick_accumulator >= self._tick_interval:
                    session.tick_accumulator -= self._tick_interval
                    direction = session.controller.get_direction()
                    session.game.tick(direction)
                    session.steps += 1
                    if not session.game.alive:
                        break
            elif _gen_path(self._replays_dir, session.generation + 1) is not None:
                self._next_gen = session.generation + 1
                session = self._start_session(
                    _gen_path(self._replays_dir, self._next_gen),
                    panel_surface,
                    game_surface,
                )
                continue

            session.control_panel.draw(
                session.controller.last_snapshot, replay_mode=True
            )
            gen_input.layout_below(session.control_panel.network_bottom_y)
            gen_input.draw(panel_surface)
            _draw_replay_nav_bar(panel_surface)
            session.renderer.draw()
            if not session.game.alive:
                if training_active and session.generation + 1 > self._latest_available:
                    subtitle = _waiting_message(session.generation + 1, training_active=True)
                elif not training_active and session.generation >= self._latest_available:
                    subtitle = "Training complete — Esc to quit"
                else:
                    subtitle = None
                if subtitle is not None:
                    _draw_status_overlay(game_surface, session.renderer, subtitle)

            status = "training" if training_active else "complete"
            pygame.display.set_caption(
                f"Genetic Snake - Live Gen {session.generation} "
                f"({session.grid_cols}x{session.grid_rows}, score {session.score}) [{status}]"
            )
            _blit_frames(screen, panel_surface, game_surface)
            pygame.display.flip()

        pygame.quit()

    def _is_training_active(self) -> bool:
        if self._training_done is None:
            return True
        return not self._training_done.is_set()

    def _refresh_latest_available(self) -> None:
        latest = -1
        for path in self._replays_dir.glob("gen_*.npz"):
            try:
                gen = int(path.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            latest = max(latest, gen)
        self._latest_available = latest

    def _start_session(
        self,
        path: Path,
        panel_surface: pygame.Surface,
        game_surface: pygame.Surface,
    ) -> _ReplaySession:
        generation, score, genome, food_seed, grid_cols, grid_rows = _load_genome_file(path)
        grid = Grid(grid_cols, grid_rows)
        game = Game(grid, food_seed=food_seed)
        controller = AIController(game, NeuralNetwork.from_genome(genome))
        cell_size, offset_x, offset_y = board_layout(
            grid_cols,
            grid_rows,
            game_surface.get_width(),
            game_surface.get_height(),
        )
        renderer = GameRenderer(
            game_surface,
            game,
            cell_size=cell_size,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        controller.get_direction()
        return _ReplaySession(
            generation=generation,
            score=score,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            game=game,
            controller=controller,
            control_panel=ControlPanel(panel_surface),
            renderer=renderer,
        )

    def _draw_waiting_frame(
        self,
        screen: pygame.Surface,
        panel_surface: pygame.Surface,
        game_surface: pygame.Surface,
        clock: pygame.time.Clock,
        gen_input: _GenerationJumpInput,
        *,
        waiting_for: int,
    ) -> bool:
        """Draw until gen 0 (or next) is ready. Returns False if the user quit."""
        jump_to: int | None = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            consumed, jump_generation = gen_input.handle_event(event)
            if jump_generation is not None:
                jump_to = jump_generation
                break
            if consumed:
                continue

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        if jump_to is not None:
            path = _gen_path(self._replays_dir, jump_to)
            if path is not None:
                self._next_gen = jump_to
            elif jump_to > self._latest_available and self._is_training_active():
                gen_input.set_message(f"Gen {jump_to} not ready yet")
            else:
                gen_input.set_message(f"Gen {jump_to} not found")

        clock.tick(config.RENDER_FPS)
        game_surface.fill(config.COLOR_BACKGROUND)
        _draw_centered_message(
            game_surface,
            "Waiting for training…",
            _waiting_message(waiting_for, self._is_training_active()),
        )
        panel_surface.fill(config.COLOR_PANEL)
        gen_input.layout_below(140)
        gen_input.draw(panel_surface)
        _draw_replay_nav_bar(panel_surface)
        status = "training" if self._is_training_active() else "complete"
        pygame.display.set_caption(f"Genetic Snake - Live watch [{status}]")
        _blit_frames(screen, panel_surface, game_surface)
        pygame.display.flip()
        return True


def _init_display() -> tuple[pygame.Surface, pygame.Surface, pygame.Surface, pygame.time.Clock]:
    pygame.init()
    pygame.display.set_caption("Genetic Snake - Replay")
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    panel_surface = pygame.Surface((config.PANEL_WIDTH, config.WINDOW_HEIGHT))
    game_surface = pygame.Surface((config.WINDOW_WIDTH - config.PANEL_WIDTH, config.WINDOW_HEIGHT))
    return screen, panel_surface, game_surface, clock


def _set_caption(generation: int, score: int, grid_cols: int, grid_rows: int) -> None:
    pygame.display.set_caption(
        f"Genetic Snake - Gen {generation} ({grid_cols}x{grid_rows}, score {score})"
    )


def _blit_frames(
    screen: pygame.Surface,
    panel_surface: pygame.Surface,
    game_surface: pygame.Surface,
) -> None:
    screen.blit(panel_surface, (0, 0))
    screen.blit(game_surface, (config.PANEL_WIDTH, 0))


def _waiting_message(next_generation: int, training_active: bool) -> str:
    if training_active:
        return f"Waiting for gen {next_generation}…"
    return "Training complete — Esc to quit"


def _draw_status_overlay(
    game_surface: pygame.Surface,
    renderer: GameRenderer,
    subtitle: str,
) -> None:
    font = pygame.font.SysFont("consolas", 20)
    text = font.render(subtitle, True, config.COLOR_TEXT)
    center_x = renderer._offset_x + renderer.board_width // 2
    center_y = renderer._offset_y + renderer.board_height // 2 + 52
    rect = text.get_rect(center=(center_x, center_y))
    game_surface.blit(text, rect)


def _draw_centered_message(
    surface: pygame.Surface,
    title: str,
    subtitle: str,
) -> None:
    title_font = pygame.font.SysFont("consolas", 28, bold=True)
    subtitle_font = pygame.font.SysFont("consolas", 20)
    center_x = surface.get_width() // 2
    center_y = surface.get_height() // 2
    title_surf = title_font.render(title, True, config.COLOR_TEXT)
    subtitle_surf = subtitle_font.render(subtitle, True, config.COLOR_TEXT)
    surface.blit(title_surf, title_surf.get_rect(center=(center_x, center_y - 16)))
    surface.blit(subtitle_surf, subtitle_surf.get_rect(center=(center_x, center_y + 20)))

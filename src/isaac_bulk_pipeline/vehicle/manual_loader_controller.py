"""Keyboard command state with no dependency on Isaac, Kit, or an articulation."""

from __future__ import annotations

from .vehicle_command import CommandSlewLimiter, VehicleCommand


class KeyboardCommandState:
    """Translate key press/release state into :class:`VehicleCommand`.

    The Isaac callback should call :meth:`handle_event` and do nothing else.
    In particular this object has no robot, stage, root-pose, or PhysX handle.
    """

    MOTION_KEYS = frozenset({"W", "S", "A", "D", "I", "K", "J", "L", "SPACE"})

    def __init__(self) -> None:
        self._pressed: set[str] = set()
        self._stopped = False

    @staticmethod
    def normalize_key(key: object) -> str:
        name = getattr(key, "name", key)
        text = str(name).split(".")[-1].strip().upper()
        aliases = {" ": "SPACE", "SPACEBAR": "SPACE", "ESC": "ESCAPE"}
        return aliases.get(text, text)

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def pressed_keys(self) -> frozenset[str]:
        return frozenset(self._pressed)

    def reset(self) -> None:
        self._pressed.clear()

    def handle_event(self, key: object, pressed: bool) -> bool:
        """Update key state and return whether the key is part of the contract."""

        name = self.normalize_key(key)
        if name == "R" and pressed:
            self.reset()
            return True
        if name == "ESCAPE" and pressed:
            self.reset()
            self._stopped = True
            return True
        if name not in self.MOTION_KEYS:
            return False
        if pressed:
            self._pressed.add(name)
        else:
            self._pressed.discard(name)
        return True

    @staticmethod
    def _axis(positive: str, negative: str, pressed: set[str]) -> float:
        return float((positive in pressed) - (negative in pressed))

    def command(self) -> VehicleCommand:
        return VehicleCommand(
            throttle=self._axis("W", "S", self._pressed),
            brake=float("SPACE" in self._pressed),
            steering=self._axis("A", "D", self._pressed),
            lift=self._axis("I", "K", self._pressed),
            bucket_curl=self._axis("J", "L", self._pressed),
        ).normalized()


class ManualLoaderController:
    """Small device-independent bridge from keyboard state to a physics tick."""

    def __init__(
        self,
        keyboard: KeyboardCommandState | None = None,
        slew_limiter: CommandSlewLimiter | None = None,
    ) -> None:
        self.keyboard = keyboard or KeyboardCommandState()
        self.slew_limiter = slew_limiter or CommandSlewLimiter()

    @property
    def stopped(self) -> bool:
        return self.keyboard.stopped

    def on_key_event(self, key: object, pressed: bool) -> bool:
        return self.keyboard.handle_event(key, pressed)

    def physics_step(self, dt_s: float) -> VehicleCommand:
        return self.slew_limiter.step(self.keyboard.command(), dt_s)

    def reset_commands(self) -> VehicleCommand:
        self.keyboard.reset()
        return self.slew_limiter.reset()

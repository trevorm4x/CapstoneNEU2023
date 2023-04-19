from typing import Tuple, Callable, Any, TypeVar, Optional
from djitellopy import Tello
from dataclasses import dataclass


tello = Tello()
tello.connect()
tello.takeoff()

tello.move_up(20)


tello.curve_xyz_speed(10,10,10,20,20,20)
tello.land()

VideoFeed = Any  # no idea what type this will actually be
InfoResult = TypeVar("InfoResult")
InfoStrategy = Callable[[VideoFeed], InfoResult]
FlightStrategy = Callable[[InfoResult], Tuple[float, float, float]]


@dataclass
class Drone:
    """
    Base class providing functionality for bootstrapping, piloting,
    and coordinating a drone on the same WiFi network.
    """
    id: int
    color: int
    ip: str
    flight_strategy: Optional[FlightStrategy]
    info_strategy: Optional[InfoStrategy]

    def connect(self): # -> VideoFeed?
        """
        Initialize connection to drone (get video feed?)
        
        Raises
        ------
        NetworkError (?)
        """
        # TODO: implement
        ...

    @property
    def video_feed(self) -> VideoFeed:
        # TODO: implement. currently no idea how this will work, subject to major change
        ...

    def command(self, command: str) -> None:
        """Directly issue a command to the drone"""
        # TODO: implement
        ...

    def set_speed(self, speed: Tuple(float, float, float)) -> None:
        """Set the (x, y, z) speed of drone"""
        # TODO: implement
        ...

    def adjust_speed(
        self,
        adjust: Tuple(float, float, float)
    ) -> Tuple(float, float, float):
        """Adjust the current speed by (dx, dy, dz) and return the new speed."""
        # TODO: implement
        ...

    def set_strategy(
        self,
        flight_strategy: FlightStrategy,
        info_strategy: InfoStrategy
    ) -> Tuple[float, float, float]:
        """
        Sets rules for getting information from the world, parsing it, and reacting to it.

        flight_strategy must be capable of accepting the result from info_strategy.
        """
        # TODO: implement
        self.flight_strategy = flight_strategy
        self.info_strategy = info_strategy
        test_result = self.flight_strategy(self.info_strategy(self.video_feed))
        return test_result
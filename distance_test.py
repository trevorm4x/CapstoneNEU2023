from djitellopy import Tello

import time


if __name__ == '__main__':

    angle = int(input("Enter a rotation angle: \n"))

    drone = Tello()
    drone.connect()
    drone.takeoff()

    drone.rotate_clockwise(angle)
    drone.land()
    drone.end()
